import torch
from clrnet.models.losses.lineiou_loss import line_iou
from clrnet.models.losses.laneiou_loss import LaneIoULoss


def distance_cost(predictions, targets, img_w):
    """
    repeat predictions and targets to generate all combinations
    use the abs distance as the new distance cost
    """
    num_priors = predictions.shape[0]
    num_targets = targets.shape[0]

    predictions = torch.repeat_interleave(
        predictions, num_targets, dim=0
    )[...,
      6:]  # repeat_interleave'ing [a, b] 2 times gives [a, a, b, b] ((np + nt) * 78)

    targets = torch.cat(
        num_priors *
        [targets])[...,
                   6:]  # applying this 2 times on [c, d] gives [c, d, c, d]

    invalid_masks = (targets < 0) | (targets >= img_w)
    lengths = (~invalid_masks).sum(dim=1)
    distances = torch.abs((targets - predictions))
    distances[invalid_masks] = 0.
    distances = distances.sum(dim=1) / (lengths.float() + 1e-9)
    distances = distances.view(num_priors, num_targets)

    return distances


def focal_cost(cls_pred, gt_labels, alpha=0.25, gamma=2, eps=1e-12):
    """
    Args:
        cls_pred (Tensor): Predicted classification logits, shape
            [num_query, num_class].
        gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

    Returns:
        torch.Tensor: cls_cost value
    """
    cls_pred = cls_pred.sigmoid()
    neg_cost = -(1 - cls_pred + eps).log() * (1 - alpha) * cls_pred.pow(gamma)
    pos_cost = -(cls_pred + eps).log() * alpha * (1 - cls_pred).pow(gamma)
    cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
    return cls_cost

class CLRNetIoUCost:
    def __init__(self, weight=1.0, lane_width=15 / 800):
        """
        LineIoU cost employed in CLRNet.
        Adapted from:
        https://github.com/Turoad/CLRNet/blob/main/clrnet/models/losses/lineiou_loss.py
        Args:
            weight (float): cost weight.
            lane_width (float): half virtual lane width.
        """
        self.weight = weight
        self.lane_width = lane_width

    def _calc_over_union(self, pred, target, pred_width, target_width):
        """
        Calculate the line iou value between predictions and targets
        Args:
            pred: lane predictions, shape: (Nlp, Nr), relative coordinate
            target: ground truth, shape: (Nlt, Nr), relative coordinate
            pred_width (torch.Tensor): virtual lane half-widths for prediction at pre-defined rows, shape (Nl, Nr).
            target_width (torch.Tensor): virtual lane half-widths for GT at pre-defined rows, shape (Nl, Nr).
        Returns:
            torch.Tensor: calculated overlap, shape (Nlp, Nlt, Nr).
            torch.Tensor: calculated union, shape (Nlp, Nlt, Nr).
        Nlp, Nlt: number of prediction and target lanes, Nr: number of rows.
        """
        px1 = pred - pred_width
        px2 = pred + pred_width
        tx1 = target - target_width
        tx2 = target + target_width

        ovr = torch.min(px2[:, None, :], tx2[None, ...]) - torch.max(
            px1[:, None, :], tx1[None, ...]
        )
        union = torch.max(px2[:, None, :], tx2[None, ...]) - torch.min(
            px1[:, None, :], tx1[None, ...]
        )
        return ovr, union

    def __call__(self, pred, target):
        """
        Calculate the line iou value between predictions and targets
        Args:
            pred: lane predictions, shape: (Nlp, Nr), relative coordinate
            target: ground truth, shape: (Nlt, Nr), relative coordinate
        Returns:
            torch.Tensor: calculated IoU matrix, shape (Nlp, Nlt)
        Nlp, Nlt: number of prediction and target lanes, Nr: number of rows.
        """
        ovr, union = self._calc_over_union(
            pred, target, self.lane_width, self.lane_width
        )
        num_pred = pred.shape[0]
        invalid_mask = target.repeat(num_pred, 1, 1)
        invalid_masks = (invalid_mask < 0) | (invalid_mask >= 1.0)
        ovr[invalid_masks] = 0.0
        union[invalid_masks] = 0.0
        iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
        return iou * self.weight

class LaneIoUCost(CLRNetIoUCost, LaneIoULoss):
    def __init__(
        self,
        weight=1.0,
        lane_width=7.5 / 800,
        img_h=320,
        img_w=1640,
        use_pred_start_end=False,
        use_giou=True,
    ):
        """
        Angle- and length-aware LaneIoU employed in CLRerNet.
        Args:
            weight (float): cost weight.
            lane_width (float): half virtual lane width.
            use_pred_start_end (bool): apply the lane range (in horizon indices) for pred lanes
            use_giou (bool): GIoU-style calculation that allow negative overlap
               when the lanes are separated
        """
        super(LaneIoUCost, self).__init__(weight, lane_width)
        self.use_pred_start_end = use_pred_start_end
        self.use_giou = use_giou
        self.max_dx = 1e4
        self.img_h = img_h
        self.img_w = img_w

    @staticmethod
    def _set_invalid_with_start_end(
        pred, target, ovr, union, start, end, pred_width, target_width
    ):
        """Set invalid rows for predictions and targets and modify overlaps and unions,
        with using start and end points of prediction lanes.

        Args:
            pred: lane predictions, shape: (Nlp, Nr), relative coordinate
            target: ground truth, shape: (Nlt, Nr), relative coordinate
            ovr (torch.Tensor): calculated overlap, shape (Nlp, Nlt, Nr).
            union (torch.Tensor): calculated union, shape (Nlp, Nlt, Nr).
            start (torch.Tensor): start row indices of predictions, shape (Nlp).
            end (torch.Tensor): end row indices of predictions, shape (Nlp).
            pred_width (torch.Tensor): virtual lane half-widths for prediction at pre-defined rows, shape (Nlp, Nr).
            target_width (torch.Tensor): virtual lane half-widths for GT at pre-defined rows, shape (Nlt, Nr).

        Returns:
            torch.Tensor: calculated overlap, shape (Nlp, Nlt, Nr).
            torch.Tensor: calculated union, shape (Nlp, Nlt, Nr).
        Nlp, Nlt: number of prediction and target lanes, Nr: number of rows.
        """
        num_gt = target.shape[0]
        pred_mask = pred.repeat(num_gt, 1, 1).permute(1, 0, 2)
        invalid_mask_pred = (pred_mask < 0) | (pred_mask >= 1.0)
        target_mask = target.repeat(pred.shape[0], 1, 1)
        invalid_mask_gt = (target_mask < 0) | (target_mask >= 1.0)

        # set invalid-pred region using start and end
        assert start is not None and end is not None
        yind = torch.ones_like(invalid_mask_pred) * torch.arange(
            0, pred.shape[-1]
        ).float().to(pred.device)
        h = pred.shape[-1] - 1
        start_idx = (start * h).long().view(-1, 1, 1)
        end_idx = (end * h).long().view(-1, 1, 1)
        invalid_mask_pred = invalid_mask_pred | (yind < start_idx) | (yind >= end_idx)

        # set ovr and union to zero at horizon lines where either pred or gt is missing
        invalid_mask_pred_gt = invalid_mask_pred | invalid_mask_gt
        ovr[invalid_mask_pred_gt] = 0
        union[invalid_mask_pred_gt] = 0

        # calculate virtual unions for pred-only or target-only horizon lines
        union_sep_target = target_width.repeat(pred.shape[0], 1, 1) * 2
        union_sep_pred = pred_width.repeat(num_gt, 1, 1).permute(1, 0, 2) * 2
        union[invalid_mask_pred_gt & ~invalid_mask_pred] += union_sep_pred[
            invalid_mask_pred_gt & ~invalid_mask_pred
        ]
        union[invalid_mask_pred_gt & ~invalid_mask_gt] += union_sep_target[
            invalid_mask_pred_gt & ~invalid_mask_gt
        ]
        return ovr, union

    @staticmethod
    def _set_invalid_without_start_end(pred, target, ovr, union):
        """Set invalid rows for predictions and targets and modify overlaps and unions,
        without using start and end points of prediction lanes.

        Args:
            pred: lane predictions, shape: (Nlp, Nr), relative coordinate
            target: ground truth, shape: (Nlt, Nr), relative coordinate
            ovr (torch.Tensor): calculated overlap, shape (Nlp, Nlt, Nr).
            union (torch.Tensor): calculated union, shape (Nlp, Nlt, Nr).

        Returns:
            torch.Tensor: calculated overlap, shape (Nlp, Nlt, Nr).
            torch.Tensor: calculated union, shape (Nlp, Nlt, Nr).
        Nlp, Nlt: number of prediction and target lanes, Nr: number of rows.
        """
        target_mask = target.repeat(pred.shape[0], 1, 1)
        invalid_mask_gt = (target_mask < 0) | (target_mask >= 1.0)
        ovr[invalid_mask_gt] = 0.0
        union[invalid_mask_gt] = 0.0
        return ovr, union

    def __call__(self, pred, target, start=None, end=None):
        """
        Calculate the line iou value between predictions and targets
        Args:
            pred: lane predictions, shape: (Nlp, Nr), relative coordinate.
            target: ground truth, shape: (Nlt, Nr), relative coordinate.
        Returns:
            torch.Tensor: calculated IoU matrix, shape (Nlp, Nlt)
        Nlp, Nlt: number of prediction and target lanes, Nr: number of rows.
        """
        pred_width, target_width = self._calc_lane_width(pred, target)
        ovr, union = self._calc_over_union(pred, target, pred_width, target_width)
        if self.use_pred_start_end is True:
            ovr, union = self._set_invalid_with_start_end(
                pred, target, ovr, union, start, end, pred_width, target_width
            )
        else:
            ovr, union = self._set_invalid_without_start_end(pred, target, ovr, union)
        iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
        return iou * self.weight

def dynamic_k_assign(cost, pair_wise_ious):
    """
    Assign grouth truths with priors dynamically.

    Args:
        cost: the assign cost.
        pair_wise_ious: iou of grouth truth and priors.

    Returns:
        prior_idx: the index of assigned prior.
        gt_idx: the corresponding ground truth index.
    """
    matching_matrix = torch.zeros_like(cost)
    ious_matrix = pair_wise_ious
    ious_matrix[ious_matrix < 0] = 0.
    n_candidate_k = 4
    topk_ious, _ = torch.topk(ious_matrix, n_candidate_k, dim=0)
    dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
    num_gt = cost.shape[1]
    for gt_idx in range(num_gt):
        _, pos_idx = torch.topk(cost[:, gt_idx],
                                k=dynamic_ks[gt_idx].item(),
                                largest=False)
        matching_matrix[pos_idx, gt_idx] = 1.0
    del topk_ious, dynamic_ks, pos_idx

    matched_gt = matching_matrix.sum(1)
    if (matched_gt > 1).sum() > 0:
        _, cost_argmin = torch.min(cost[matched_gt > 1, :], dim=1)
        matching_matrix[matched_gt > 1, 0] *= 0.0
        matching_matrix[matched_gt > 1, cost_argmin] = 1.0

    prior_idx = matching_matrix.sum(1).nonzero()
    gt_idx = matching_matrix[prior_idx].argmax(-1)
    return prior_idx.flatten(), gt_idx.flatten()

def _clrnet_cost(predictions, targets, img_w, img_h, distance_cost_weight=3., cls_cost_weight=1.):
    
    #distances cost
    distances_score = distance_cost(predictions, targets, img_w)
    distances_score = 1 - (distances_score / torch.max(distances_score)
                           ) + 1e-2  # normalize the distance

    # classification cost
    cls_score = focal_cost(predictions[:, :2], targets[:, 1].long())
    num_priors = predictions.shape[0]
    num_targets = targets.shape[0]

    target_start_xys = targets[:, 2:4]  # num_targets, 2
    target_start_xys[..., 0] *= (img_h - 1)
    prediction_start_xys = predictions[:, 2:4]
    prediction_start_xys[..., 0] *= (img_h - 1)

    start_xys_score = torch.cdist(prediction_start_xys, target_start_xys,
                                  p=2).reshape(num_priors, num_targets)
    start_xys_score = (1 - start_xys_score / torch.max(start_xys_score)) + 1e-2

    target_thetas = targets[:, 4].unsqueeze(-1)
    theta_score = torch.cdist(predictions[:, 4].unsqueeze(-1),
                              target_thetas,
                              p=1).reshape(num_priors, num_targets) * 180
    theta_score = (1 - theta_score / torch.max(theta_score)) + 1e-2

    cost = -(distances_score * start_xys_score * theta_score
             )**2 * distance_cost_weight + cls_score * cls_cost_weight
    
    return cost

def _clrernet_cost(predictions, targets, pred_xs, target_xs, img_w=320, img_h=1640, use_pred_length_for_iou = True, distance_cost_weight = 3):
    """_summary_

    Args:
        predictions (Dict[torch.Trnsor]): predictions predicted by each stage, including:
            cls_logits: shape (Np, 2), anchor_params: shape (Np, 3),
            lengths: shape (Np, 1) and xs: shape (Np, Nr).
        targets (torch.Tensor): lane targets, shape: (Ng, 6+Nr).
            The first 6 elements are classification targets (2 ch), anchor starting point xy (2 ch),
            anchor theta (1ch) and anchor length (1ch).
        pred_xs (torch.Tensor): predicted x-coordinates on the predefined rows, shape (Np, Nr).
        target_xs (torch.Tensor): GT x-coordinates on the predefined rows, shape (Ng, Nr).

    Returns:
        torch.Tensor: cost matrix, shape (Np, Ng).
    Np: number of priors (anchors), Ng: number of GT lanes, Nr: number of rows.
    """
    start = end = None
    if use_pred_length_for_iou:
        y0 = predictions[:, 2:5][:, 0].detach().clone()
        length =  predictions[... , 5].detach().clone()
        start = (1 - y0).clamp(min=0, max=1)
        end = (start + length).clamp(min=0, max=1)
        
    laneiou_cost = LaneIoUCost(lane_width=30 / 800, img_w=img_w, img_h=img_h, use_pred_start_end=True)
    iou_cost = laneiou_cost(
        pred_xs,
        target_xs,
        start,
        end,
    )
    iou_score = 1 - (1 - iou_cost) / torch.max(1 - iou_cost) + 1e-2
    # classification cost
    cls_score = focal_cost(
        predictions[:, :2].detach().clone(), targets[:, 1].long()
    )
    cost = -iou_score * distance_cost_weight + cls_score
    return cost

def assign(
    predictions,
    targets,
    img_w,
    img_h,
    distance_cost_weight=3.,
    cls_cost_weight=1.,
    cost_combination = 0
):
    
    predictions = predictions.detach().clone()
    predictions[:, 3] *= (img_w - 1)
    predictions[:, 6:] *= (img_w - 1)
    targets = targets.detach().clone()
    
    if cost_combination == 0:
        cost = _clrnet_cost(predictions, targets, img_w, img_h, distance_cost_weight, cls_cost_weight)
        iou = line_iou(predictions[..., 6:], targets[..., 6:], img_w, aligned=False)
        
    elif cost_combination == 1:
        pred_xs, target_xs = predictions[..., 6:], targets[..., 6:]
        
        cost = _clrernet_cost(predictions, targets, pred_xs, target_xs, img_w=img_w, img_h=img_h)
        laneiou_cost = LaneIoUCost(img_w=img_w, img_h=img_h)
        iou = laneiou_cost(pred_xs, target_xs)
    else:
        raise NotImplementedError(
            f"cost_combination {cost_combination} is not implemented!"
        )
        
    matched_row_inds, matched_col_inds = dynamic_k_assign(cost, iou)

    return matched_row_inds, matched_col_inds



# def assign(
#     predictions,
#     targets,
#     img_w,
#     img_h,
#     distance_cost_weight=3.,
#     cls_cost_weight=1.,
# ):
#     '''
#     computes dynamicly matching based on the cost, including cls cost and lane similarity cost
#     Args:
#         predictions (Tensor): predictions predicted by each stage, shape: (num_priors, 78)
#         targets (Tensor): lane targets, shape: (num_targets, 78)
#     return:
#         matched_row_inds (Tensor): matched predictions, shape: (num_targets)
#         matched_col_inds (Tensor): matched targets, shape: (num_targets)
#     '''
#     predictions = predictions.detach().clone()
#     predictions[:, 3] *= (img_w - 1)
#     predictions[:, 6:] *= (img_w - 1)
#     targets = targets.detach().clone()

#     # distances cost
#     distances_score = distance_cost(predictions, targets, img_w)
#     distances_score = 1 - (distances_score / torch.max(distances_score)
#                            ) + 1e-2  # normalize the distance

#     # classification cost
#     cls_score = focal_cost(predictions[:, :2], targets[:, 1].long())
#     num_priors = predictions.shape[0]
#     num_targets = targets.shape[0]

#     target_start_xys = targets[:, 2:4]  # num_targets, 2
#     target_start_xys[..., 0] *= (img_h - 1)
#     prediction_start_xys = predictions[:, 2:4]
#     prediction_start_xys[..., 0] *= (img_h - 1)

#     start_xys_score = torch.cdist(prediction_start_xys, target_start_xys,
#                                   p=2).reshape(num_priors, num_targets)
#     start_xys_score = (1 - start_xys_score / torch.max(start_xys_score)) + 1e-2

#     target_thetas = targets[:, 4].unsqueeze(-1)
#     theta_score = torch.cdist(predictions[:, 4].unsqueeze(-1),
#                               target_thetas,
#                               p=1).reshape(num_priors, num_targets) * 180
#     theta_score = (1 - theta_score / torch.max(theta_score)) + 1e-2

#     cost = -(distances_score * start_xys_score * theta_score
#              )**2 * distance_cost_weight + cls_score * c

#     iou = line_iou(predictions[..., 6:], targets[..., 6:], img_w, aligned=False)
#     matched_row_inds, matched_col_inds = dynamic_k_assign(cost, iou)

#     return matched_row_inds, matched_col_inds

