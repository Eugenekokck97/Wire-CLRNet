import argparse
import cv2
import torch
import numpy as np
import open3d as o3d
import os
import time
import matplotlib.pyplot as plt
from clrnet.models.registry import build_net
from clrnet.utils.net_utils import load_network
from clrnet.utils.config import Config
from mmcv.parallel import MMDataParallel
from clrnet.utils.visualization import imshow_lanes

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Wire Detection and Reconstruction')
    parser.add_argument('--config', default='config.py',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', default='model/best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--stereo', action='store_true',
                    help='Enable stereo depth computation')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization of point cloud')
    parser.add_argument('--input_dir', default='data',
                        help='Directory containing input images')
    parser.add_argument('--pred_dir', default='pred',
                    help='Directory containing predictions')
    return parser.parse_args()

def initialize_network(config_path, checkpoint_path):
    """Initialize and load the neural network model."""
    cfg = Config.fromfile(config_path)
    net = build_net(cfg)
    net = MMDataParallel(net, device_ids=[0]).cuda()
    load_network(net, checkpoint_path)
    net.eval()
    return net, cfg

def depth_estimation(diff, baseline=90, focal_length=1108.512):
    """Calculate depth from disparity."""
    return (baseline * focal_length) / (diff * 1000)

def pointcloud(u, v, z, fx=1108.512, fy=1108.512, cx=640, cy=360):
    """Convert 2D image points to 3D point cloud."""
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.vstack((x, y, z)).T

def feature_matching(top_lines, btm_lines, im1, im2, visualization=True):
    """Match features between top and bottom images to generate point cloud."""
    xyz = np.empty((0, 3))
    matched_lines2 = set()
    
    # Determine which image has fewer lines for efficient matching
    lines1, lines2, transform = (btm_lines, top_lines, True) if len(btm_lines) < len(top_lines) else (top_lines, btm_lines, False)
    
    for i, l1 in enumerate(lines1):
        min_diff = float('inf')
        best_diff = None
        chosen_ind = None
        best_line2 = None
        
        # Find best matching line
        for j, l2 in enumerate(lines2):
            if j in matched_lines2:
                continue
                
            common_x = np.intersect1d(l1[:, 0].astype(int), l2[:, 0].astype(int))
            line1 = l1[np.isin(l1[:, 0].astype(int), common_x)]
            line2 = l2[np.isin(l2[:, 0].astype(int), common_x)]
            
            diff = line2[:, 1] - line1[:, 1] if transform else line1[:, 1] - line2[:, 1]
            
            if len(diff) > 0 and np.mean(diff) > 0 and np.mean(diff) < min_diff:
                min_diff = np.mean(diff)
                best_diff = diff
                best_line2 = line2
                chosen_ind = j
        
        if chosen_ind is not None:
            matched_lines2.add(chosen_ind)
            line_color = np.random.randint(0, 256, 3)
            
            if transform:
                u, v = l1[:, 0], l1[:, 1]             
                im1[l1[:, 1].astype(int), l1[:, 0].astype(int)] = line_color
                im2[best_line2[:, 1].astype(int), best_line2[:, 0].astype(int)] = line_color
                    
            else:
                u, v = best_line2[:, 0], best_line2[:, 1]
                im1[best_line2[:, 1].astype(int), best_line2[:, 0].astype(int)] = line_color
                im2[l1[:, 1].astype(int), l1[:, 0].astype(int)] = line_color
            
            z = depth_estimation(best_diff)
            xyz = np.vstack((xyz, pointcloud(u, v, z)))
            print(f"Match {i} -> {chosen_ind}, Min Z: {np.min(z):.2f}")
    
    if visualization:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(im1)
        ax[0].set_title('Top image with matched lines')
        ax[1].imshow(im2)
        ax[1].set_title('Bottom image with matched lines')
        
    return xyz, im1, im2

def line_extraction(lanes, img_shape=(1279, 719)):
    """Extract and interpolate lane points from model predictions."""
    lanes_xys = []
    y_samples = np.arange(799, -1, -1)
    
    for lane in lanes:
        xys = [(x, y_samples[int(round(y))]) for x, y in lane if x >= 0 and y >= 0]
        if xys:
            lanes_xys.append(xys)
    
    lanes_xys.sort(key=lambda x: x[0][0])
    processed_lanes = []
    
    for xys in lanes_xys:
        X = np.array([x * img_shape[1] / 799 for x, _ in xys])
        Y = np.array([y * img_shape[0] / 799 for _, y in xys])
        processed_lanes.append(np.vstack((Y, X)).T)
    
    return processed_lanes

def main():
    """Main function to process images and generate point cloud."""
    args = parse_args()
    
    # Initialize network
    net, cfg = initialize_network(args.config, args.checkpoint)
    
    # Define input/output paths
    root = args.input_dir
    
    if args.stereo:
        im_bottom = os.path.join(root, 'btm.png')
        im_top = os.path.join(root, 'top.png')
        top_pred = os.path.join(root, 'top_pred.png')
        btm_pred = os.path.join(root, 'btm_pred.png')
        pc_pred = os.path.join(root, 'pointcloud.ply')
        
        # Load and preprocess images
        ori_btm_image = cv2.imread(im_bottom)
        ori_top_image = cv2.imread(im_top)
        
        if ori_btm_image is None or ori_top_image is None:
            print("Error: Failed to load input images")
            return
        
        # Resize and rotate images
        im_bottom = cv2.rotate(cv2.resize(ori_btm_image, (800, 800)), cv2.ROTATE_90_COUNTERCLOCKWISE)
        im_top = cv2.rotate(cv2.resize(ori_top_image, (800, 800)), cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Convert to tensors
        tensor_btm = torch.from_numpy(im_bottom).permute(2, 0, 1).float() / 255.0
        tensor_top = torch.from_numpy(im_top).permute(2, 0, 1).float() / 255.0
        tensor_btm = tensor_btm.unsqueeze(0).cuda()
        tensor_top = tensor_top.unsqueeze(0).cuda()
        
        # Perform inference
        with torch.no_grad():
            start_time = time.time()
            output_btm = net(tensor_btm)
            output_btm = net.module.heads.get_lanes(output_btm)
            output_top = net(tensor_top)
            output_top = net.module.heads.get_lanes(output_top)
            print(f"Inference time: {time.time() - start_time:.2f} seconds")
        
        # Extract lanes
        btm_lanes = line_extraction([lane.to_array2(cfg) for lanes in output_btm for lane in lanes])
        top_lanes = line_extraction([lane.to_array2(cfg) for lanes in output_top for lane in lanes])
        
        # Generate point cloud
        xyz, im1, im2 = feature_matching(top_lanes, btm_lanes, ori_btm_image, ori_top_image, visualization=args.visualize)
        
        # Save predictions
        cv2.imwrite(btm_pred, im1)
        cv2.imwrite(top_pred, im2)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        o3d.io.write_point_cloud(os.path.join(args.pred_dir, 'pc.ply'), pcd)
        
        # 3D Visualization (Optional)
        if args.visualize:
            o3d.visualization.draw_geometries([pcd])
            
    else:
        for id in os.listdir(root):
            im = os.path.join(root, id)
            im = cv2.imread(im)
            
            if im is None:
                print("Error: Failed to load input images")
                return
            
            im = cv2.rotate(cv2.resize(im, (800, 800)), cv2.ROTATE_90_COUNTERCLOCKWISE)
            tensor = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).cuda()
            with torch.no_grad():
                start_time = time.time()
                output = net(tensor)
                output = net.module.heads.get_lanes(output)
                print(f"Inference time: {time.time() - start_time:.2f} seconds")
                
            output = [line.to_array2(cfg) for lines in output for line in lines]
            
            imshow_lanes(im, output, show=args.visualize, width=2, out_file= os.path.join(args.pred_dir, str(id) + '.png')) 

if __name__ == '__main__':
    main()