<div align="center">

# Trellis wire reconstruction by line anchor-based detection with vertical stereo vision

</div>


Pytorch implementation of the paper "[Trellis wire reconstruction by line anchor-based detection with vertical stereo vision](https://doi.org/10.1016/j.compag.2025.109948)".

## Introduction
![Wire-CLRNet](.github/wire-clrnet.png)
![Framework](.github/system_framework.png)
- A novel deep learning-based object-focused stereo vision system was developed.
- Line anchor-based method was proposed for wire 2D detection and reconstruction.
- Proposed method yields better accuracy and robustness in obscured settings.
- Proposed scheme recovers 3D information of obscured trellis wires in orchard.

## Installation

### Prerequisites
Only test on Ubuntu18.04 and 20.04 with:
- Python >= 3.8 (tested with Python3.8.19)
- PyTorch >= 1.6 (tested with Pytorch1.13.1)
- CUDA (tested with cuda12)
- Other dependencies described in `requirements.txt`

### Clone this repository
Clone this code to your workspace. 
```Shell
git clone https://github.com/Eugenekokck97/Wire-clrnet
```

### Create a conda virtual environment and activate it (conda is optional)

```Shell
conda create -n wire-clrnet python=3.8 -y
conda activate wire-clrnet
```

### Install dependencies

```Shell
# Install pytorch firstly, the cudatoolkit version should be same in your system.

conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# Or you can install via pip
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# Install python packages
python setup.py build develop
```

## Getting Started

### Inference
For wire detection only, run
```Shell
python main.py --input_dir data/images --visualize
```

For stereo reconstruction, run
```Shell
python main.py --input_dir data/stereo --stereo --visuaize
```

| Arguments | Description |
| :---  |  :---:   |
| config | Path to configuration file |
| checkpoint | Path to model checkpoint |
| stereo | Enable stereo depth computation |
| visualize | Enable visualization of point cloud |
| input_dir | Directory containing input images |
| pred_dir | Directory containing predictions |

## Results

## Citation

If our paper and code are beneficial to your work, please consider citing:
```
@article{kok2025trellis,
  title={Trellis wire reconstruction by line anchor-based detection with vertical stereo vision},
  author={Kok, Eugene and Liu, Tianhao and Chen, Chao},
  journal={Computers and Electronics in Agriculture},
  volume={231},
  pages={109948},
  year={2025},
  publisher={Elsevier}
}
```

## Acknowledgement
<!--ts-->
* [Turoad/clrnet](https://github.com/Turoad/CLRNet)
* [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
* [pytorch/vision](https://github.com/pytorch/vision)
<!--te-->
