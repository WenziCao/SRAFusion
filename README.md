# SRAFusion

"Semantic Region Adaptive Fusion of Infrared and Visible Images via Dual-DeepLab Guidance"

## 1. To Train
### 1.1 To preTrain visible segmentation
Run `python main.py -tort train -mode train_vsg`

### 1.2 To preTrain infrared segmentation
Run `python main.py -tort train -mode train_isg`

### 1.3 To train one_stage_fuse
Run `python main.py -tort train -mode train_sra_onestg`

### 1.4 To train two_stage_fuse
Run `python main.py -tort train -mode train_sra_twostg`

## 2. To Test
### 2.1 To test visible segmentation
Run `python main.py -tort test -mode test_VSeg`

### 2.2 To test infrared segmentation
Run `python main.py -tort test -mode test_ISeg`

### 2.3 To test fuse
One_stage_fuse: set MODEL.FUSION_NET.STG_TYPE:'SRA_OSTG'
Run `python main.py -tort test -mode test_fuse`

Two_stage_fuse: set MODEL.FUSION_NET.STG_TYPE:'SRA_TSTG'
Run `python main.py -tort test -mode test_fuse`

## 3. For quantitative evaluation
You can use the code provided by Dr. Tang [here](https://github.com/Linfeng-Tang/SeAFusion/tree/main/Evaluation)

## 4. Recommended Environment
- torch 1.13.1+cu116
- torchvision 0.14.1+cu116
- numpy 1.21.6
- pillow 9.4.0

## 5. Thanks to the following work for providing us with inspiration and help

- [https://github.com/VainF/DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch)

Citations:
- Tang, Linfeng, et al. "Image fusion in the loop of high-level vision tasks: A semantic-aware real-time infrared and visible image fusion network." Information Fusion 82 (2022): 28-42.
- Zhou, Huabing, et al. "Semantic-supervised infrared and visible image fusion via a dual-discriminator generative adversarial network." IEEE Transactions on Multimedia (2021).

