# DFR-DEIM: Detail-Aware Feature Recalibration Network for Efficient Aerial Object Detection

This repository provides the implementation of **DFR-DEIM**, a detail-aware feature recalibration network for efficient aerial object detection.

DFR-DEIM is designed for aerial remote sensing images with large scale variations, dense object distributions, complex backgrounds, and boundary-sensitive targets. The method improves multi-scale feature encoding by introducing a **Detail-Aware Feature Recalibration Pyramid Network (DFR-PN)**, together with **Scale-Aware Feature Enhancement (SFE)**, **Edge-Guided Reduction (EGR)**, and **Edge-Guided Upsampling (EGU)** modules.Compared with DEIM-S, DFR-DEIM introduces moderate computational overhead due to the detail-aware recalibration and edge-guided scale transformation modules. The parameter size increases from 10.18M to 12.76M and GFLOPs increase from 24.9 to 37.3, while mAP50:95 improves by 3.7 points on SIMD and 2.6 points on RSOD.

## Highlights

* **DFR-PN**: A center-scale-guided feature recalibration pyramid network for enhanced cross-scale feature interaction.
* **SFE**: A scale-aware feature enhancement module with dynamic multi-kernel receptive-field selection.
* **EGR**: An edge-guided reduction module that preserves boundary cues during downsampling.
* **EGU**: An edge-guided upsampling module that enhances boundary-sensitive details during feature reconstruction.
* **Efficient detection**: DFR-DEIM improves detection accuracy while maintaining compact model complexity.

## Main Results

| Dataset | mAP<sub>50:95</sub> | mAP<sub>50</sub> | mAP<sub>75</sub> | Params | GFLOPs |
| ------- | ------------------: | ---------------: | ---------------: | -----: | -----: |
| SIMD    |                69.6 |             84.6 |             81.8 | 12.76M |   37.3 |
| RSOD    |                63.9 |             93.8 |             69.2 | 12.76M |   37.3 |

All results are reported with an input resolution of `640 × 640`.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/wuzhongmiwu/DFR-DEIM.git
cd DFR-DEIM
```

### 2. Create a virtual environment

```bash
conda create -n dfr-deim python=3.10 -y
conda activate dfr-deim
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Recommended environment:

```text
Python 3.10
PyTorch 2.2.2
CUDA 12.1
TensorRT 10.11.0
```

## Dataset Preparation

Experiments are conducted on two public aerial remote sensing object detection datasets:

* **SIMD**
* **RSOD**

Please download the datasets from their official sources and organize them according to the paths specified in the configuration files.

A common dataset structure is:

```text
datasets/
├── SIMD/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── annotations/
│       ├── train.json
│       ├── val.json
│       └── test.json
└── RSOD/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── annotations/
        ├── train.json
        ├── val.json
        └── test.json
```

Please modify the dataset paths in the configuration files before training or evaluation.

## Training

Train DFR-DEIM

```bash
python train.py -c configs/yaml/deim_dfine_hgnetv2_n_mg.yml
```

If only one configuration file is provided, please modify the dataset path, number of classes, category names, and training schedule according to the target dataset.

## Evaluation

Evaluate a trained model:

```bash
python train.py \
  -c configs/yaml/deim_dfine_hgnetv2_n_mg.yml \
  -r path/to/checkpoint.pth \
  --test-only
```

For RSOD, replace the configuration file and checkpoint path accordingly.

## Inference

Run inference on custom aerial images:

```bash
python tools/inference/detect/torch_inf.py \
  -c configs/yaml/deim_dfine_hgnetv2_n_mg.yml \
  -r path/to/checkpoint.pth \
  --input path/to/images \
  --output runs/inference \
  -t 0.2
```

Arguments:

```text
-c        Path to the configuration YAML file.
-r        Path to the trained model checkpoint.
--input   Input source, including a single image, video, or image folder.
--output  Directory for saving inference results.
-t        Confidence threshold for detection filtering. The default value is 0.2.
```

## Checkpoints

The trained checkpoints can be provided upon reasonable request or released after acceptance.

| Model    | Dataset | Checkpoint  |
| -------- | ------- | ----------- |
| DFR-DEIM | SIMD    | Coming soon |
| DFR-DEIM | RSOD    | Coming soon |

## Citation

If this work is useful for your research, please consider citing:

```bibtex
@misc{li2026dfrdeim,
  title={DFR-DEIM: Detail-Aware Feature Recalibration Network for Efficient Aerial Object Detection},
  author={Huang, Yuan and Li, Jiajia},
  year={2026},
  note={Manuscript under review}
}
```

## Acknowledgements

This implementation is developed for aerial object detection research. We thank the contributors of related open-source detection frameworks and the providers of the SIMD and RSOD datasets.


## Contact

For questions or discussions, please contact:

```text
Jiajia Li
Email: slpersist@qq.com
```
