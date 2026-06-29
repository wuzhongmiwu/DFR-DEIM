# DFR-DEIM: Detail-Aware Feature Recalibration Network for Efficient Aerial Object Detection

This repository provides the official implementation of **DFR-DEIM**, a detail-aware feature recalibration network for efficient aerial object detection.

DFR-DEIM is designed for aerial remote sensing images with large scale variations, dense object distributions, complex backgrounds, and boundary-sensitive targets. The method improves multi-scale feature encoding by introducing a **Detail-Aware Feature Recalibration Pyramid Network (DFR-PN)**, together with **Scale-Aware Feature Enhancement (SFE)**, **Edge-Guided Reduction (EGR)**, and **Edge-Guided Upsampling (EGU)** modules.

## Highlights

* **DFR-PN**: A center-scale-guided feature recalibration pyramid network for enhanced cross-scale feature interaction.
* **SFE**: A scale-aware feature enhancement module with dynamic multi-kernel receptive-field selection.
* **EGR**: An edge-guided reduction module that preserves boundary cues during downsampling.
* **EGU**: An edge-guided upsampling module that enhances boundary-sensitive details during feature reconstruction.
* **Efficient detection**: DFR-DEIM achieves improved detection accuracy while maintaining compact model complexity.

## Main Results

| Dataset | mAP<sub>50:95</sub> | mAP<sub>50</sub> | mAP<sub>75</sub> | Params | GFLOPs |
| ------- | ------------------: | ---------------: | ---------------: | -----: | -----: |
| SIMD    |                69.6 |             84.6 |             81.8 | 12.76M |   37.3 |
| RSOD    |                63.9 |             93.8 |             69.2 | 12.76M |   37.3 |

All results are reported using an input resolution of `640 × 640`.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/wuzhongmiwu/DFR-DEIM
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

Please download the datasets from their official sources and organize them following the expected format in the configuration files.

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

If your dataset uses another annotation format, please convert it to the format required by the training configuration.

## Training

Train DFR-DEIM on SIMD:

```bash
python train.py -c configs/yaml/deim_dfine_hgnetv2_n_mg.yml
```


## Evaluation

Evaluate a trained model

```bash
python train.py -c configs/yaml/deim_dfine_hgnetv2_n_mg.yml --test-only
```

## Inference

Run inference on custom aerial images:

```bash
-c: Path of configuration yaml file
-r: Path of trained model weight file
--input: Input source, supports single image, single video, or an entire folder
--output: Directory path to save inference results
-t: Confidence threshold for detection filtering, default value is 0.2

```


## Citation

If this work is useful for your research, please consider citing:

```bibtex
@article{li2026dfrdeim,
  title={DFR-DEIM: Detail-Aware Feature Recalibration Network for Efficient Aerial Object Detection},
  author={Li, Jiajia and Huang, Yuan},
  journal={Journal of the Indian Society of Remote Sensing},
  year={2026},
  note={Under review}
}
```

## Acknowledgements

This implementation is developed for aerial object detection research. We thank the contributors of related open-source detection frameworks and the providers of the SIMD and RSOD datasets.

## License

This project is released under the license specified in the `LICENSE` file. Please also follow the licenses of the datasets and third-party code used in this repository.

## Contact

For questions or discussions, please contact:

```text
Jiajia Li
Email: slpersist@qq.com
```
