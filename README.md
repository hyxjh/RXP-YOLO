

# RXP-YOLO: Lightweight Cross-Scale Feature Aggregation for Real-Time Small Object Detection in UAV Aerial Imagery

This repository contains the official implementation for the manuscript: **"RXP-YOLO: Lightweight Cross-Scale Feature Aggregation for Real-Time Small Object Detection in UAV Imagery"**. 

## Installation

```bash
conda create -n rxp python=3.12
conda activate rxp
pip install "torch>=2.0.0" "torchvision>=0.15.0" --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Datasets

We evaluate on VisDrone2019 and UAVDT. Organize them in YOLO format:

```
datasets/
├── VisDrone/
│   ├── images/train, val, test
│   └── labels/train, val, test
└── UAVDT/
    ├── images/train, test
    └── labels/train, test
```

## Model Weights

Pretrained weights for reproducing the main results are available at the following permanent link (DOI):

- **RXPY-S (VisDrone):** [Zenodo DOI link] (placeholder)
- **RXPY-S (UAVDT):** [Zenodo DOI link] (placeholder)

Download and place the `.pt` files under `weights/` before running evaluation.

## Key Modules

| Module | Description |
|--------|-------------|
| **EADNet** | Edge-Aware Dual Network for enhancing small-object boundary information |
| **SEGFPN** | Squeeze-and-Excitation Guided Feature Pyramid Network for cross-scale feature aggregation |
| **Entropy-Guided Pruning** | Structural re-parameterization with BN-weight based channel pruning for lightweight deployment |

## Training

Train from YOLOv10 pretrained weights (150 epochs, batch=8, SGD, lr=0.01, imgsz=640):

```bash
# RXPY-N (lightweight)
python scripts/train.py --model rxpy-n --data configs/data_visdrone.yaml --epochs 150 --batch 8 --imgsz 640 --device 0

# RXPY-S (standard)
python scripts/train.py --model rxpy-s --data configs/data_visdrone.yaml --epochs 150 --batch 8 --imgsz 640 --device 0
```

## Evaluation

Test set evaluation:

```bash
python scripts/eval.py --weights weights/rxpy-s-visdrone.pt --data configs/data_visdrone.yaml --batch 1 --imgsz 640 --task test
```

Save PR curve data (101 recall points):

```bash
python scripts/eval.py --weights weights/rxpy-s-visdrone.pt --data configs/data_visdrone.yaml --batch 1 --imgsz 640 --save-pr --task test
```

## Channel Pruning

Apply entropy-guided channel pruning to further reduce model size after training:

```bash
python scripts/prune.py --input weights/last.pt --output weights/pruned.pt --factor 0.1
```

## Model Export

Export the trained model to ONNX or TensorRT for deployment:

```bash
# ONNX export
python scripts/export.py --weights weights/rxpy-s-visdrone.pt --format onnx

# TensorRT export
python scripts/export.py --weights weights/rxpy-s-visdrone.pt --format engine --half
```

## Project Structure

```
RXP-YOLO/
├── configs/              # Dataset and model configs
│   ├── data_visdrone.yaml
│   ├── data_uavdt.yaml
│   └── yolov10s_eos.yaml
├── rxp_yolo/nn/modules/  # EADNet, SEGFPN
├── scripts/              # Training, evaluation, pruning, export
│   ├── train.py
│   ├── eval.py
│   ├── prune.py
│   └── export.py
├── datasets/             # Dataset root (not included)
├── weights/              # Pretrained / trained weights
└── requirements.txt
```

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0)
