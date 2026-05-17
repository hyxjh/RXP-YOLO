

# RXP-YOLO: Lightweight Cross-Scale Feature Aggregation for Real-Time Small Object Detection in UAV Imagery

This repository contains the official implementation for the paper: **"RXP-YOLO: Lightweight Cross-Scale Feature Aggregation for Real-Time Small Object Detection in UAV Imagery"**. If you use this code, please consider citing the manuscript.

## Installation

```bash
conda create -n rxp python=3.12
conda activate rxp
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Datasets

We use VisDrone2019 and UAVDT. Organize them in YOLO format:

```
datasets/
├── VisDrone/
│   ├── images/train, val, test
│   └── labels/train, val, test
└── UAVDT/
    ├── images/train, test
    └── labels/train, test
```

Run the conversion scripts if needed:

```bash
python scripts/visdrone2yolo.py --root datasets/VisDrone
python scripts/uavdt2yolo.py --root datasets/UAVDT
```

## Training

Train from YOLOv10 pretrained weights (150 epochs, batch=8, SGD, lr=0.01, imgsz=640):

```bash
# RXPY-N (lightweight)
python train.py --model rxpy-n --data configs/visdrone.yaml --epochs 150 --batch 8 --imgsz 640 --device 0

# RXPY-S (standard)
python train.py --model rxpy-s --data configs/visdrone.yaml --epochs 150 --batch 8 --imgsz 640 --device 0
```

## Evaluation

Test set evaluation:

```bash
python val.py --weights weights/rxpy-s-visdrone.pt --data configs/visdrone.yaml --batch 1 --imgsz 640 --task test
```

Save PR curve data (101 recall points):

```bash
python val.py --weights weights/rxpy-s-visdrone.pt --data configs/visdrone.yaml --batch 1 --imgsz 640 --save-pr --task test
```

## Benchmark

Measure Params, GFLOPs and FPS on the same GPU:

```bash
python benchmark.py --weights weights/rxpy-s-visdrone.pt --imgsz 640 --device 0 --batch 1
```

