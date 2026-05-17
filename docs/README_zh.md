# RXP-YOLO
Enhanced YOLOv10 for Real-Time Object Detection

## Dataset Preparation

### VisDrone Dataset

1. Download VisDrone dataset from [Official Website](https://github.com/VisDrone/VisDrone-Dataset)

2. Organize the dataset as follows:
```
datasets/
└── VisDrone/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

3. Update `configs/data_visdrone.yaml` with your paths:
```yaml
path: /path/to/your/datasets/VisDrone
train: images/train
val: images/val
test: images/test
```

### UAVDT Dataset

Similar structure as VisDrone. Update `configs/data_uavdt.yaml` accordingly.

## Training

### Basic Training

```bash
python scripts/train.py --data configs/data_visdrone.yaml --epochs 150
```

### Advanced Training

```bash
python scripts/train.py \
    --data configs/data_visdrone.yaml \
    --weights weights/yolov10s.pt \
    --epochs 150 \
    --batch 16 \
    --imgsz 640 \
    --device 0
```

## Pruning

After training, prune the model:

```bash
python scripts/prune.py \
    --input runs/train/exp/weights/last.pt \
    --output runs/pruned/weights/prune.pt \
    --factor 0.1
```

The `factor` parameter controls the channel retention rate (0.0-1.0).

## Evaluation

```bash
python scripts/eval.py \
    --weights runs/train/exp/weights/best.pt \
    --data configs/data_visdrone.yaml
```

## Model Export

Export to ONNX:
```bash
from ultralytics import YOLO
model = YOLO('weights/best.pt')
model.export(format='onnx')
```

Export to TensorRT:
```bash
model.export(format='engine', half=True, opset=13)
```
