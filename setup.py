# RXP-YOLO

Enhanced YOLOv10 for real-time object detection with EADNet, SEGFPN, and channel pruning.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from ultralytics import YOLOv10

model = YOLOv10('yolov10s.pt')
model.train(data='data.yaml', epochs=100)
```
