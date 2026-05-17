"""
Training script for RXP-YOLO

Usage:
    python scripts/train.py --data configs/data_visdrone.yaml --epochs 150
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLOv10
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Train RXP-YOLO')
    parser.add_argument('--data', type=str, default='configs/data_visdrone.yaml',
                        help='Dataset configuration file')
    parser.add_argument('--model', type=str, default='configs/yolov10s_eos.yaml',
                        help='Model configuration file')
    parser.add_argument('--weights', type=str, default='',
                        help='Pretrained weights path')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device')
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Project directory')
    parser.add_argument('--name', type=str, default='exp',
                        help='Experiment name')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load model
    if args.weights:
        print(f"Loading pretrained weights from: {args.weights}")
        model = YOLOv10(args.weights)
    else:
        print(f"Loading model from: {args.model}")
        model = YOLOv10(args.model)
    
    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        optimizer='SGD',
        amp=False,
        workers=8,
    )
    
    print("\nTraining completed!")
    print(f"Results saved to: {args.project}/{args.name}")


if __name__ == '__main__':
    main()
