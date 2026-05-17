"""
Evaluation script for RXP-YOLO

Usage:
    python scripts/eval.py --weights weights/best.pt --data configs/data_visdrone.yaml
"""

import argparse
from ultralytics import YOLO
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate RXP-YOLO')
    parser.add_argument('--weights', '-w', type=str, required=True,
                        help='Model weights path')
    parser.add_argument('--data', '-d', type=str, required=True,
                        help='Dataset configuration')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Dataset split')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Loading model from: {args.weights}")
    model = YOLO(args.weights)
    
    print(f"Evaluating on {args.split} set...")
    results = model.val(
        data=args.data,
        batch=args.batch,
        imgsz=args.imgsz,
        split=args.split,
    )
    
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    print("="*50)


if __name__ == '__main__':
    main()
