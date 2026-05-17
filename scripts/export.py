"""
Model export script for RXP-YOLO

Usage:
    python scripts/export.py --weights weights/best.pt --format onnx
"""

import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='Export RXP-YOLO model')
    parser.add_argument('--weights', '-w', type=str, required=True,
                        help='Model weights path')
    parser.add_argument('--format', '-f', type=str, default='onnx',
                        choices=['onnx', 'engine', 'torchscript', 'tflite'],
                        help='Export format')
    parser.add_argument('--opset', type=int, default=13,
                        help='ONNX opset version')
    parser.add_argument('--half', action='store_true',
                        help='Use FP16 half precision')
    parser.add_argument('--simplify', action='store_true', default=True,
                        help='Simplify ONNX model')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Loading model from: {args.weights}")
    model = YOLO(args.weights)
    
    print(f"Exporting to {args.format}...")
    
    if args.format == 'onnx':
        model.export(
            format='onnx',
            opset=args.opset,
            simplify=args.simplify
        )
    elif args.format == 'engine':
        model.export(
            format='engine',
            half=args.half,
            opset=args.opset,
            workspace=16
        )
    elif args.format == 'torchscript':
        model.export(format='torchscript')
    elif args.format == 'tflite':
        model.export(format='tflite', int8=True)
    
    print(f"Export completed!")
    print(f"Output: {model.ckpt_path.replace('.pt', f'.{args.format}')}")


if __name__ == '__main__':
    main()
