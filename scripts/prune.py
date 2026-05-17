"""
Channel Pruning Script for RXP-YOLO

Reduces model complexity by pruning channels based on BN weights.
Only effective channels are retained after training.

Usage:
    python scripts/prune.py --input weights/last.pt --output weights/pruned.pt --factor 0.1
"""

import argparse
import torch
from pathlib import Path

from ultralytics import YOLO
from ultralytics.nn.modules import *


def parse_args():
    parser = argparse.ArgumentParser(description='Prune RXP-YOLO model')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input model path (.pt)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output model path (.pt)')
    parser.add_argument('--factor', '-f', type=float, default=0.1,
                        help='Channel retention factor (0.0-1.0)')
    return parser.parse_args()


def _prune(c1, c2, threshold):
    """Prune conv layer based on BN weights."""
    wet = c1.bn.weight.data.detach()
    bis = c1.bn.bias.data.detach()
    li = []
    _threshold = threshold
    
    while len(li) < 8:
        li = torch.where(wet.abs() >= _threshold)[0]
        _threshold = _threshold * 0.5
    
    i = len(li)
    
    # Prune BN layer
    c1.bn.weight.data = wet[li]
    c1.bn.bias.data = bis[li]
    c1.bn.running_var.data = c1.bn.running_var.data[li]
    c1.bn.running_mean.data = c1.bn.running_mean.data[li]
    c1.bn.num_features = i
    
    # Prune conv layer
    c1.conv.weight.data = c1.conv.weight.data[li]
    c1.conv.out_channels = i
    if c1.conv.bias is not None:
        c1.conv.bias.data = c1.conv.bias.data[li]
    
    # Update subsequent layers
    if not isinstance(c2, list):
        c2 = [c2]
    for item in c2:
        if item is not None:
            if isinstance(item, Conv):
                conv = item.conv
            else:
                conv = item
            conv.in_channels = i
            if conv.weight.shape[1] < i:
                conv.groups = i
            else:
                conv.weight.data = conv.weight.data[:, li]


def _prune2(c1, c2, c3, threshold):
    """Prune sequential conv layers."""
    wet = c2.weight.data.detach()
    bis = c2.bias.data.detach()
    li = []
    _threshold = threshold
    
    while len(li) < 8:
        li = torch.where(wet.abs() >= _threshold)[0]
        _threshold = _threshold * 0.5
    
    i = len(li)
    
    # Prune BN layer
    c2.weight.data = wet[li]
    c2.bias.data = bis[li]
    c2.running_var.data = c2.running_var.data[li]
    c2.running_mean.data = c2.running_mean.data[li]
    c2.num_features = i
    
    # Update conv layers
    c1.weight.data = c1.weight.data[li]
    c1.out_channels = i
    if c1.bias is not None:
        c1.bias.data = c1.bias.data[li]
    
    if not isinstance(c3, list):
        c3 = [c3]
    for item in c3:
        if item is not None:
            conv = item
            conv.in_channels = i
            if conv.weight.shape[1] < i:
                conv.groups = i
            else:
                conv.weight.data = conv.weight.data[:, li]


def prune(m1, m2, threshold):
    """Prune module based on type."""
    if isinstance(m1, (C2f, SPPF, SCDown)):
        m1 = m1.cv2
    if not isinstance(m2, list):
        m2 = [m2]
    for i, item in enumerate(m2):
        if isinstance(item, (C2f, SPPF, SCDown)):
            m2[i] = item.cv1
    if m1 is not None:
        _prune(m1, m2, threshold)


def main():
    args = parse_args()
    
    print(f"Loading model from: {args.input}")
    yolo = YOLO(args.input)
    yolo.export(format="onnx")
    
    model = yolo.model
    ws = []
    bs = []
    
    # Collect BN weights
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            w = m.weight.abs().detach()
            b = m.bias.abs().detach()
            ws.append(w)
            bs.append(b)
    
    ws = torch.cat(ws)
    threshold = torch.sort(ws, descending=True)[0][int(len(ws) * args.factor)]
    
    print(f"Pruning with factor: {args.factor}")
    print(f"Threshold: {threshold:.6f}")
    
    # Prune bottleneck layers
    for _, m in model.named_modules():
        if isinstance(m, (Bottleneck,)):
            _prune(m.cv1, m.cv2, threshold)
    
    # Prune main backbone layers
    seq = model.model
    for i in [2, 3, 8]:
        prune(seq[i], seq[i + 1], threshold)
    
    # Prune EADNet layers
    se = seq[0].conv_layers[3]
    for j in [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42]:
        _prune2(se[j], se[j + 1], se[j + 3], threshold)
    
    # Prune detection head
    detect = seq[-1]
    for cv2, cv3 in zip(detect.cv2, detect.cv3):
        prune(cv2[0], cv2[1], threshold)
        prune(cv2[1], cv2[2], threshold)
        prune(cv3[1][1], cv3[2], threshold)
    
    # Enable gradients
    for _, p in yolo.model.named_parameters():
        p.requires_grad = True
    
    # Save pruned model
    os.makedirs(Path(args.output).parent, exist_ok=True)
    torch.save(yolo.ckpt, args.output)
    
    # Export to ONNX
    model = YOLO(args.output)
    model.export(format="onnx")
    
    print(f"Pruned model saved to: {args.output}")
    print("Done!")


if __name__ == '__main__':
    import os
    main()
