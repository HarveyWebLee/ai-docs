"""
用 PyTorch 张量计算 IoU：文档案例 交集=60、并集=120 → IoU=0.5。

对应文档：docs/06-计算机视觉/004-目标检测与图像分割.md

运行：python code/06-计算机视觉/004-目标检测与图像分割/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch


def box_iou_area(pred_area: torch.Tensor, gt_area: torch.Tensor, inter: torch.Tensor) -> torch.Tensor:
    """IoU = 交集 / 并集，并集 = |P| + |G| - 交集。"""
    union = pred_area + gt_area - inter
    return inter / union


def main() -> None:
    seed_all(42)
    dev = device()

    # 文档数值：P 面积 80，G 面积 100，交集 60 → 并集 120
    pred_area = torch.tensor(80.0, device=dev)
    gt_area = torch.tensor(100.0, device=dev)
    intersection = torch.tensor(60.0, device=dev)

    union = pred_area + gt_area - intersection
    iou = box_iou_area(pred_area, gt_area, intersection)

    print("目标检测 IoU 计算（文档案例）")
    print(f"预测框面积 P = {pred_area.item():.0f}")
    print(f"真实框面积 G = {gt_area.item():.0f}")
    print(f"交集面积     = {intersection.item():.0f}")
    print(f"并集面积     = {union.item():.0f}  (= 80+100-60)")
    print(f"IoU          = {intersection.item():.0f}/{union.item():.0f} = {iou.item():.2f}")
    print(f"\nIoU ≥ 0.5 判为 TP：{(iou >= 0.5).item()}")


if __name__ == "__main__":
    main()
