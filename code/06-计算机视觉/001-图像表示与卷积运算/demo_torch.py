"""
用 nn.Conv2d 实现竖直 Sobel 边缘检测，验证卷积「小模板在图上滑动」。

对应文档：docs/06-计算机视觉/001-图像表示与卷积运算.md

运行：python code/06-计算机视觉/001-图像表示与卷积运算/demo_torch.py
"""

import sys
from pathlib import Path

_code_root = next(p for p in Path(__file__).resolve().parents if p.name == "code")
sys.path.insert(0, str(_code_root))
from _common.torch_utils import device, seed_all

import torch
import torch.nn as nn


def main() -> None:
    seed_all(42)
    dev = device()

    # 5×5 灰度图：左暗右亮 → 竖直边缘在中间
    image = torch.zeros(1, 1, 5, 5, device=dev)
    image[:, :, :, :2] = 0.0
    image[:, :, :, 2:] = 1.0

    # 竖直 Sobel 核：检测左右亮度差（竖直边）
    sobel_v = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=dev,
    )
    conv = nn.Conv2d(1, 1, kernel_size=3, bias=False)
    conv.weight.data = sobel_v.unsqueeze(1)
    conv.eval()

    with torch.no_grad():
        out = conv(image)

    print("输入 5×5（左暗右亮，中间竖直边缘）：")
    print(image.squeeze().tolist())
    print("\n竖直 Sobel 卷积输出 3×3（竖直边缘处响应强）：")
    print(out.squeeze().tolist())
    print(f"\n边缘列响应 out[0,0,:,1] = {[f'{v:.1f}' for v in out[0, 0, :, 1].tolist()]}（竖直边处应显著非零）")


if __name__ == "__main__":
    main()
