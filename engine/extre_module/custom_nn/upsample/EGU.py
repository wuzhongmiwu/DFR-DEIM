import torch
import torch.nn as nn
import torch.nn.functional as F
from engine.extre_module.ultralytics_nn.conv import Conv


class EGU(nn.Module):
    """
    Edge-Guided Upsampling (EGU) - 终极参数化版
    针对微小目标的边缘感知上采样模块，结合双线性插值与 Sobel 物理先验锐化。
    引入 refine_weight 旋钮，可灵活控制边缘激活强度。
    """

    def __init__(self, in_ch, out_ch, scale_factor=2, refine_weight=1.0):
        super().__init__()
        self.scale_factor = scale_factor

        # 🎯 核心旋钮：控制边缘锐化的强度 (默认 1.0 为激进版，适合极端微小目标)
        self.refine_weight = refine_weight

        # 1. 如果通道数需要改变，用一个 1x1 卷积过渡；如果不变，就是 Identity（零参数）
        self.proj = Conv(in_ch, out_ch, k=1) if in_ch != out_ch else nn.Identity()

        # 2. 轻量级边缘锐化卷积 (3x3 深度可分离卷积，极低参数量)
        self.refine = Conv(out_ch, out_ch, k=3, g=out_ch)

        # 3. 注册固化的 Sobel 算子 (零参数注入，呼应 EGR 的物理降噪)
        sobel_x = torch.tensor([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ])
        sobel_y = torch.tensor([
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ])
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3), persistent=False)

        # 4. 轻量级边缘注意力生成器
        self.edge_att_gen = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Step 1: 平滑的双线性上采样 (不改变结构，不产生形变)
        x_up = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x_up = self.proj(x_up)

        # Step 2: 提取放大后的边缘幅值 (Sobel Edge Magnitude)
        gray = x_up.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)

        # Step 3: 生成边缘注意力掩码 (0~1之间)
        edge_att = self.edge_att_gen(mag)

        # Step 4: 边缘特征锐化与残差融合
        # 仅在目标边缘处（edge_att 较高的地方）应用 refine 卷积
        x_refined = self.refine(x_up)

        # 🎯 将旋钮 refine_weight 乘入最终的残差流中
        out = x_up + self.refine_weight * x_refined * edge_att

        return out


if __name__ == '__main__':
    # 本地跑通测试
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 256, 256, 16, 16
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # 实例化 EGU (默认激进版 refine_weight=1.0)
    module = EGU(in_channel, out_channel, scale_factor=2).to(device)
    outputs = module(inputs)

    print(f'输入尺寸: {inputs.size()} -> 输出尺寸: {outputs.size()}')
    print(f'当前 EGU 锐化权重 (refine_weight): {module.refine_weight}')