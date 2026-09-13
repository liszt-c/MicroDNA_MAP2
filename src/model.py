"""
src/model.py - ResNet with Self-Attention (1D DNA 序列分类)

结构 (输入 400bp):
  Input:  (B, 4, 400)
  Conv1:  (B, LAYER_SIZE, 400)
  Layer1 (stride=1) + SelfAttention: (B, LAYER_SIZE, 400)
  Layer2 (stride=2) + SelfAttention: (B, LAYER_SIZE*2, 200)
  Layer3 (stride=2) + SelfAttention: (B, LAYER_SIZE*4, 100)
  AvgPool1d(5):                      (B, LAYER_SIZE*4, 20)
  Flatten:                           (B, LAYER_SIZE*4*20)
  Heads: Linear -> Linear -> Linear(num_classes)

消融实验安全改动:
  - SelfAttention 内部通道数 max(1, in_channels // SCALING), 防止 LAYER_SIZE 过小时除零
  - Head 隐藏层维度 max(1, ...), 防止退化
  - MultiheadAttention 仅在 embed_dim >= num_heads 时创建, 否则置 None
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SEQUENCE_LENGTH

LAYER_SIZE = 8
NUM_CLASS = 2
POOLING_KERNEL = 5
SCALING = 4


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        # 保护: LAYER_SIZE 过小时 in_channels // SCALING 可能为 0
        inner_channels = max(1, in_channels // SCALING)
        self.query_conv = nn.Conv1d(in_channels, inner_channels, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, inner_channels, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, length = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, length).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, length)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, length)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, length)
        out = self.gamma * out + x
        return out


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetSelfAttention(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=(2, 2, 2),
                 num_classes=NUM_CLASS, layer_size=LAYER_SIZE):
        """
        :param layer_size: 基础通道数, 支持消融实验动态调整
        """
        super(ResNetSelfAttention, self).__init__()
        self.layer_size = layer_size
        self.in_planes = layer_size

        self.shared_layers = nn.Sequential(
            nn.Conv1d(4, layer_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(layer_size),
            self._make_layer(block, layer_size, num_blocks[0], stride=1),
            SelfAttention(layer_size),
            self._make_layer(block, layer_size * 2, num_blocks[1], stride=2),
            SelfAttention(layer_size * 2),
            self._make_layer(block, layer_size * 4, num_blocks[2], stride=2),
            SelfAttention(layer_size * 4),
            nn.AvgPool1d(kernel_size=POOLING_KERNEL)
        )

        # 两层 stride=2 下采样 + AvgPool: 400 / 2 / 2 / 5 = 20
        pool_out_len = SEQUENCE_LENGTH // 4 // POOLING_KERNEL       # = 20
        linear_input_size = layer_size * 4 * pool_out_len

        # 保护: 防止 LAYER_SIZE 过小时隐藏层退化为 0
        hidden1 = max(1, linear_input_size // 20)
        hidden2 = max(1, linear_input_size // 160)

        self.heads = nn.Sequential(
            nn.Linear(linear_input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes)
        )

        # MultiheadAttention: 仅在维度合法时创建 
        # 早期项目残留代码, 当前 forward 未使用
        # 若未来启用, 需同步修改 forward() 并重新验证
        # 注释以防增加显存
        '''
        mha_embed_dim = layer_size * 4
        mha_num_heads = 8
        if mha_embed_dim >= mha_num_heads and mha_embed_dim % mha_num_heads == 0:
            self.multihead_attn = nn.MultiheadAttention(
                embed_dim=mha_embed_dim, num_heads=mha_num_heads, batch_first=True
            )
        else:
            self.multihead_attn = None
        '''
        self.multihead_attn = None

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.shared_layers(x)
        out = out.view(out.size(0), -1)
        out = self.heads(out)
        return out