"""
src/model.py - ResNet with Self-Attention (1D DNA 序列分类)

结构 (输入 400bp):
  Input:  (B, 4, 400)
  Conv1:  (B, 8, 400)
  Layer1 (stride=1) + SelfAttention: (B, 8, 400)
  Layer2 (stride=2) + SelfAttention: (B, 16, 200)
  Layer3 (stride=2) + SelfAttention: (B, 32, 100)
  AvgPool1d(5):                      (B, 32, 20)
  Flatten:                           (B, 640)
  Heads: Linear(640,32) -> Linear(32,4) -> Linear(4,2)

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
        self.query_conv = nn.Conv1d(in_channels, in_channels // SCALING, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, in_channels // SCALING, kernel_size=1)
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
    def __init__(self, block=BasicBlock, num_blocks=(2, 2, 2), num_classes=NUM_CLASS):
        super(ResNetSelfAttention, self).__init__()
        self.in_planes = LAYER_SIZE

        self.shared_layers = nn.Sequential(
            nn.Conv1d(4, LAYER_SIZE, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(LAYER_SIZE),
            self._make_layer(block, LAYER_SIZE, num_blocks[0], stride=1),
            SelfAttention(LAYER_SIZE),
            self._make_layer(block, LAYER_SIZE * 2, num_blocks[1], stride=2),
            SelfAttention(LAYER_SIZE * 2),
            self._make_layer(block, LAYER_SIZE * 4, num_blocks[2], stride=2),
            SelfAttention(LAYER_SIZE * 4),
            nn.AvgPool1d(kernel_size=POOLING_KERNEL)
        )

        # 两层 stride=2 下采样 + AvgPool: 400 / 2 / 2 / 5 = 20
        pool_out_len = SEQUENCE_LENGTH // 4 // POOLING_KERNEL       # = 20
        linear_input_size = LAYER_SIZE * 4 * pool_out_len           # = 32*20 = 640
        hidden1 = linear_input_size // 20                           # = 32
        hidden2 = linear_input_size // 160                          # = 4

        if hidden1 <= 0 or hidden2 <= 0:
            raise ValueError(
                f"Degenerate head dims ({hidden1}, {hidden2}) — check SEQUENCE_LENGTH={SEQUENCE_LENGTH}")

        self.heads = nn.Sequential(
            nn.Linear(linear_input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes)
        )

        # 
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=LAYER_SIZE * 4, num_heads=8, batch_first=True
        )

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