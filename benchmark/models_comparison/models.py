#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark/models_comparison/models.py

包含论文 3.1 节用于对比测试的模型架构:
1. ResNet50: 标准的 ResNet50 一维版本
2. ResNetNoAttention: 剔除了 Conv-SA 的基础模型 (Ablation)
3. TransformerClassifier: 纯 Transformer 模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import BasicBlock

# --- 1. ResNet50 ---
class Bottleneck(nn.Module):
    def __init__(self, In_channel, Med_channel, Out_channel, downsample=False):
        super(Bottleneck, self).__init__()
        self.stride = 2 if downsample else 1
        self.relu = nn.ReLU(inplace=True)

        self.layer = nn.Sequential(
            nn.Conv1d(In_channel, Med_channel, 1, self.stride, bias=False),
            nn.BatchNorm1d(Med_channel),
            nn.ReLU(inplace=True),
            nn.Conv1d(Med_channel, Med_channel, 3, padding=1, bias=False),
            nn.BatchNorm1d(Med_channel),
            nn.ReLU(inplace=True),
            nn.Conv1d(Med_channel, Out_channel, 1, bias=False),
            nn.BatchNorm1d(Out_channel),
        )

        if In_channel != Out_channel:
            self.res_layer = nn.Sequential(
                nn.Conv1d(In_channel, Out_channel, 1, self.stride, bias=False),
                nn.BatchNorm1d(Out_channel)
            )
        else:
            self.res_layer = None

    def forward(self, x):
        residual = self.res_layer(x) if self.res_layer is not None else x
        out = self.layer(x) + residual
        out = self.relu(out)
        return out

class ResNet50(nn.Module):
    def __init__(self, in_channels=4, classes=2):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 512, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, 2, 1),
            
            Bottleneck(512, 256, 1024, True),
            Bottleneck(1024, 256, 1024, False),
            Bottleneck(1024, 256, 1024, False),

            Bottleneck(1024, 512, 2048, True),
            Bottleneck(2048, 512, 2048, False),
            Bottleneck(2048, 512, 2048, False),
            Bottleneck(2048, 512, 2048, False),

            Bottleneck(2048, 1024, 4096, True),
            Bottleneck(4096, 1024, 4096, False),
            Bottleneck(4096, 1024, 4096, False),
            Bottleneck(4096, 1024, 4096, False),
            Bottleneck(4096, 1024, 4096, False),
            Bottleneck(4096, 1024, 4096, False),
                        
            Bottleneck(4096, 2048, 8192, True),
            Bottleneck(8192, 2048, 8192, False),
            Bottleneck(8192, 2048, 8192, False),

            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(8192, classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# --- 2. ResNet without Self-Attention (消融实验组) ---
class ResNetNoAttention(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=(2, 2, 2), num_classes=2, layer_size=8):
        super(ResNetNoAttention, self).__init__()
        self.in_planes = layer_size

        self.shared_layers = nn.Sequential(
            nn.Conv1d(4, layer_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(layer_size),
            self._make_layer(block, layer_size, num_blocks[0], stride=1),
            # 移除了 SelfAttention
            self._make_layer(block, layer_size * 2, num_blocks[1], stride=2),
            # 移除了 SelfAttention
            self._make_layer(block, layer_size * 4, num_blocks[2], stride=2),
            # 移除了 SelfAttention
            nn.AvgPool1d(kernel_size=5)
        )

        pool_out_len = 400 // 4 // 5
        linear_input_size = layer_size * 4 * pool_out_len
        hidden1 = max(1, linear_input_size // 20)
        hidden2 = max(1, linear_input_size // 160)

        self.heads = nn.Sequential(
            nn.Linear(linear_input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes)
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


# --- 3. Transformer ---
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, dropout):
        super(TransformerBlock, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(embed_dim, heads)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, src):
        out = self.multi_head_attention(src, src, src)[0]
        out = self.dropout1(out)
        out = self.layer_norm1(out + src)

        src2 = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.layer_norm2(out + src2)
        return out

class TransformerClassifier(nn.Module):
    def __init__(self, num_tokens=4, num_classes=2, embedding_dim=96, 
                 transformer_depth=24, heads=24, dropout=0.5):
        super(TransformerClassifier, self).__init__()
        self.token_embedding = nn.Embedding(num_tokens, embedding_dim)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embedding_dim, heads, dropout) for _ in range(transformer_depth)]
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # 兼容最新 DataLoader 的 One-hot float 张量 (B, 4, L)
        # 将其无损转换为 Token 索引序列 (B, L)
        if x.dim() == 3 and x.size(1) == 4:
            x = torch.argmax(x, dim=1)
            
        x = self.token_embedding(x)
        x = x.transpose(0, 1)  # PyTorch MHA 期望形状: (Seq_len, Batch, Embed_dim)
        x = self.transformer_blocks(x)
        x = x.transpose(0, 1)  # 还原回 (Batch, Seq_len, Embed_dim)
        x = self.pooling(x.transpose(1, 2))
        x = x.squeeze(-1)
        x = self.fc(x)
        return x