#!/usr/bin/env python3
"""
Time-aware версии моделей
"""

import torch
import torch.nn as nn


class TimeAwareLSTM(nn.Module):
    def __init__(self, input_size=42, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # Последний временной шаг
        out = self.dropout(out)
        return self.fc(out)


class TimeAwareMamba(nn.Module):
    def __init__(self, input_size=42, d_model=128, n_layers=4, dropout=0.3):
        super().__init__()
        try:
            from mamba_ssm import Mamba
            self.mamba_blocks = nn.ModuleList([
                Mamba(d_model=d_model) for _ in range(n_layers)
            ])
        except:
            # Fallback если mamba_ssm не установлен
            self.mamba_blocks = nn.ModuleList([
                nn.LSTM(d_model, d_model, batch_first=True) for _ in range(n_layers)
            ])
        
        self.input_proj = nn.Linear(input_size, d_model)
        self.output_proj = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        x = self.input_proj(x)
        for block in self.mamba_blocks:
            if isinstance(block, nn.LSTM):
                x, _ = block(x)
            else:
                x = block(x)
        out = x[:, -1, :]
        out = self.dropout(out)
        return self.output_proj(out)


def create_timeaware_model(model_name, input_size=42, d_model=128, hidden_size=128):
    if model_name == 'lstm_time':
        return TimeAwareLSTM(input_size=input_size, hidden_size=hidden_size)
    elif model_name == 'mamba_time':
        return TimeAwareMamba(input_size=input_size, d_model=d_model)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
