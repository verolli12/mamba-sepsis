#!/usr/bin/env python3
"""
Модели для классификации медицинских временных рядов
PhysioNet 2019 Sepsis Prediction Challenge
"""

import torch
import torch.nn as nn
import math
from mamba_ssm import Mamba


# ═══════════════════════════════════════════════════════════════
# 1. LSTM CLASSIFIER
# ═══════════════════════════════════════════════════════════════
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.fc(h_n[-1]).squeeze(-1)
        return out


# ═══════════════════════════════════════════════════════════════
# 2. TRANSFORMER CLASSIFIER
# ═══════════════════════════════════════════════════════════════
class TransformerClassifier(nn.Module):
    def __init__(self, input_size=40, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.3):
        super().__init__()
        self.d_model = d_model
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)


class PositionalEncoding(nn.Module):
    """Positional encoding для Transformer (batch_first=True)"""
    def __init__(self, d_model, dropout=0.3, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Создаём positional encoding правильной формы
        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # [d_model/2]
        
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Добавляем batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        # self.pe: [1, max_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ═══════════════════════════════════════════════════════════════
# 3. REAL MAMBA CLASSIFIER
# ═══════════════════════════════════════════════════════════════
class RealMambaClassifier(nn.Module):
    def __init__(self, input_size=40, d_model=128, d_state=16, n_layer=4, 
                 d_conv=4, expand=2, dropout=0.3):  # Убрали bidirectional
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        
        self.input_projection = nn.Linear(input_size, d_model)
        
        self.mamba_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(n_layer):
            self.mamba_layers.append(
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv,
                      expand=expand)  # Убрали bidirectional
            )
            self.norms.append(nn.LayerNorm(d_model))
        
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x, mask=None):
        x = self.input_projection(x)
        
        for i in range(self.n_layer):
            residual = x
            x = self.norms[i](x)
            x = self.mamba_layers[i](x)
            x = x + residual
        
        if mask is not None:
            mask_seq = mask.mean(dim=2).unsqueeze(-1)
            x = (x * mask_seq).sum(dim=1) / (mask_seq.sum(dim=1) + 1e-8)
        else:
            x = x.mean(dim=1)
        
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x.squeeze(-1)

# ═══════════════════════════════════════════════════════════════
# 4. GRU-D CLASSIFIER
# ═══════════════════════════════════════════════════════════════
class GRUD(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.x_mean = nn.Parameter(torch.zeros(input_size))
        self.gamma_x = nn.Parameter(torch.ones(input_size))
        self.gamma_h = nn.Parameter(torch.ones(hidden_size))
        
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape
        x_imputed = x.clone()
        h_decay = torch.ones(batch, self.hidden_size, device=x.device)
        
        if mask is None:
            mask = torch.ones_like(x)
        
        for t in range(seq_len):
            if t > 0:
                h_decay = h_decay * torch.exp(-self.gamma_h)
            
            m_t = mask[:, t, :]
            x_imputed[:, t, :] = m_t * x[:, t, :] + (1 - m_t) * self.x_mean
            x_imputed[:, t, :] = x_imputed[:, t, :] * m_t + \
                                  (1 - m_t) * self.x_mean * torch.exp(-self.gamma_x)
        
        gru_out, h_n = self.gru(x_imputed)
        h_final = h_n[-1] * h_decay
        
        return self.fc(h_final).squeeze(-1)


# ═══════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════
def create_model(model_name, input_size=40, **kwargs):
    if model_name == 'lstm':
        return LSTMClassifier(input_size=input_size, **kwargs)
    elif model_name == 'transformer':
        return TransformerClassifier(input_size=input_size, **kwargs)
    elif model_name == 'real_mamba':
        return RealMambaClassifier(input_size=input_size, **kwargs)
    elif model_name == 'grud':
        return GRUD(input_size=input_size, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("ТЕСТ МОДЕЛЕЙ")
    print("=" * 60)
    
    # Проверяем наличие GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    models = ['lstm', 'transformer', 'real_mamba', 'grud']
    batch_size = 4
    seq_len = 24
    input_size = 40
    
    for model_name in models:
        print(f"\n{model_name.upper()}:")
        model = create_model(model_name, input_size=input_size)
        model = model.to(device)  # Перемещаем модель на GPU
        x = torch.randn(batch_size, seq_len, input_size).to(device)  # Данные на GPU
        mask = torch.ones(batch_size, seq_len, input_size).to(device)  # Маска на GPU
        
        with torch.no_grad():
            out = model(x, mask)
        
        params = count_parameters(model)
        print(f"  Output: {out.shape}, Parameters: {params:,}, ✅ OK")