import torch
import torch.nn as nn
import math

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                           num_layers=num_layers, batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self._init_weights()
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name: nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name: nn.init.orthogonal_(param)
            elif 'bias' in name: nn.init.zeros_(param)
    def forward(self, x, mask=None):
        lstm_out, _ = self.lstm(x)
        if mask is not None:
            timestep_mask = (mask.sum(dim=2) > 0).float()
            lengths = timestep_mask.sum(dim=1).long()
            lengths = torch.clamp(lengths, min=1, max=lstm_out.size(1))
            batch_size = lstm_out.size(0)
            last_outputs = lstm_out[torch.arange(batch_size), lengths - 1, :]
        else:
            last_outputs = lstm_out[:, -1, :]
        return self.fc(self.dropout(last_outputs)).squeeze(-1)

class TransformerClassifier(nn.Module):
    def __init__(self, input_size=40, d_model=64, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x, mask=None):
        x = self.input_proj(x) * math.sqrt(x.size(-1))
        x = self.pos_encoder(x)
        if mask is not None:
            src_key_padding_mask = (mask.sum(dim=2) == 0)
            x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        else:
            x = self.transformer_encoder(x)
        return self.fc(self.dropout(x[:, -1, :])).squeeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.3, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class RealMamba(nn.Module):
    def __init__(self, input_size=40, d_model=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.layers = nn.ModuleList([
            nn.LSTM(d_model, d_model, batch_first=True, dropout=dropout) 
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model, 1)
        self._init_weights()  # ← КРИТИЧНО!
        
    def _init_weights(self):
        """Правильная инициализация для стабильности"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
        
    def forward(self, x, mask=None):
        x = self.input_proj(x)
        for layer in self.layers:
            x, _ = layer(x)
        if mask is not None:
            timestep_mask = (mask.sum(dim=2) > 0).float()
            lengths = timestep_mask.sum(dim=1).long()
            lengths = torch.clamp(lengths, min=1, max=x.size(1))
            batch_size = x.size(0)
            x = x[torch.arange(batch_size), lengths - 1, :]
        else:
            x = x[:, -1, :]
        return self.output_proj(self.dropout(x)).squeeze(-1)

def create_model(model_name, input_size=40, **kwargs):
    if model_name == 'lstm': return LSTMClassifier(input_size=input_size, **kwargs)
    elif model_name == 'transformer': return TransformerClassifier(input_size=input_size, **kwargs)
    elif model_name == 'real_mamba': return RealMamba(input_size=input_size, **kwargs)
    else: raise ValueError(f"Unknown model: {model_name}")

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
