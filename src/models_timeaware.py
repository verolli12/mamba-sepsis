import torch
import torch.nn as nn


class BaselineLSTM(nn.Module):
    def __init__(self, input_size: int = 40, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        
        # 🛡️ Берём последний ВРЕМЕННОЙ ШАГ С УЧЁТОМ МАСКИ
        if mask is not None:
            seq_lengths = mask.sum(dim=1).long()  # [B]
            last_indices = (seq_lengths - 1).clamp(min=0)
            out = lstm_out[torch.arange(x.size(0)), last_indices]
        else:
            out = lstm_out[:, -1, :]
            
        out = self.dropout(out)
        return self.fc(out)


class BaselineMamba(nn.Module):
    def __init__(self, input_size: int = 40, d_model: int = 128, 
                 n_layers: int = 4, dropout: float = 0.3):
        super().__init__()
        try:
            from mamba_ssm import Mamba
            self.mamba_blocks = nn.ModuleList([
                Mamba(d_model=d_model) for _ in range(n_layers)
            ])
            self.uses_mamba = True
        except ImportError:
            raise ImportError(
                "mamba-ssm не установлен. Установи: pip install mamba-ssm\n"
                "Или используй BaselineLSTM вместо BaselineMamba."
            )
        
        self.input_proj = nn.Linear(input_size, d_model)
        self.output_proj = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.mamba_blocks:
            x = block(x)  # Mamba возвращает только hidden state
        
        # 🛡️ Аналогично LSTM: берём последний валидный шаг
        if mask is not None:
            seq_lengths = mask.sum(dim=1).long()
            last_indices = (seq_lengths - 1).clamp(min=0)
            out = x[torch.arange(x.size(0)), last_indices]
        else:
            out = x[:, -1, :]
            
        out = self.dropout(out)
        return self.output_proj(out)


def create_baseline_model(model_name: str, input_size: int = 40, **kwargs):
    models = {
        'lstm_time': lambda: BaselineLSTM(input_size=input_size, **kwargs),
        'mamba_time': lambda: BaselineMamba(input_size=input_size, **kwargs),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Доступно: {list(models.keys())}")
    return models[model_name]()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)