import torch
import torch.nn as nn


class ConvStem1D(nn.Module):
    """Temporal branch used by EdgeNetTF."""

    def __init__(self, num_channels=12, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, d_model, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FreqStem1D(nn.Module):
    """FFT-magnitude branch used by EdgeNetTF."""

    def __init__(self, num_channels=12, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class EdgeNetTF(nn.Module):
    """Final concatenation-based EdgeNetTF model reported in the manuscript."""

    def __init__(self, num_classes=49, num_channels=12, d_model=128, dropout=0.3):
        super().__init__()
        self.time_stem = ConvStem1D(num_channels=num_channels, d_model=d_model)
        self.freq_stem = FreqStem1D(num_channels=num_channels, d_model=d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_time, x_freq):
        t_feat = self.time_stem(x_time)
        f_feat = self.freq_stem(x_freq)
        return self.classifier(torch.cat([t_feat, f_feat], dim=1))


class TemporalOnlyNet(nn.Module):
    def __init__(self, num_classes=49, num_channels=12, d_model=128, dropout=0.3):
        super().__init__()
        self.time_stem = ConvStem1D(num_channels=num_channels, d_model=d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_time, x_freq=None):
        return self.classifier(self.time_stem(x_time))


class FFTOnlyNet(nn.Module):
    def __init__(self, num_classes=49, num_channels=12, d_model=128, dropout=0.3):
        super().__init__()
        self.freq_stem = FreqStem1D(num_channels=num_channels, d_model=d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_time, x_freq):
        return self.classifier(self.freq_stem(x_freq))


class CrossModalFusion(nn.Module):
    """Adaptive gated fusion used only in the EdgeNetTF-Gated ablation."""

    def __init__(self, channels=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(channels * 2, channels // 2),
            nn.BatchNorm1d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, channels * 2),
            nn.Sigmoid(),
        )

    def forward(self, x_time, x_freq):
        weights = self.attention(torch.cat([x_time, x_freq], dim=1))
        w_time, w_freq = weights.chunk(2, dim=1)
        return x_time * w_time + x_freq * w_freq


class EdgeNetTFGated(nn.Module):
    """Gated-fusion ablation. This is not the final manuscript model."""

    def __init__(self, num_classes=49, num_channels=12, d_model=128, dropout=0.3):
        super().__init__()
        self.time_stem = ConvStem1D(num_channels=num_channels, d_model=d_model)
        self.freq_stem = FreqStem1D(num_channels=num_channels, d_model=d_model)
        self.fusion = CrossModalFusion(channels=d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_time, x_freq):
        t_feat = self.time_stem(x_time)
        f_feat = self.freq_stem(x_freq)
        return self.classifier(self.fusion(t_feat, f_feat))
