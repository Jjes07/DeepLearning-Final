"""Modelos de fondo: implícito por imagen (A) y amortizado (C)."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_coordinate_grid(H, W, device="cpu"):
    """Grid de coordenadas normalizadas en [-1, 1]², forma (H*W, 2)."""
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)


class BaseSuave(nn.Module):
    """Base Φ(x,y): polinomios de bajo grado + Fourier con frecuencias enteras hasta f_max."""

    def __init__(self, H, W, f_max=4, grado_poli=2):
        super().__init__()
        self.H, self.W = H, W
        self.f_max = f_max
        self.grado_poli = grado_poli
        coords = build_coordinate_grid(H, W, device="cpu")
        self.register_buffer("coords", coords)
        self.register_buffer("base", self._construir_base(coords))
        self.n_coef = self.base.shape[1]

    def _construir_base(self, coords):
        x = coords[:, 0]
        y = coords[:, 1]
        cols = []
        # Polinomios
        for g in range(self.grado_poli + 1):
            for j in range(g + 1):
                cols.append((x ** (g - j)) * (y ** j))
        # Fourier
        for k in range(1, self.f_max + 1):
            cols.extend([
                torch.cos(2 * np.pi * k * x),
                torch.sin(2 * np.pi * k * x),
                torch.cos(2 * np.pi * k * y),
                torch.sin(2 * np.pi * k * y),
            ])
        return torch.stack(cols, dim=1)

    def forward(self, coef):
        """coef: (B, n_coef) o (n_coef,). Devuelve fondo (B, H, W) o (H, W)."""
        if coef.dim() == 1:
            return (self.base @ coef).view(self.H, self.W)
        return (self.base @ coef.T).view(coef.shape[0], self.H, self.W)


class ImplicitFourierMLP(nn.Module):
    """Red implícita por imagen: MLP sobre coordenadas + Fourier features."""

    def __init__(self, H, W, hidden=(128, 128, 128), fourier_dim=64):
        super().__init__()
        self.H, self.W = H, W
        self.fourier_dim = fourier_dim
        self.B = nn.Parameter(torch.randn(2, fourier_dim) * 1.0, requires_grad=False)
        in_dim = 2 * fourier_dim
        capas = []
        for i, h in enumerate(hidden):
            capas.append(nn.Linear(in_dim if i == 0 else hidden[i - 1], h))
            capas.append(nn.ReLU(inplace=True))
        capas.append(nn.Linear(hidden[-1], 1))
        self.net = nn.Sequential(*capas)

    def _features(self, coords):
        proj = 2 * np.pi * coords @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, coords):
        return self.net(self._features(coords)).squeeze(-1)


class SirenBackgroundNetwork(nn.Module):
    """SIREN de baja frecuencia (similar al código guía) para fondo."""

    class SineLayer(nn.Module):
        def __init__(self, in_features, out_features, omega_0=1.0, is_first=False):
            super().__init__()
            self.omega_0 = omega_0
            self.is_first = is_first
            self.linear = nn.Linear(in_features, out_features)
            self._init_weights()

        def _init_weights(self):
            with torch.no_grad():
                if self.is_first:
                    self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
                else:
                    n = self.linear.in_features
                    self.linear.weight.uniform_(-np.sqrt(6 / n) / self.omega_0, np.sqrt(6 / n) / self.omega_0)

        def forward(self, x):
            return torch.sin(self.omega_0 * self.linear(x))

    def __init__(self, hidden_dim=128, n_layers=2, omega_0=1.0):
        super().__init__()
        layers = [self.SineLayer(2, hidden_dim, omega_0=omega_0, is_first=True)]
        for _ in range(n_layers):
            layers.append(self.SineLayer(hidden_dim, hidden_dim, omega_0=omega_0))
        self.net = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, coords):
        return self.output(self.net(coords)).squeeze(-1)


class EncoderAmortizado(nn.Module):
    """Codificador convolucional pequeño: u -> coeficientes de la base suave."""

    def __init__(self, n_coef, base_channels=32):
        super().__init__()
        self.n_coef = n_coef
        c = base_channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c, c * 2, 3, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c * 2, c * 4, 3, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(c * 4, n_coef)

    def forward(self, u):
        """u: (B, 1, H, W). Devuelve (B, n_coef)."""
        x = self.conv(u)
        return self.fc(x.view(x.shape[0], -1))
