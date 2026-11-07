import torch
import torch.nn as nn
import numpy as np


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features,
                 omega_0=30, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.in_features,
                     1 / self.in_features
                )
            else:
                n = self.linear.weight.shape[1]
                self.linear.weight.uniform_(
                    -np.sqrt(6 / n) / self.omega_0,
                     np.sqrt(6 / n) / self.omega_0
                )

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class ImplicitPhaseNetwork(nn.Module):
    """
    Red SIREN para representación implícita de fase.
    Entrada:  coordenadas (x, y) normalizadas en [-1, 1]
    Salida:   valor de fase continua φ(x,y)
    """
    def __init__(self, hidden_dim=256, n_layers=3, omega_0=10):
        super().__init__()

        # Capa de entrada
        layers = [SineLayer(2, hidden_dim,
                            omega_0=omega_0, is_first=True)]

        # Capas ocultas
        for _ in range(n_layers):
            layers.append(SineLayer(hidden_dim, hidden_dim,
                                    omega_0=omega_0))

        self.net = nn.Sequential(*layers)

        # Capa de salida lineal (sin activación)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, coords):
        """
        coords: tensor (N, 2)
        return: tensor (N,)
        """
        features = self.net(coords)
        phase = self.output(features)
        return phase.squeeze(-1)


class LowFreqBackgroundNetwork(nn.Module):
    """
    Red de baja frecuencia para estimar el background.
    omega_0 pequeño = solo captura variaciones lentas.
    """
    def __init__(self, hidden_dim=128, n_layers=2, omega_0=1.0):
        super().__init__()

        layers = [SineLayer(2, hidden_dim,
                            omega_0=omega_0, is_first=True)]
        for _ in range(n_layers):
            layers.append(SineLayer(hidden_dim, hidden_dim,
                                    omega_0=omega_0))

        self.net = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, coords):
        return self.output(self.net(coords)).squeeze(-1)


def build_coordinate_grid(H, W, device='cpu'):
    """
    Crea grid normalizado de coordenadas en [-1, 1].
    Retorna tensor (H*W, 2)
    """
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    coords = torch.stack([grid_x.flatten(),
                          grid_y.flatten()], dim=1)
    return coords