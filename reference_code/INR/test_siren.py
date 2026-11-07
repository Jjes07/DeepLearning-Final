import torch
import numpy as np
from siren import ImplicitPhaseNetwork, build_coordinate_grid

H, W = 64, 64
x = np.linspace(0, 2*np.pi, W)
y = np.linspace(0, 2*np.pi, H)
XX, YY = np.meshgrid(x, y)
true_phase = XX
wrapped = np.angle(np.exp(1j * true_phase))

device  = torch.device('cpu')
coords  = build_coordinate_grid(H, W, device=device)
wrapped_t = torch.tensor(wrapped, dtype=torch.float32)

# Calcular gradientes raw
gx_raw = torch.zeros(H, W)
gx_raw[:, :-1] = torch.tensor(
    np.angle(np.exp(1j*(true_phase[:, 1:] - true_phase[:, :-1])))
, dtype=torch.float32)

print(f"gx_raw rango: [{gx_raw.min():.4f}, {gx_raw.max():.4f}]")
print(f"gx_raw valor típico: {gx_raw[:, 30].mean():.4f}")
print(f"Gradiente verdadero por pixel: {2*np.pi/W:.4f}")

# La pregunta es: ¿los gradientes raw contienen info suficiente?
# Si gx_raw ≈ gradiente_verdadero, la red debería poder aprender
# Si gx_raw ≈ 0 en todas partes, no hay información

nonzero = (gx_raw.abs() > 0.01).float().mean()

network = ImplicitPhaseNetwork(hidden_dim=256, n_layers=3, omega_0=10)
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)


def charbonnier(x, eps=1e-6):
    return torch.sqrt(x ** 2 + eps ** 2)


print("\n¿Puede la red aprender con solo gradientes?")
for i in range(2000):
    optimizer.zero_grad()

    coords_g = coords.detach().requires_grad_(True)
    phase = network(coords_g)

    grad = torch.autograd.grad(phase.sum(), coords_g,
                               create_graph=True)[0]
    pred_gx = grad[:, 0]
    pred_gy = grad[:, 1]

    gx_ = gx_raw.reshape(-1)

    L_data = charbonnier(pred_gx - gx_).mean()
    loss = L_data
    loss.backward()
    optimizer.step()

    if i % 400 == 0:
        with torch.no_grad():
            ph = network(coords)
        print(f"iter {i} | loss: {loss.item():.6f} | "
              f"fase: [{ph.min():.3f}, {ph.max():.3f}]")

print(f"\nFase verdadera: [{true_phase.min():.3f}, {true_phase.max():.3f}]")

from scipy.integrate import cumulative_trapezoid

print("\n=== Solución: integrar gradientes ===")
network2  = ImplicitPhaseNetwork(hidden_dim=256, n_layers=3, omega_0=10)
optimizer3 = torch.optim.Adam(network2.parameters(), lr=1e-3)

# Fase obtenida integrando los gradientes raw
# Esta es la "fase verdadera aproximada" que podemos comparar
gx_np = gx_raw.numpy()
phase_integrated = np.cumsum(gx_np, axis=1)
phase_integrated_t = torch.tensor(
    phase_integrated, dtype=torch.float32
).flatten()

print(f"Fase integrada: [{phase_integrated.min():.3f}, "
      f"{phase_integrated.max():.3f}]")
print(f"Fase verdadera: [{true_phase.min():.3f}, "
      f"{true_phase.max():.3f}]")

for i in range(2000):
    optimizer3.zero_grad()
    phase_pred = network2(coords)
    loss = ((phase_pred - phase_integrated_t)**2).mean()
    loss.backward()
    optimizer3.step()
    if i % 400 == 0:
        with torch.no_grad():
            ph = network2(coords)
        print(f"iter {i} | loss: {loss.item():.6f} | "
              f"fase: [{ph.min():.3f}, {ph.max():.3f}]")