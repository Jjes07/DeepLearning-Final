import torch
import torch.nn.functional as F


def compute_wrapped_gradients(wrapped_phase):
    def wrap(x):
        return torch.atan2(torch.sin(x), torch.cos(x))

    diff_x = wrapped_phase[:, 1:] - wrapped_phase[:, :-1]
    diff_y = wrapped_phase[1:, :] - wrapped_phase[:-1, :]

    gx = F.pad(wrap(diff_x), (0, 1))
    gy = F.pad(wrap(diff_y), (0, 0, 0, 1))

    return gx, gy


def compute_reliability_map(wrapped_phase,
                             window_size=5, alpha=2.0):
    H, W = wrapped_phase.shape
    pad  = window_size // 2

    gx, gy = compute_wrapped_gradients(wrapped_phase)

    # Kernel gaussiano en vez de uniforme
    def gaussian_kernel(size, sigma=1.0):
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        k = g.outer(g)
        return k.unsqueeze(0).unsqueeze(0).to(wrapped_phase.device)

    kernel = gaussian_kernel(window_size, sigma=1.5)

    def local_variance(g):
        g4d    = g.unsqueeze(0).unsqueeze(0)
        mean   = F.conv2d(g4d, kernel, padding=pad).squeeze()
        mean_sq = F.conv2d(g4d**2, kernel, padding=pad).squeeze()
        return (mean_sq - mean**2).clamp(min=0)

    var_x = local_variance(gx)
    var_y = local_variance(gy)

    def wrap(x):
        return torch.atan2(torch.sin(x), torch.cos(x))

    # Residuos de segundo orden
    gxx = F.pad(wrap(gx[:, 1:] - gx[:, :-1]), (0, 1))
    gyy = F.pad(wrap(gy[1:, :] - gy[:-1, :]), (0, 0, 0, 1))

    S = var_x + var_y + gxx.abs() + gyy.abs()

    # Suavizar S antes del max pooling
    S_smooth = F.conv2d(
        S.unsqueeze(0).unsqueeze(0),
        gaussian_kernel(window_size),
        padding=pad
    ).squeeze()

    # Normalizar S a [0,1] antes de exp
    S_norm = (S_smooth - S_smooth.min()) / (
        S_smooth.max() - S_smooth.min() + 1e-8
    )

    reliability = torch.exp(-alpha * S_norm)

    return reliability