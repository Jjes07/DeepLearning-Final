import numpy as np
import matplotlib.pyplot as plt
import torch
from unwrapper import spinr_unwrap, compensate_background
from reliability import compute_reliability_map
from unwrapper import (spinr_unwrap,
                        compensate_background_zernike,
                        extract_sample_phase)


def create_test_phase(H=128, W=128):
    x = np.linspace(-np.pi, np.pi, W)
    y = np.linspace(-np.pi, np.pi, H)
    XX, YY = np.meshgrid(x, y)

    true_phase = (3 * np.sin(XX) +
                  2 * np.cos(YY) +
                  0.5 * XX * YY)

    noise = np.random.normal(0, 0.1, (H, W))
    true_phase += noise

    wrapped = np.angle(np.exp(1j * true_phase))
    return true_phase, wrapped
'''
def create_test_phase(H=64, W=64):
    x = np.linspace(0, 2 * np.pi, W)
    y = np.linspace(0, 2 * np.pi, H)
    XX, YY = np.meshgrid(x, y)
    true_phase = XX  # rampa simple
    wrapped = np.angle(np.exp(1j * true_phase))
    return true_phase, wrapped
'''
if __name__ == "__main__":
    np.random.seed(42)

    true_phase, wrapped = create_test_phase(H=128, W=128)

    # Parámetros actualizados
    unwrapped = spinr_unwrap(
        wrapped,
        n_iter_pretrain=300,  # ← antes n_iter_init
        n_iter_refine=500,  # ← antes n_iter_refine
        lr_pretrain=1e-3,  # ← antes lr_init
        lr_refine=1e-4,
        lambda_tv=0.001,
        lambda_cgrad=0.005,
        lambda_cmag=0.005
    )

    reliability = compute_reliability_map(
        torch.tensor(wrapped, dtype=torch.float32)
    ).numpy()

    compensated, background = compensate_background_zernike(
        unwrapped, reliability,
        n_terms=15,
        n_iter_cauchy=20,
        n_bandwidth=5,
        n_iter_corr=10
    )

    offset = np.mean(true_phase) - np.mean(compensated)
    compensated_aligned = compensated + offset

    rms_error = np.sqrt(
        np.mean((compensated_aligned - true_phase) ** 2)
    )
    print(f"\nError RMS: {rms_error:.4f} rad")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    im0 = axes[0,0].imshow(true_phase, cmap='jet')
    axes[0,0].set_title('Fase verdadera')
    plt.colorbar(im0, ax=axes[0,0])

    im1 = axes[0,1].imshow(wrapped, cmap='jet')
    axes[0,1].set_title('Wrapped phase (entrada)')
    plt.colorbar(im1, ax=axes[0,1])

    im2 = axes[0,2].imshow(reliability, cmap='gray')
    axes[0,2].set_title('Mapa de confiabilidad')
    plt.colorbar(im2, ax=axes[0,2])

    im3 = axes[1,0].imshow(unwrapped, cmap='jet')
    axes[1,0].set_title('Fase desenvuelta (SPINR)')
    plt.colorbar(im3, ax=axes[1,0])

    im4 = axes[1,1].imshow(compensated_aligned, cmap='jet')
    axes[1,1].set_title('Fase compensada (resultado final)')
    plt.colorbar(im4, ax=axes[1,1])

    error_map = np.abs(compensated_aligned - true_phase)
    im5 = axes[1,2].imshow(error_map, cmap='hot')
    axes[1,2].set_title(f'Mapa de error (RMS={rms_error:.3f})')
    plt.colorbar(im5, ax=axes[1,2])

    plt.tight_layout()
    plt.savefig('spinr_resultado.png', dpi=150)
    plt.show()
    print("Imagen guardada como spinr_resultado.png")