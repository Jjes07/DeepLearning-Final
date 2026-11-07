import torch
import torch.nn.functional as F


def charbonnier(x, eps=1e-6):
    return torch.sqrt(x ** 2 + eps ** 2)


def get_predicted_gradients(network, coords, image_shape):
    coords_grad = coords.detach().requires_grad_(True)
    phase = network(coords_grad)

    grad = torch.autograd.grad(
        phase.sum(), coords_grad, create_graph=True
    )[0]

    H, W = image_shape
    grad_x = grad[:, 0] * W
    grad_y = grad[:, 1] * H

    return phase, grad_x, grad_y


def loss_initial(network, coords, reliability,
                 gx_raw, gy_raw, image_shape,
                 lambda_tv=0.001):
    H, W = image_shape

    phase, pred_gx, pred_gy = get_predicted_gradients(
        network, coords, image_shape
    )

    r  = reliability.reshape(-1)
    gx = gx_raw.reshape(-1)
    gy = gy_raw.reshape(-1)

    # L_data: consistencia de gradientes
    err_x  = charbonnier(pred_gx - gx)
    err_y  = charbonnier(pred_gy - gy)
    L_data = (r * (err_x + err_y)).mean()

    # L_tv: MUY pequeño para no aplastar la fase
    eps_tv = 1e-6
    tv_w   = (1.0 - r).clamp(0, 0.3)  # limitar peso TV
    L_tv   = (tv_w * torch.sqrt(
        pred_gx ** 2 + pred_gy ** 2 + eps_tv
    )).mean()

    # L_wrap: ancla la fase a los valores de wrapped phase
    # esto evita el colapso a constante
    phi_w = wrapped_phase_from_gradients(gx, gy, phase, H, W)

    return L_data + lambda_tv * L_tv


def wrapped_phase_from_gradients(gx, gy, phase, H, W):
    return None  # placeholder


def loss_initial_v2(network, coords, reliability,
                    gx_raw, gy_raw, wrapped_phase,
                    image_shape, lambda_tv=0.001,
                    lambda_wrap=0.5):
    """
    Versión mejorada: agrega constraint de wrapped phase
    directamente para evitar colapso a constante.
    """
    H, W = image_shape

    phase, pred_gx, pred_gy = get_predicted_gradients(
        network, coords, image_shape
    )

    r  = reliability.reshape(-1)
    gx = gx_raw.reshape(-1)
    gy = gy_raw.reshape(-1)
    pw = wrapped_phase.reshape(-1)

    # L_data: consistencia de gradientes ponderada
    err_x  = charbonnier(pred_gx - gx)
    err_y  = charbonnier(pred_gy - gy)
    L_data = (r * (err_x + err_y)).mean()

    # L_tv: muy reducido
    eps_tv = 1e-6
    tv_w   = (1.0 - r).clamp(0, 0.2)
    L_tv   = (tv_w * torch.sqrt(
        pred_gx ** 2 + pred_gy ** 2 + eps_tv
    )).mean()

    # L_wrap: la fase predicha debe ser consistente
    # con la wrapped phase (diferencia debe ser múltiplo de 2π)
    residual = torch.atan2(
        torch.sin(pw - phase),
        torch.cos(pw - phase)
    )
    L_wrap = charbonnier(residual).mean()

    return L_data + lambda_tv * L_tv + lambda_wrap * L_wrap


def loss_refined(network, coords, wrapped_phase,
                 phase_init, image_shape,
                 lambda_cgrad=0.05, lambda_cmag=0.01):

    phase_hat, grad_x, grad_y = get_predicted_gradients(
        network, coords, image_shape
    )

    phi_w    = wrapped_phase.reshape(-1)
    phi_init = phase_init.reshape(-1)

    delta = phase_hat - phi_init

    residual = torch.atan2(
        torch.sin(phi_w - phase_hat),
        torch.cos(phi_w - phase_hat)
    )
    L_wrap = charbonnier(residual).mean()

    H, W    = image_shape
    phi2d   = phi_init.reshape(H, W)
    dx_init = phi2d[:, 1:] - phi2d[:, :-1]
    dy_init = phi2d[1:, :] - phi2d[:-1, :]
    G_init  = torch.sqrt(
        (dx_init ** 2).mean() + (dy_init ** 2).mean()
    ) + 1e-6

    delta2d = delta.reshape(H, W)
    d_dx    = delta2d[:, 1:] - delta2d[:, :-1]
    d_dy    = delta2d[1:, :] - delta2d[:-1, :]
    L_cgrad = (
        (d_dx ** 2).mean() + (d_dy ** 2).mean()
    ) / (G_init ** 2)

    L_cmag = (delta ** 2).mean()

    return L_wrap + lambda_cgrad * L_cgrad + lambda_cmag * L_cmag