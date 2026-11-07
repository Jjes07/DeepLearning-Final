import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from siren import (ImplicitPhaseNetwork,
                   LowFreqBackgroundNetwork,
                   build_coordinate_grid)
from reliability import (compute_reliability_map,
                          compute_wrapped_gradients)


def charbonnier(x, eps=1e-6):
    return torch.sqrt(x**2 + eps**2)


def spinr_unwrap(wrapped_phase_np,
                 n_iter_pretrain=300,
                 n_iter_refine=500,
                 lr_pretrain=1e-3,
                 lr_refine=1e-4,
                 lambda_tv=0.001,
                 lambda_cgrad=0.05,
                 lambda_cmag=0.05):
    """
    Phase unwrapping SPINR - versión que funciona.

    Paso 0: Integrar gradientes → fase aproximada desenvuelta
    Paso 1: Pretrain SIREN en fase integrada (lr alto)
    Paso 2: Refinar con L_wrap + L_cgrad + L_cmag
    """
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Usando: {device}")

    H, W = wrapped_phase_np.shape
    wrapped = torch.tensor(
        wrapped_phase_np, dtype=torch.float32, device=device
    )
    coords = build_coordinate_grid(H, W, device=device)

    # ── Mapa de confiabilidad y gradientes ─────────────────
    reliability = compute_reliability_map(wrapped)
    gx_raw, gy_raw = compute_wrapped_gradients(wrapped)

    print(f"Wrapped phase: [{wrapped.min():.3f}, {wrapped.max():.3f}]")
    print(f"gx_raw: [{gx_raw.min():.4f}, {gx_raw.max():.4f}]")
    print(f"Reliability: [{reliability.min():.3f}, {reliability.max():.3f}]")

    # ── Fase integrada (objetivo para pretrain) ─────────────
    gx_np = gx_raw.cpu().numpy()
    gy_np = gy_raw.cpu().numpy()
    phase_from_gx   = np.cumsum(gx_np, axis=1)
    phase_from_gy   = np.cumsum(gy_np, axis=0)
    phase_integrated = 0.5 * (phase_from_gx + phase_from_gy)
    phase_int_t      = torch.tensor(
        phase_integrated, dtype=torch.float32, device=device
    ).flatten()

    print(f"Fase integrada: [{phase_integrated.min():.3f}, "
          f"{phase_integrated.max():.3f}]")

    # ── Red SIREN ───────────────────────────────────────────
    network   = ImplicitPhaseNetwork(
        hidden_dim=256, n_layers=3, omega_0=10
    ).to(device)

    # ── PASO 1: Pretrain en fase integrada ─────────────────
    optimizer_pre = torch.optim.Adam(
        network.parameters(), lr=lr_pretrain
    )

    print("\n=== Paso 1: Pretrain en fase integrada ===")
    for i in tqdm(range(n_iter_pretrain)):
        optimizer_pre.zero_grad()
        phase = network(coords)
        loss  = ((phase - phase_int_t)**2).mean()
        loss.backward()
        optimizer_pre.step()

        if i % 100 == 0:
            with torch.no_grad():
                ph = network(coords)
            tqdm.write(
                f"  iter {i} | loss: {loss.item():.4f} | "
                f"fase: [{ph.min():.3f}, {ph.max():.3f}]"
            )

    with torch.no_grad():
        ph = network(coords)
    print(f"\nDespués pretrain: [{ph.min():.3f}, {ph.max():.3f}]")
    print(f"Fase integrada:   [{phase_integrated.min():.3f}, "
          f"{phase_integrated.max():.3f}]")

    # Fase inicial congelada como referencia
    with torch.no_grad():
        phase_init = network(coords).reshape(H, W).detach()

    # ── PASO 2: Refinar con física ──────────────────────────
    optimizer2 = torch.optim.Adam(
        network.parameters(), lr=lr_refine
    )

    pw       = wrapped.reshape(-1)
    phi_init = phase_init.reshape(-1)

    print("\n=== Paso 2: Refinamiento con física ===")
    for i in tqdm(range(n_iter_refine)):
        optimizer2.zero_grad()

        phase_hat = network(coords)
        delta     = phase_hat - phi_init

        # L_wrap: consistencia periódica (ec. 9)
        residual = torch.atan2(
            torch.sin(pw - phase_hat),
            torch.cos(pw - phase_hat)
        )
        L_wrap = charbonnier(residual).mean()

        # G_init: gradiente RMS de fase inicial (ec. 12)
        phi2d   = phi_init.reshape(H, W)
        dx_init = phi2d[:, 1:] - phi2d[:, :-1]
        dy_init = phi2d[1:, :] - phi2d[:-1, :]
        G_init  = torch.sqrt(
            (dx_init**2).mean() + (dy_init**2).mean()
        ) + 1e-6

        # L_cgrad: variación espacial de δ (ec. 10)
        delta2d = delta.reshape(H, W)
        d_dx    = delta2d[:, 1:] - delta2d[:, :-1]
        d_dy    = delta2d[1:, :] - delta2d[:-1, :]
        L_cgrad = (
            (d_dx**2).mean() + (d_dy**2).mean()
        ) / (G_init**2)

        # L_cmag: magnitud de δ (ec. 11)
        L_cmag = ((delta / torch.pi)**2).mean()

        loss = (L_wrap
                + lambda_cgrad * L_cgrad
                + lambda_cmag  * L_cmag)
        loss.backward()
        optimizer2.step()

        if i % 100 == 0:
            with torch.no_grad():
                ph = network(coords)
            tqdm.write(
                f"  iter {i} | "
                f"L_wrap: {L_wrap.item():.4f} | "
                f"L_cgrad: {L_cgrad.item():.4f} | "
                f"fase: [{ph.min():.3f}, {ph.max():.3f}]"
            )

    with torch.no_grad():
        unwrapped = network(coords).reshape(H, W).cpu().numpy()

    return unwrapped


def compensate_background(unwrapped_np, reliability_np,
                           n_iter=500, lr=1e-3):
    """
    Compensación de background con ratio adaptativo.
    Paper sección 2.3, ecuaciones 16-18.
    """
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    H, W = unwrapped_np.shape

    phase       = torch.tensor(
        unwrapped_np, dtype=torch.float32, device=device
    )
    reliability = torch.tensor(
        reliability_np, dtype=torch.float32, device=device
    )
    coords = build_coordinate_grid(H, W, device=device)

    # ── Ratio adaptativo (ec. 16) ──────────────────────────
    dxx = phase[:, 2:] - 2*phase[:, 1:-1] + phase[:, :-2]
    dyy = phase[2:, :] - 2*phase[1:-1, :] + phase[:-2, :]
    dxx = F.pad(dxx, (1, 1))
    dyy = F.pad(dyy, (0, 0, 1, 1))
    second_order = (dxx.abs() + dyy.abs())

    so_norm = (second_order - second_order.min()) / (
        second_order.max() - second_order.min() + 1e-8
    )

    # Background: baja variación de segundo orden + alta reliability
    bg_score = (1.0 - so_norm) * reliability
    rho_raw  = (bg_score > bg_score.median()).float().mean().item()
    rho      = float(torch.tensor(rho_raw).clamp(0.3, 0.9))
    N_keep   = int(H * W * rho)

    print(f"\nRatio background adaptativo: {rho:.3f} "
          f"({N_keep} píxeles de {H*W})")

    # ── Red de background (128, 2 capas, omega_bg=1) ────────
    bg_net    = LowFreqBackgroundNetwork(
        hidden_dim=128, n_layers=2, omega_0=1.0
    ).to(device)
    optimizer = torch.optim.Adam(bg_net.parameters(), lr=lr)

    print("\n=== Etapa 3: Background Compensation ===")
    for i in tqdm(range(n_iter)):
        optimizer.zero_grad()
        bg_pred   = bg_net(coords)
        residuals = (phase.flatten() - bg_pred).abs()
        sorted_r, _ = torch.sort(residuals)
        threshold   = sorted_r[N_keep - 1]
        mask        = residuals <= threshold
        loss        = residuals[mask].mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        background = bg_net(coords).reshape(H, W).cpu().numpy()

    compensated = unwrapped_np - background
    return compensated, background


def build_zernike_basis(H, W, n_terms=15):
    """
    Construye base de polinomios de Zernike normalizados
    sobre una grilla de coordenadas [-1, 1].
    Retorna matriz Z de shape (H*W, n_terms).
    """
    # Coordenadas normalizadas [-1, 1]
    y = np.linspace(-1, 1, H)
    x = np.linspace(-1, 1, W)
    XX, YY = np.meshgrid(x, y)
    R   = np.sqrt(XX**2 + YY**2)
    Phi = np.arctan2(YY, XX)

    # Máscara circular (radio <= 1)
    mask = R <= 1.0

    # Polinomios de Zernike estándar (hasta orden 4)
    Z = np.zeros((H * W, n_terms))

    # Z1: piston
    Z[:, 0] = 1.0
    # Z2: tilt x
    Z[:, 1] = 2 * R.flatten() * np.cos(Phi.flatten())
    # Z3: tilt y
    Z[:, 2] = 2 * R.flatten() * np.sin(Phi.flatten())
    # Z4: defocus
    Z[:, 3] = np.sqrt(3) * (2*R**2 - 1).flatten()
    # Z5: astigmatismo oblicuo
    Z[:, 4] = np.sqrt(6) * (R**2 * np.sin(2*Phi)).flatten()
    # Z6: astigmatismo horizontal
    Z[:, 5] = np.sqrt(6) * (R**2 * np.cos(2*Phi)).flatten()
    # Z7: coma y
    Z[:, 6] = np.sqrt(8) * ((3*R**3 - 2*R) * np.sin(Phi)).flatten()
    # Z8: coma x
    Z[:, 7] = np.sqrt(8) * ((3*R**3 - 2*R) * np.cos(Phi)).flatten()
    # Z9: trefoil y
    Z[:, 8] = np.sqrt(8) * (R**3 * np.sin(3*Phi)).flatten()
    # Z10: trefoil x
    Z[:, 9] = np.sqrt(8) * (R**3 * np.cos(3*Phi)).flatten()
    # Z11: aberración esférica
    Z[:, 10] = np.sqrt(5) * (6*R**4 - 6*R**2 + 1).flatten()
    # Z12-Z15: orden 4
    if n_terms > 11:
        Z[:, 11] = np.sqrt(10) * ((4*R**4 - 3*R**2) * np.sin(2*Phi)).flatten()
        Z[:, 12] = np.sqrt(10) * ((4*R**4 - 3*R**2) * np.cos(2*Phi)).flatten()
        Z[:, 13] = np.sqrt(10) * (R**4 * np.sin(4*Phi)).flatten()
        Z[:, 14] = np.sqrt(10) * (R**4 * np.cos(4*Phi)).flatten()

    return Z, mask.flatten()


def compensate_background_zernike(unwrapped_np, reliability_np,
                                   n_terms=15,
                                   n_iter_cauchy=20,
                                   n_bandwidth=5,
                                   n_iter_corr=10,
                                   cauchy_scale=1.0,
                                   ridge=1e-6):
    """
    Compensación de aberraciones con Zernike + Correntropy.

    Implementa el método de dos etapas:
    Etapa 1: Diferencias multiscala + pesos Cauchy (inicialización robusta)
    Etapa 2: Maximum Correntropy con annealing de bandwidth (refinamiento)

    Referencia: ecuaciones 18-38 del paper de compensación.

    Parámetros:
        n_terms:       número de polinomios de Zernike (15 = hasta orden 4)
        n_iter_cauchy: iteraciones de Cauchy IRLS en Etapa 1
        n_bandwidth:   niveles de annealing de bandwidth en Etapa 2
        n_iter_corr:   iteraciones por nivel de bandwidth
        cauchy_scale:  parámetro γ de peso Cauchy
        ridge:         regularización ridge para estabilidad numérica
    """
    H, W = unwrapped_np.shape
    phi  = unwrapped_np.flatten()
    r    = reliability_np.flatten()

    print("\n=== Compensación Zernike + Correntropy ===")

    # ── Construir base de Zernike ───────────────────────────
    Z, circ_mask = build_zernike_basis(H, W, n_terms)
    print(f"  Base Zernike: {n_terms} términos, "
          f"{circ_mask.sum()} píxeles válidos")

    # ── ETAPA 1: Diferencias multiscala + Cauchy ────────────
    print("  Etapa 1: Inicialización diferencial Cauchy...")

    # Construir observaciones diferenciales multiscala
    # Escalas p = 1, 2, 4 — primer y segundo orden
    phi_2d = unwrapped_np.copy()
    obs_list = []
    A_list   = []

    for p in [1, 2, 4]:
        # Primer orden x: normalizado por sqrt(2)
        diff_x = (phi_2d[:, p:] - phi_2d[:, :-p]) / np.sqrt(2)
        dZ_x   = (Z.reshape(H, W, n_terms)[:, p:, :]
                  - Z.reshape(H, W, n_terms)[:, :-p, :]) / np.sqrt(2)
        obs_list.append(diff_x.flatten())
        A_list.append(dZ_x.reshape(-1, n_terms))

        # Primer orden y: normalizado por sqrt(2)
        diff_y = (phi_2d[p:, :] - phi_2d[:-p, :]) / np.sqrt(2)
        dZ_y   = (Z.reshape(H, W, n_terms)[p:, :, :]
                  - Z.reshape(H, W, n_terms)[:-p, :, :]) / np.sqrt(2)
        obs_list.append(diff_y.flatten())
        A_list.append(dZ_y.reshape(-1, n_terms))

        # Segundo orden xx: normalizado por sqrt(6)
        if p < H - p:
            diff_xx = (phi_2d[:, 2*p:] - 2*phi_2d[:, p:-p]
                       + phi_2d[:, :-2*p]) / np.sqrt(6)
            dZ_xx   = (Z.reshape(H, W, n_terms)[:, 2*p:, :]
                       - 2*Z.reshape(H, W, n_terms)[:, p:-p, :]
                       + Z.reshape(H, W, n_terms)[:, :-2*p, :]) / np.sqrt(6)
            obs_list.append(diff_xx.flatten())
            A_list.append(dZ_xx.reshape(-1, n_terms))

    # Apilar todas las observaciones diferenciales
    d_obs = np.concatenate(obs_list)
    A_mat = np.vstack(A_list)

    # Inicialización con mínimos cuadrados
    a_ls, _, _, _ = np.linalg.lstsq(A_mat, d_obs, rcond=None)
    a_cur = a_ls.copy()

    # IRLS con pesos Cauchy (ec. 22-23)
    for it in range(n_iter_cauchy):
        r_res   = d_obs - A_mat @ a_cur
        r_med   = np.median(r_res)
        r_mad   = 1.4826 * np.median(np.abs(r_res - r_med)) + 1e-8
        r_norm  = (r_res - r_med) / (cauchy_scale * r_mad)
        w_cauchy = 1.0 / (1.0 + r_norm**2)

        W_diag  = w_cauchy
        AtW     = A_mat.T * W_diag
        a_new   = np.linalg.solve(
            AtW @ A_mat + ridge * np.eye(n_terms),
            AtW @ d_obs
        )
        delta_a = np.linalg.norm(a_new - a_cur) / (
            np.linalg.norm(a_cur) + 1e-8
        )
        a_cur   = a_new
        if delta_a < 1e-6:
            print(f"    Cauchy convergió en iter {it+1}")
            break

    a_init = a_cur.copy()
    print(f"  Coeficientes Zernike iniciales: "
          f"[{a_init.min():.4f}, {a_init.max():.4f}]")

    # ── ETAPA 2: Maximum Correntropy con annealing ──────────
    print("  Etapa 2: Refinamiento con Correntropy annealing...")

    # Estimar bandwidth desde residuos (ec. 33-35)
    phi_aber_init = Z @ a_init
    res_init      = phi - phi_aber_init

    # Noise scale desde diferencias mixtas de segundo orden
    res_2d  = res_init.reshape(H, W)
    Q       = (res_2d[1:, 1:] - res_2d[1:, :-1]
               - res_2d[:-1, 1:] + res_2d[:-1, :-1])
    q_vec   = Q.flatten()
    sigma_n = 1.4826 * np.median(np.abs(q_vec - np.median(q_vec))) / 2.0

    # Residual scale global
    sigma_e = 1.4826 * np.median(np.abs(res_init - np.median(res_init)))

    # Bandwidths inicial y final (ec. 35)
    alpha_bw, beta_bw, gamma_bw = 3.0, 2.0, 0.5
    sigma_f = max(alpha_bw * sigma_n, gamma_bw)
    sigma_i = max(beta_bw * sigma_e, sigma_f)

    # Secuencia geométrica de bandwidths (ec. 36)
    if sigma_i > sigma_f:
        bandwidths = np.geomspace(sigma_i, sigma_f, n_bandwidth)
    else:
        bandwidths = np.ones(n_bandwidth) * sigma_f

    print(f"  Bandwidths: {sigma_i:.4f} → {sigma_f:.4f}")

    a_cur = a_init.copy()
    for b_idx, sigma_b in enumerate(bandwidths):
        for it in range(n_iter_corr):
            res_l  = phi - Z @ a_cur
            # Pesos Correntropy (ec. 29)
            v_l    = np.exp(-res_l**2 / (2 * sigma_b**2))
            # Actualización fixed-point (ec. 30)
            V_diag = v_l
            ZtV    = Z.T * V_diag
            a_new  = np.linalg.solve(
                ZtV @ Z + ridge * np.eye(n_terms),
                ZtV @ phi
            )
            delta_a = np.linalg.norm(a_new - a_cur) / (
                np.linalg.norm(a_cur) + 1e-8
            )
            a_cur   = a_new
            if delta_a < 1e-7:
                break

        print(f"  Bandwidth {b_idx+1}/{n_bandwidth}: "
              f"σ={sigma_b:.4f}, "
              f"coef=[{a_cur.min():.4f}, {a_cur.max():.4f}]")

    # ── Reconstruir y compensar aberración (ec. 37-38) ──────
    phi_aber  = (Z @ a_cur).reshape(H, W)

    # Corregir offset global usando zonas de alta reliability
    bg_mask   = reliability_np > np.percentile(reliability_np, 70)
    offset    = np.median(unwrapped_np[bg_mask] - phi_aber[bg_mask])
    phi_aber  = phi_aber + offset

    compensated = unwrapped_np - phi_aber

    # Asegurar que el fondo quede en 0
    bg_residual = np.median(compensated[bg_mask])
    compensated = compensated - bg_residual

    print(f"\n  Aberración estimada: "
          f"[{phi_aber.min():.3f}, {phi_aber.max():.3f}] rad")
    print(f"  Fase compensada:     "
          f"[{compensated.min():.3f}, {compensated.max():.3f}] rad")

    return compensated, phi_aber


def extract_sample_phase(compensated_np, reliability_np,
                          tau=10.0, alpha_s=1.0, beta_s=0.5,
                          mask_threshold=0.5, min_area=50):
    """
    Extracción adaptativa de fase de muestra.
    Paper sección 2.3, ecuaciones 19-20.

    S_o(x,y) = φ_b(x,y) si M_o=1
               0          si M_o=0
    """
    H, W = compensated_np.shape

    # ── Suavizar fase compensada ────────────────────────────
    phi_smooth = ndimage.gaussian_filter(compensated_np, sigma=2.0)

    # ── Gradiente local ─────────────────────────────────────
    grad_x   = np.gradient(phi_smooth, axis=1)
    grad_y   = np.gradient(phi_smooth, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # ── Candidatos de background ────────────────────────────
    bg_candidates = (
        (grad_mag < np.percentile(grad_mag, 40)) &
        (reliability_np > np.percentile(reliability_np, 40))
    )

    # ── Normalizar residuo con mediana y MAD ─────────────────
    residuo = compensated_np.copy()
    if bg_candidates.sum() > 0:
        bg_baseline = np.median(residuo[bg_candidates])
        bg_mad      = np.median(
            np.abs(residuo[bg_candidates] - bg_baseline)
        ) + 1e-8
    else:
        bg_baseline = np.median(residuo)
        bg_mad      = np.median(
            np.abs(residuo - bg_baseline)
        ) + 1e-8

    residuo_norm = (residuo - bg_baseline) / bg_mad

    # ── Detección adaptativa con quantiles ──────────────────
    low_thresh  = np.percentile(residuo_norm, 60)
    high_thresh = np.percentile(residuo_norm, 85)

    sample_candidates = (residuo_norm > low_thresh)

    # Componentes conectados con área y pico suficiente
    labeled, n_components = ndimage.label(sample_candidates)
    sample_support = np.zeros_like(sample_candidates, dtype=bool)

    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        comp_area = comp_mask.sum()
        comp_peak = residuo_norm[comp_mask].max()
        if comp_area >= min_area and comp_peak > high_thresh:
            sample_support |= comp_mask

    if sample_support.sum() == 0:
        sample_support = residuo_norm > low_thresh

    # ── Máscara suave con distancia signed (ec. 19) ─────────
    dist_inside  =  distance_transform_edt(sample_support)
    dist_outside = -distance_transform_edt(~sample_support)
    signed_dist  = np.where(
        sample_support, dist_inside, dist_outside
    )

    z_confidence = (residuo_norm - residuo_norm.min()) / (
        residuo_norm.max() - residuo_norm.min() + 1e-8
    )

    M_loc = 1.0 / (1.0 + np.exp(
        -(alpha_s * signed_dist / (tau + 1e-8)
          + beta_s * z_confidence)
    ))

    # Binarizar → M_o
    M_binary = (M_loc >= mask_threshold).astype(np.float32)

    # ── Extraer fase muestra (ec. 20) ───────────────────────
    # Preserva valores reales en muestra, fondo = 0
    sample_phase = compensated_np * M_binary

    print(f"\nExtracción de muestra:")
    print(f"  Región: {M_binary.sum():.0f} píxeles "
          f"({100*M_binary.mean():.1f}%)")
    if M_binary.sum() > 0:
        print(f"  Fase muestra: "
              f"[{sample_phase[M_binary>0].min():.3f}, "
              f"{sample_phase[M_binary>0].max():.3f}] rad")

    return sample_phase, M_binary, M_loc