"""Frozen principal observation protocols and metrics for IRIME-only reproduction."""
from __future__ import annotations
import hashlib, json
import numpy as np
from scipy.optimize import linear_sum_assignment
GOLDEN_FRACTION = (np.sqrt(5.0)-1.0)/2.0
PAPER_THETA_TRUE=np.array([30.42,120.27,240.51])
PAPER_PHI_TRUE=np.array([60.39,29.42,45.55])
H=2.5
B=896
TRIALS=500
SNR_VALUES=(-5,-3,-1,1,3,5)
SNAPSHOT_VALUES=(50,100,200,300,500)
SOURCE_COUNTS=tuple(range(2,10))
SNAPSHOTS=1024
MAIN_SPACINGS=(.25,.5,1,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5,4,5,6,8,10)
CONTEXTS={
    "main_mid":{"theta_deg":83.17,"phi_deg":35.29,"bearing_deg":0.0},
    "low_elevation":{"theta_deg":11.83,"phi_deg":15.37,"bearing_deg":0.0},
    "mid_elevation":{"theta_deg":181.73,"phi_deg":35.17,"bearing_deg":0.0},
    "high_elevation":{"theta_deg":221.47,"phi_deg":55.13,"bearing_deg":0.0},
    "zenith_wrap":{"theta_deg":359.21,"phi_deg":68.43,"bearing_deg":0.0}}
LEGACY_PROTOCOL="SEARIME-V7H-R24-PERFORMANCE-REPLACEMENT-20260908-V1"
SOURCE_PROTOCOL="SPFRIME-SOURCE-COUNT-K2K8-PERSOURCE10DB-20260920-V1"
RESOLUTION_PROTOCOL="UCA12-FIXED-ELEVATION-AZIMUTH-RESOLUTION-20260907-V5"
MISSING_SPHERICAL_PENALTY_DEG=180.0
COMPLETE_SET_THRESHOLD_DEG=3.0


def paper_case() -> dict:
    return {
        "wavelength": 1.0,
        "snapshot": 100,
        "source_num": 3,
        "snr": 5.0,
        # Frozen main evidence uses deterministic orthogonal tones.  Alternative
        # waveform models are opt-in stress tests and must be recorded in their
        # own result metadata rather than silently changing the frozen protocol.
        "source_waveform_model": "deterministic_tones",
        "theta_true": PAPER_THETA_TRUE.copy(),
        "phi_true": PAPER_PHI_TRUE.copy(),
        "M": 12,
        "theta_range": np.arange(0, 361, 1, dtype=float),
        "phi_range": np.arange(0, 91, 1, dtype=float),
    }


def circular_array_positions(M: int = 12, wavelength: float = 1.0) -> np.ndarray:
    r = wavelength / (4 * np.sin(np.pi / M))
    array_position = np.zeros((M, 3), dtype=float)
    for i in range(M):
        angle = 2 * np.pi * i / M
        array_position[i] = [r * np.cos(angle), r * np.sin(angle), 0.0]
    return array_position


def make_source_signals(snapshot: int, freqs=(0.1, 0.2, 0.3), source_num: int | None = None) -> np.ndarray:
    t = np.arange(snapshot)
    if source_num is not None:
        source_num = int(source_num)
        if source_num <= len(freqs):
            freqs = tuple(freqs[:source_num])
        else:
            freqs = tuple(np.linspace(0.08, 0.08 + 0.04 * (source_num - 1), source_num))
    return np.array([np.exp(1j * 2 * np.pi * f * t) for f in freqs])


def regular_source_geometry(k: int, seed: int = 2026, phi_low: float = 20.0, phi_high: float = 70.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed) + 1_000_003 * int(k) + 31_337)
    rotation = float(rng.uniform(0.0, 360.0))
    phase = float(rng.uniform(0.0, 1.0))
    theta = np.mod(rotation + np.arange(k, dtype=float) * (360.0 / k), 360.0)
    u = np.mod(phase + np.arange(k, dtype=float) * GOLDEN_FRACTION, 1.0)
    phi = float(phi_low) + (float(phi_high) - float(phi_low)) * u
    order = np.argsort(theta)
    return theta[order].copy(), phi[order].copy()


def gen_steering_vector(theta, phi, array_pos, wavelength):
    """
    生成阵列流型向量
    
    参数:
        theta: 方位角(度)
        phi: 俯仰角(度)
        array_pos: 阵元位置坐标, shape=(N,3)
        wavelength: 波长
    
    返回:
        a: 流型向量
    """
    # 角度转弧度
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    
    # 计算方向矢量
    u = np.array([np.cos(phi)*np.cos(theta), np.cos(phi)*np.sin(theta), np.sin(phi)])
    
    # 计算相位差
    phase = -2 * np.pi / wavelength * (array_pos @ u)
    
    # 生成流型向量
    a = np.exp(1j * phase)
    
    return a


def gen_array_signal(theta_list, phi_list, s, array_pos, wavelength, snr_db=20):
    """
    生成阵列接收信号
    
    参数:
        theta_list: 信源方位角列表(度)
        phi_list: 信源俯仰角列表(度) 
        s: 信源信号, shape=(K,L), K为信源数,L为快拍数
        array_pos: 阵元位置坐标, shape=(N,3)
        wavelength: 波长
        snr_db: 信噪比(dB)
    
    返回:
        X: 阵列接收信号, shape=(N,L)
    """
    K = len(theta_list) # 信源数
    N = array_pos.shape[0] # 阵元数
    L = s.shape[1] # 快拍数
    
    # 构造流型矩阵
    A = np.zeros((N,K), dtype=complex)
    for k in range(K):
        A[:,k] = gen_steering_vector(theta_list[k], phi_list[k], array_pos, wavelength)
    
    # 生成接收信号
    X = A @ s
    
    # 加入噪声
    snr = 10**(snr_db/10)
    noise_power = np.sum(np.abs(X)**2)/(N*L*snr)
    noise = np.sqrt(noise_power/2) * (np.random.randn(N,L) + 1j*np.random.randn(N,L))
    X = X + noise
    
    return X


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32 - 1)


def unit_direction(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(np.asarray(theta_deg, dtype=float))
    phi = np.deg2rad(np.asarray(phi_deg, dtype=float))
    return np.column_stack(
        [np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)]
    )


def great_circle_matrix(
    theta_true: np.ndarray,
    phi_true: np.ndarray,
    theta_est: np.ndarray,
    phi_est: np.ndarray,
) -> np.ndarray:
    u = unit_direction(theta_true, phi_true)
    v = unit_direction(theta_est, phi_est)
    dot = np.clip(u @ v.T, -1.0, 1.0)
    return np.rad2deg(np.arccos(dot))


def spherical_match(
    theta_true: np.ndarray,
    phi_true: np.ndarray,
    theta_est: np.ndarray,
    phi_est: np.ndarray,
) -> dict:
    theta_true = np.asarray(theta_true, dtype=float).reshape(-1)
    phi_true = np.asarray(phi_true, dtype=float).reshape(-1)
    theta_est = np.asarray(theta_est, dtype=float).reshape(-1)
    phi_est = np.asarray(phi_est, dtype=float).reshape(-1)
    valid = np.isfinite(theta_est) & np.isfinite(phi_est)
    theta_est = theta_est[valid]
    phi_est = phi_est[valid]
    k = len(theta_true)
    errors = np.full(k, MISSING_SPHERICAL_PENALTY_DEG, dtype=float)
    matched_theta = np.full(k, np.nan)
    matched_phi = np.full(k, np.nan)
    if len(theta_est):
        cost = great_circle_matrix(theta_true, phi_true, theta_est, phi_est)
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            errors[r] = cost[r, c]
            matched_theta[r] = theta_est[c]
            matched_phi[r] = phi_est[c]
    trial_rmse = float(np.sqrt(np.mean(errors**2)))
    return {
        "errors_deg": errors,
        "matched_theta": matched_theta,
        "matched_phi": matched_phi,
        "trial_rmse_deg": trial_rmse,
        "complete_3deg": float(np.all(errors < COMPLETE_SET_THRESHOLD_DEG)),
        "max_error_deg": float(np.max(errors)),
        "n_detected": int(np.sum(np.isfinite(matched_theta))),
    }


def steering_derivatives(theta_deg: float, phi_deg: float, array_pos: np.ndarray, wavelength: float):
    theta = np.deg2rad(float(theta_deg))
    phi = np.deg2rad(float(phi_deg))
    a = gen_steering_vector(float(theta_deg), float(phi_deg), array_pos, wavelength)
    du_dtheta = np.array(
        [-np.cos(phi) * np.sin(theta), np.cos(phi) * np.cos(theta), 0.0], dtype=float
    )
    du_dphi = np.array(
        [-np.sin(phi) * np.cos(theta), -np.sin(phi) * np.sin(theta), np.cos(phi)], dtype=float
    )
    dphase_theta = -2.0 * np.pi / float(wavelength) * (array_pos @ du_dtheta)
    dphase_phi = -2.0 * np.pi / float(wavelength) * (array_pos @ du_dphi)
    return a, 1j * dphase_theta * a, 1j * dphase_phi * a


def conditional_crb(
    theta: np.ndarray,
    phi: np.ndarray,
    s: np.ndarray,
    array_pos: np.ndarray,
    wavelength: float,
    sigma2: float,
) -> dict:
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    k = len(theta)
    A = np.zeros((array_pos.shape[0], k), dtype=complex)
    dtheta = np.zeros_like(A)
    dphi = np.zeros_like(A)
    for i in range(k):
        a, dt, dp = steering_derivatives(theta[i], phi[i], array_pos, wavelength)
        A[:, i] = a
        dtheta[:, i] = dt
        dphi[:, i] = dp
    projector = np.eye(array_pos.shape[0], dtype=complex) - A @ np.linalg.pinv(A)
    D = np.column_stack([dtheta, dphi])
    source_index = np.r_[np.arange(k), np.arange(k)]
    gram = s @ s.conj().T
    gram_expanded = gram.T[np.ix_(source_index, source_index)]
    spatial = D.conj().T @ projector @ D
    fim = (2.0 / float(sigma2)) * np.real(spatial * gram_expanded)
    fim = 0.5 * (fim + fim.T)
    rank = int(np.linalg.matrix_rank(fim))
    cond = float(np.linalg.cond(fim))
    if rank < 2 * k:
        return {
            "crb_full_rank": False,
            "fim_rank": rank,
            "fim_condition": cond,
            "crb_joint_angular_deg": float("nan"),
            "crb_spherical_local_deg": float("nan"),
        }
    cov = np.linalg.pinv(fim, rcond=1e-12)
    diag = np.maximum(np.real(np.diag(cov)), 0.0)
    var_theta = diag[:k]
    var_phi = diag[k:]
    rad2deg = 180.0 / np.pi
    joint = float(np.sqrt(np.mean(var_theta + var_phi)) * rad2deg)
    spherical_local = float(
        np.sqrt(np.mean((np.cos(np.deg2rad(phi)) ** 2) * var_theta + var_phi)) * rad2deg
    )
    return {
        "crb_full_rank": True,
        "fim_rank": rank,
        "fim_condition": cond,
        "crb_joint_angular_deg": joint,
        "crb_spherical_local_deg": spherical_local,
    }


def circular_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the shortest unsigned azimuth difference in ``[0, 180]``.

    The modulo form is valid even when an optimizer or imported result reports
    an angle outside a single 360-degree turn.  The former ``min(d, 360-d)``
    expression became negative for differences greater than 360 degrees.
    """
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return np.abs((delta + 180.0) % 360.0 - 180.0)


def pair_doa_errors(
    theta_true: np.ndarray,
    phi_true: np.ndarray,
    theta_est: np.ndarray,
    phi_est: np.ndarray,
    missing_theta_penalty_deg: float = 180.0,
    missing_phi_penalty_deg: float = 90.0,
) -> dict:
    theta_true = np.asarray(theta_true, dtype=float).reshape(-1)
    phi_true = np.asarray(phi_true, dtype=float).reshape(-1)
    theta_est = np.asarray(theta_est, dtype=float).reshape(-1)
    phi_est = np.asarray(phi_est, dtype=float).reshape(-1)

    valid = np.isfinite(theta_est) & np.isfinite(phi_est)
    theta_est = theta_est[valid]
    phi_est = phi_est[valid]

    n_true = len(theta_true)
    n_est = len(theta_est)
    if n_true == 0:
        raise ValueError('theta_true must not be empty.')
    if len(phi_true) != n_true:
        raise ValueError('theta_true and phi_true must have the same length.')
    if len(phi_est) != n_est:
        raise ValueError('theta_est and phi_est must have the same length.')
    if missing_theta_penalty_deg <= 0 or missing_phi_penalty_deg <= 0:
        raise ValueError('Missing-target penalties must be positive.')

    matched_theta = np.full(n_true, np.nan)
    matched_phi = np.full(n_true, np.nan)
    # A missed source must not disappear from an aggregate localization metric.
    # Use the worst physically meaningful UCA angular errors as finite penalties,
    # so means, standard deviations, confidence intervals, and plots remain valid.
    theta_err = np.full(n_true, float(missing_theta_penalty_deg))
    phi_err = np.full(n_true, float(missing_phi_penalty_deg))

    if n_est > 0:
        theta_cost = circular_diff_deg(theta_true[:, None], theta_est[None, :])
        phi_cost = np.abs(phi_true[:, None] - phi_est[None, :])
        joint_cost = np.sqrt(theta_cost ** 2 + phi_cost ** 2)
        row_ind, col_ind = linear_sum_assignment(joint_cost)
    else:
        row_ind = np.array([], dtype=int)
        col_ind = np.array([], dtype=int)

    for r, c in zip(row_ind, col_ind):
        matched_theta[r] = theta_est[c]
        matched_phi[r] = phi_est[c]
        theta_err[r] = theta_cost[r, c]
        phi_err[r] = phi_cost[r, c]

    theta_mae = float(np.mean(theta_err))
    phi_mae = float(np.mean(phi_err))
    joint_rmse = float(np.sqrt(np.mean(theta_err ** 2 + phi_err ** 2)))

    return {
        'theta_mae': theta_mae,
        'phi_mae': phi_mae,
        'joint_rmse': joint_rmse,
        'matched_theta_est': matched_theta,
        'matched_phi_est': matched_phi,
        'theta_errors': theta_err,
        'phi_errors': phi_err,
        'n_detected': int(np.sum(np.isfinite(matched_theta))),
    }


def directions(theta, phi) -> np.ndarray:
    t, p = np.deg2rad(theta), np.deg2rad(phi)
    return np.column_stack((np.cos(p) * np.cos(t), np.cos(p) * np.sin(t), np.sin(p)))


def steering(points: np.ndarray, positions: np.ndarray, wavelength: float = 1.) -> np.ndarray:
    return np.exp((-2j * np.pi / wavelength) * (directions(points[:, 0], points[:, 1]) @ positions.T))


def great_circle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # atan2 keeps small angular differences accurate without arccos cancellation.
    u, v = directions(a[:, 0], a[:, 1]), directions(b[:, 0], b[:, 1])
    return np.rad2deg(np.arctan2(np.linalg.norm(np.cross(u[:, None], v[None, :]), axis=-1), u @ v.T))


def fixed_elevation_azimuth_pair(context_id: str, spacing_deg: float) -> np.ndarray:
    """Construct an equal-elevation pair with exact spherical separation."""
    if context_id not in CONTEXTS:
        raise ValueError(f"unknown context: {context_id}")
    spacing = float(spacing_deg)
    if not 0.0 < spacing < 180.0:
        raise ValueError("spacing_deg must be in (0, 180)")
    context = CONTEXTS[context_id]
    theta = float(context["theta_deg"])
    phi = float(context["phi_deg"])
    phi_rad = np.deg2rad(phi)
    cosine_increment = (
        np.cos(np.deg2rad(spacing)) - np.sin(phi_rad) ** 2
    ) / max(np.cos(phi_rad) ** 2, np.finfo(float).eps)
    if cosine_increment < -1.0 - 1e-12 or cosine_increment > 1.0 + 1e-12:
        raise ValueError("declared spherical spacing is infeasible at this elevation")
    delta_theta = float(
        np.rad2deg(np.arccos(np.clip(cosine_increment, -1.0, 1.0)))
    )
    pair = np.asarray(
        [[theta, phi], [(theta + delta_theta) % 360.0, phi]], dtype=float
    )
    measured = float(great_circle(pair[:1], pair[1:])[0, 0])
    if not np.isclose(measured, spacing, atol=2e-11, rtol=0.0):
        raise AssertionError(f"pair construction produced {measured} deg, expected {spacing}")
    return pair


def seed_for(context_id: str, spacing_deg: float, trial: int, role: int) -> int:
    context_index = tuple(CONTEXTS).index(context_id) + 1
    sequence = np.random.SeedSequence(
        [
            20260909,
            context_index,
            round(float(spacing_deg) * 1000),
            int(trial),
            int(role),
        ]
    )
    return int(sequence.generate_state(1)[0])


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def resolution_metrics(cfg: dict, theta, phi) -> dict:
    truth = np.column_stack((cfg["theta_true"], cfg["phi_true"]))
    true_spacing = float(great_circle(truth[:1], truth[1:])[0, 0])
    try:
        theta = np.asarray(theta, dtype=float).ravel()
        phi = np.asarray(phi, dtype=float).ravel()
    except (TypeError, ValueError):
        theta, phi = np.array([]), np.array([])
    valid = bool(
        theta.size == phi.size == 2
        and np.isfinite(theta).all()
        and np.isfinite(phi).all()
    )
    errors = np.full(2, np.nan)
    matched = np.full((2, 2), np.nan)
    estimated_spacing = float("nan")
    if valid:
        estimated = np.column_stack((theta, phi))
        costs = great_circle(truth, estimated)
        rows, columns = linear_sum_assignment(costs ** 2)
        errors[rows] = costs[rows, columns]
        matched[rows] = estimated[columns]
        estimated_spacing = float(great_circle(estimated[:1], estimated[1:])[0, 0])
    primary = bool(valid and float(np.max(errors)) < 0.49 * true_spacing)
    strict = bool(primary and estimated_spacing >= 0.5 * true_spacing)
    return {
        "estimate_valid": valid,
        "estimate_count": int(theta.size),
        "primary_resolution_success": int(primary),
        "strict_resolution_success": int(strict),
        "true_spherical_spacing_deg": true_spacing,
        "primary_tolerance_deg": 0.49 * true_spacing,
        "error_fixed_deg": float(errors[0]),
        "error_moving_deg": float(errors[1]),
        "mean_squared_error_deg2": float(np.mean(errors ** 2)),
        "trial_rmse_deg": float(np.sqrt(np.mean(errors ** 2))),
        "estimated_spherical_spacing_deg": estimated_spacing,
        "spacing_bias_deg": estimated_spacing - true_spacing,
        "collapse": bool(valid and estimated_spacing < 0.5 * true_spacing),
        "matched_directions_json": json.dumps(matched.tolist()),
    }



def main_observation(family, value, local_trial):
    """Reproduce the frozen SNR/snapshot cell and estimator seed exactly."""
    trial=70000+int(local_trial)
    condition=f"{value:g}dB" if family=="snr" else f"L{int(value)}"
    cfg=paper_case()
    cfg.update(snr=float(value) if family=="snr" else 0.0,
               snapshot=100 if family=="snr" else int(value))
    seed=stable_seed(LEGACY_PROTOCOL,family,condition,trial,"observation")
    np.random.seed(seed)
    pos=circular_array_positions(12,1.0)
    s=make_source_signals(cfg["snapshot"],source_num=3)
    x=gen_array_signal(cfg["theta_true"],cfg["phi_true"],s,pos,1.0,cfg["snr"])
    method_seed=stable_seed(LEGACY_PROTOCOL,family,condition,trial,"irime_compact")
    sigma2=np.mean(np.abs(np.column_stack([gen_steering_vector(t,p,pos,1.0)
        for t,p in zip(cfg["theta_true"],cfg["phi_true"])])@s)**2)/10**(cfg["snr"]/10)
    return cfg,pos,s,x,method_seed,float(sigma2),trial

def source_observation(k, local_trial):
    trial=int(local_trial)
    cfg=paper_case()
    theta,phi=regular_source_geometry(k,seed=stable_seed(SOURCE_PROTOCOL,"geometry",k,trial),phi_low=20,phi_high=70)
    cfg.update(source_num=int(k),snapshot=300,snr=10.0,theta_true=theta,phi_true=phi,
               source_waveform_model="iid_complex_gaussian")
    pos=circular_array_positions(12,1.0)
    rng=np.random.default_rng(stable_seed(SOURCE_PROTOCOL,"source",k,trial))
    s=(rng.standard_normal((k,300))+1j*rng.standard_normal((k,300)))/np.sqrt(2.0)
    s/=np.sqrt(np.mean(np.abs(s)**2,axis=1,keepdims=True))
    a=np.column_stack([gen_steering_vector(t,p,pos,1.0) for t,p in zip(theta,phi)])
    rng=np.random.default_rng(stable_seed(SOURCE_PROTOCOL,"noise",k,trial))
    noise=np.sqrt(.1/2)*(rng.standard_normal((12,300))+1j*rng.standard_normal((12,300)))
    x=a@s+noise
    seed=stable_seed(SOURCE_PROTOCOL,"method",k,trial,"irime_compact")
    return cfg,pos,s,x,seed,.1,trial

def resolution_observation(spacing, snr, local_trial, context="main_mid"):
    trial=80000+int(local_trial)
    points=fixed_elevation_azimuth_pair(context,float(spacing))
    cfg=paper_case()
    cfg.update(source_num=2,snapshot=1024,snr=float(snr),theta_true=points[:,0],phi_true=points[:,1])
    pos=circular_array_positions(12,1.0)
    rng=np.random.default_rng(seed_for(context,spacing,trial,0))
    s=(rng.standard_normal((2,1024))+1j*rng.standard_normal((2,1024)))/np.sqrt(2.0)
    rng=np.random.default_rng(seed_for(context,spacing,trial,1))
    noise=(rng.standard_normal((12,1024))+1j*rng.standard_normal((12,1024)))/np.sqrt(2.0)
    sigma2=10**(-float(snr)/10)
    x=steering(points,pos).T@s+np.sqrt(sigma2)*noise
    # The original METHODS tuple gives IRIME index 17, hence role 37.
    seed=seed_for(context,spacing,trial,37)
    return cfg,pos,s,x,seed,sigma2,trial

def efficiency_observation(local_trial):
    seed=180001+int(local_trial)
    cfg=paper_case();cfg.update(snr=0.0,snapshot=100,source_num=3)
    pos=circular_array_positions(12,1.0)
    s=make_source_signals(100,source_num=3)
    np.random.seed(seed)
    x=gen_array_signal(cfg["theta_true"],cfg["phi_true"],s,pos,1.0,0.0)
    return cfg,pos,s,x,seed,float(np.mean(np.abs(np.column_stack([gen_steering_vector(t,p,pos,1.0)
        for t,p in zip(cfg["theta_true"],cfg["phi_true"])])@s)**2)),seed
