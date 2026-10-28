"""Differential Subspace Correction (DSC / DSD) — single canonical implementation.

论文 §3.3「差分子空间修正」的唯一实现。此前该机制在
``irime_v7_compact.py`` 与 ``irime_v7_compact_switchable.py`` 中各有一份副本，
且两份的边界处理不同：

* 旧副本（switchable）在质心优化和 1-D 间隔搜索中都**不检查俯仰边界**，
  越界候选仍参与打分；
* 新副本（compact）在质心目标、候选构造、最终接受三处都做了
  俯仰边界拒绝 + 方位周期回绕。

论文 §3.3 明确要求：

    "方位角按 [0°, 360°) 周期回绕；越出允许俯仰角范围的候选予以拒绝。"
    "每个对称候选在评分前都对方位角进行周期回绕；俯仰角超出 [0°, 90°]
     的候选不参与评分。"

因此以**新副本**为准（本模块），两条代码路径统一导入此处，
使主实验（compact）与消融实验（switchable）跑的是同一个 D 机制。

公式对应关系
------------
* 式 (24)(25)   : 2x2 Gram 条件数选对，κ_th = 10
* 式 (28)       : 导向矢量对 (θ, φ) 的解析导数（含 deg→rad 的 π/180 因子）
* 式 (29)(30)   : 质心处导数矩阵 D_c 与去共同模态后的 D^⊥
* 式 (31)       : 复最小二乘求分离轴系数 z
* 式 (32)(33)   : 特征向量相位不变化 ψ = ½·arg(zᵀz)，ū = Re{z·e^{-jψ}}
* 式 (35)       : 局部球面度量归一化（含 |cos φ_c| 的 1e-3 下限）
* 式 (36)       : 仅当 F̃_DML 下降时接受
"""
from __future__ import annotations

import numpy as np
import scipy.optimize as opt

from irime_v3_clean import EPS, _steering

__all__ = ["differential_subspace_decouple", "steering_derivatives"]

# 论文 §3.3 / Algorithm 1: 条件数门限 κ_th = 10
CONDITION_NUMBER_THRESHOLD = 10.0
# 论文 §4.1: 局部球面度量归一化时 |cos φ_c| 的数值下限
COS_PHI_FLOOR = 1e-3
# 论文 §3.3: 质心最多 20 次 Nelder–Mead 迭代
CENTROID_MAX_ITER = 20
# 论文 §3.3: 一维有界搜索最多 15 次迭代、绝对收敛容差 0.02°
INTERVAL_MAX_ITER = 15
INTERVAL_XATOL_DEG = 0.02
INTERVAL_MIN_S_DEG = 0.05
# 论文 §3.3: s_max = min{2.5°, (180/π)λ0/(8 d_max)}
INTERVAL_MAX_S_DEG = 2.5


def steering_derivatives(point: np.ndarray, array_pos: np.ndarray, wavelength: float):
    """论文式 (28)：UCA 导向矢量对 (θ, φ) 的解析偏导（**角度制**）。

    返回 ``(a, da/dθ_deg, da/dφ_deg)``。

    式 (28) 的弧度形式为
        [∂a/∂ζ]_m = -j (2π/λ0) r_mᵀ (∂d/∂ζ) [a]_m ,
    再乘以 π/180 转换到"度"（论文 §3.3 的 deg-导数约定）。
    """
    theta_rad = np.deg2rad(point[0])
    phi_rad = np.deg2rad(point[1])
    deg_per_rad = np.pi / 180.0
    # 式 (26)(27)：方向矢量对弧度的偏导
    dk_dtheta = (
        np.array(
            [
                -np.cos(phi_rad) * np.sin(theta_rad),
                np.cos(phi_rad) * np.cos(theta_rad),
                0.0,
            ]
        )
        * deg_per_rad
    )
    dk_dphi = (
        np.array(
            [
                -np.sin(phi_rad) * np.cos(theta_rad),
                -np.sin(phi_rad) * np.sin(theta_rad),
                np.cos(phi_rad),
            ]
        )
        * deg_per_rad
    )

    a = _steering(point[None, :], array_pos, wavelength)[:, 0]
    k_factor = -2.0 * np.pi / float(wavelength)
    da_dtheta = 1j * (k_factor * (array_pos @ dk_dtheta)) * a
    da_dphi = 1j * (k_factor * (array_pos @ dk_dphi)) * a
    return a, da_dtheta, da_dphi


def differential_subspace_decouple(
    estimates: np.ndarray,
    covariance: np.ndarray,
    array_pos: np.ndarray,
    wavelength: float,
    theta_bounds: tuple[float, float],
    phi_bounds: tuple[float, float],
) -> tuple[np.ndarray, dict]:
    """论文 §3.3：对条件数最差的方向对执行差分子空间修正。

    参数
    ----
    estimates : (K, 2) 当前方向集合（度）
    covariance : (M, M) 秩稳定化协方差 R̃x（论文式 (19)）
    array_pos : (M, 3) 阵元位置
    theta_bounds, phi_bounds : 搜索范围（度）

    返回
    ----
    (修正后方向集合, 诊断字典)。未触发或未改善时原样返回 ``estimates``。
    """
    calls = {"centroid_fitness": 0, "set_dml_fitness": 0}
    k_sources = len(estimates)
    def diagnostic(data):
        return dict(data, fitness_calls=dict(calls),
                    equivalent_angle_evaluations=int(calls["centroid_fitness"] + k_sources * calls["set_dml_fitness"]),
                    evaluation_count_rule="centroid calls + K*set-DML calls; includes boundary rejections, excludes geometry/derivatives")
    if k_sources < 2:
        return estimates, diagnostic({"applied": False, "reason": "k_less_than_2"})

    sensor_count = len(array_pos)
    cov_tr = float(np.real(np.trace(covariance)))
    theta_span = float(theta_bounds[1] - theta_bounds[0])

    # ---- 1. 式 (24)(25)：2x2 Gram 条件数，取最差方向对 ----------------
    closest_pair = None
    max_cond = 0.0
    for i in range(k_sources):
        for j in range(i + 1, k_sources):
            A2 = _steering(estimates[[i, j]], array_pos, wavelength)
            G = A2.conj().T @ A2
            w = np.linalg.eigvalsh(G)
            cond = float(w[-1] / max(w[0], 1e-12))
            if cond > max_cond:
                max_cond = cond
                closest_pair = (i, j)

    if closest_pair is None or max_cond < CONDITION_NUMBER_THRESHOLD:
        return estimates, diagnostic({
            "applied": False,
            "max_cond": max_cond,
            "reason": "manifold_well_conditioned",
        })

    i, j = closest_pair
    # ---- Q_o = I - P_{A_{-{i,j}}}；K=2 时 Q_o = I_M（论文 §3.3）----------
    other_idx = [idx for idx in range(k_sources) if idx not in (i, j)]
    if other_idx:
        A_other = _steering(estimates[other_idx], array_pos, wavelength)
        Q = np.eye(sensor_count, dtype=complex) - A_other @ np.linalg.pinv(A_other)
    else:
        Q = np.eye(sensor_count, dtype=complex)

    R_proj = Q @ covariance @ Q
    _w_cov, v_cov = np.linalg.eigh(R_proj)
    v1 = v_cov[:, -1]  # 质心模态（和波束）
    v2 = v_cov[:, -2]  # 差分模态（差波束）

    # 当前 F̃_DML（式 (16)，用 R̃x）
    A_curr = _steering(estimates, array_pos, wavelength)
    P_curr = A_curr @ np.linalg.pinv(A_curr)
    curr_loss = 1.0 - float(np.real(np.trace(P_curr @ covariance))) / max(cov_tr, EPS)
    calls["set_dml_fitness"] += 1

    # ---- 2. 质心对齐 v1（式 (21) 的近源区间版本）----------------------
    # 初始方位角取最短圆周弧中点，初始俯仰角取算术平均（论文 §3.3）。
    p_mid = 0.5 * (estimates[i] + estimates[j])
    theta_diff = (
        estimates[j, 0] - estimates[i, 0] + 0.5 * theta_span
    ) % theta_span - 0.5 * theta_span
    p_mid[0] = estimates[i, 0] + 0.5 * theta_diff
    p_mid[0] = (p_mid[0] - theta_bounds[0]) % theta_span + theta_bounds[0]
    p_mid[1] = np.clip(p_mid[1], *phi_bounds)

    def cent_obj(pt: np.ndarray) -> float:
        calls["centroid_fitness"] += 1
        if not np.all(np.isfinite(pt)) or not (phi_bounds[0] <= pt[1] <= phi_bounds[1]):
            return 1e10
        pt = pt.copy()
        pt[0] = (pt[0] - theta_bounds[0]) % theta_span + theta_bounds[0]
        a = _steering(pt[None, :], array_pos, wavelength)[:, 0]
        Qa = Q @ a
        den = float(np.real(np.vdot(Qa, Qa)))
        if den < 128.0 * EPS:
            return 1e10
        return -float(np.abs(np.vdot(v1, Qa)) ** 2) / den

    res_c = opt.minimize(
        cent_obj,
        x0=p_mid,
        method="Nelder-Mead",
        options={"maxiter": CENTROID_MAX_ITER},
    )
    cent = np.asarray(res_c.x, dtype=float)
    cent[0] = (cent[0] - theta_bounds[0]) % theta_span + theta_bounds[0]
    cent[1] = np.clip(cent[1], *phi_bounds)

    # ---- 3. 式 (29)-(33)：解析切空间投影，闭式求分离轴 u ---------------
    a_mid, da_dth, da_dph = steering_derivatives(cent, array_pos, wavelength)
    D = np.column_stack([da_dth, da_dph])

    Qa_mid = Q @ a_mid
    denom = float(np.real(np.vdot(Qa_mid, Qa_mid)))
    if denom < 128.0 * EPS:
        return estimates, diagnostic({"applied": False, "reason": "singular_centroid"})

    # 式 (30)：从投影导数中去除共同模态方向
    P_mid_perp = (
        np.eye(sensor_count, dtype=complex)
        - np.outer(Qa_mid, Qa_mid.conj()) / denom
    )
    D_proj = P_mid_perp @ (Q @ D)

    # 式 (31)：复最小二乘
    coeff, _, _, _ = np.linalg.lstsq(D_proj, v2, rcond=None)
    # 式 (32)：特征向量相位不变化；zᵀz 退化时用主导系数相位兜底
    z = np.sum(coeff * coeff)
    phase = (
        0.5 * np.angle(z)
        if abs(z) > 1e-15
        else np.angle(coeff[np.argmax(np.abs(coeff))])
    )
    # 式 (33)：相位对齐后的实方向
    u = np.real(coeff * np.exp(-1j * phase))
    cphi = max(abs(np.cos(np.deg2rad(cent[1]))), COS_PHI_FLOOR)
    # 式 (35)：局部球面度量归一化
    u_norm = float(np.sqrt((cphi * u[0]) ** 2 + u[1] ** 2))
    if u_norm < 1e-12:
        return estimates, diagnostic({"applied": False, "reason": "zero_tangent_norm"})
    u /= u_norm

    # ---- 4. 一维有界间隔搜索（论文 §3.3）------------------------------
    d_max = float(
        np.max(np.linalg.norm(array_pos[:, None, :] - array_pos[None, :, :], axis=-1))
    )
    rayleigh_half_bw_deg = float(np.rad2deg(wavelength / (2.0 * d_max)))
    # s_max = min{2.5°, (180/π)λ0/(8 d_max)}
    max_search_s = min(INTERVAL_MAX_S_DEG, rayleigh_half_bw_deg / 4.0)

    def candidate_at(s: float) -> np.ndarray | None:
        """构造对称方向对 c ± s·u；俯仰越界则**拒绝**（论文 §3.3）。"""
        cand = estimates.copy()
        cand[i] = cent - s * u
        cand[j] = cent + s * u
        pair = cand[[i, j], :]
        if (
            not np.all(np.isfinite(pair))
            or np.any(pair[:, 1] < phi_bounds[0])
            or np.any(pair[:, 1] > phi_bounds[1])
        ):
            return None
        cand[[i, j], 0] = (
            cand[[i, j], 0] - theta_bounds[0]
        ) % theta_span + theta_bounds[0]
        return cand

    def dml_1d(s: float) -> float:
        calls["set_dml_fitness"] += 1
        cand = candidate_at(s)
        if cand is None:
            return 1e10
        A_c = _steering(cand, array_pos, wavelength)
        P_c = A_c @ np.linalg.pinv(A_c)
        return 1.0 - float(np.real(np.trace(P_c @ covariance))) / max(cov_tr, EPS)

    res_line = opt.minimize_scalar(
        dml_1d,
        bounds=(INTERVAL_MIN_S_DEG, max_search_s),
        method="bounded",
        options={"xatol": INTERVAL_XATOL_DEG, "maxiter": INTERVAL_MAX_ITER},
    )
    # 式 (36)：F̃_DML 下降才接受
    if res_line.fun < curr_loss - 64.0 * EPS:
        cand = candidate_at(res_line.x)
        if cand is None:
            return estimates, diagnostic({"applied": False, "reason": "candidate_out_of_bounds"})
        return cand, diagnostic({
            "applied": True,
            "closest_pair": closest_pair,
            "max_cond": max_cond,
            "s_star_deg": float(res_line.x),
            "loss_drop": float(curr_loss - res_line.fun),
        })

    return estimates, diagnostic({
        "applied": False,
        "max_cond": max_cond,
        "reason": "no_loss_improvement",
    })
