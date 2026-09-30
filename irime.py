"""Reference implementation of IRIME for 2-D ML-DOA estimation.

Pipeline:
1) rank-stabilized recursive projected-subspace initialization;
2) differential-subspace decoupling (DSD);
3) two cyclic source-wise conditional RIME searches;
4) observation-driven bounded terminal refinement.

Angles at the public API are in degrees. The source count is assumed known.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize, minimize_scalar, nnls

EPS = np.finfo(float).eps


def steering(points, array_pos, wavelength):
    p = np.asarray(points, float)
    if p.ndim == 1:
        p = p[None, :]
    th, ph = np.deg2rad(p[:, 0]), np.deg2rad(p[:, 1])
    u = np.c_[np.cos(ph) * np.cos(th), np.cos(ph) * np.sin(th), np.sin(ph)]
    phase = -2 * np.pi / float(wavelength) * (np.asarray(array_pos, float) @ u.T)
    return np.exp(1j * phase)


def _wrap(x, lo, hi):
    return lo + np.mod(np.asarray(x) - lo, hi - lo)


def _projector(A, m):
    if A.size == 0:
        return np.zeros((m, m), complex)
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    if not len(s) or s[0] <= 0:
        return np.zeros((m, m), complex)
    U = U[:, s > EPS * max(A.shape) * s[0]]
    return U @ U.conj().T


def _stabilize(R, k):
    mu, E = np.linalg.eigh((R + R.conj().T) / 2)
    q = np.argsort(mu)[::-1]
    mu, E = mu[q], E[:, q]
    sigma0 = float(np.mean(mu[k:]).real)
    mu[k:] = sigma0
    Rt = (E * mu[None, :]) @ E.conj().T
    return (Rt + Rt.conj().T) / 2, sigma0


def _loss(D, R, pos, wavelength):
    A = steering(D, pos, wavelength)
    P = _projector(A, len(pos))
    return float(np.trace((np.eye(len(pos)) - P) @ R).real /
                 max(np.trace(R).real, EPS))


def _score(points, V, Q, pos, wavelength):
    A = steering(points, pos, wavelength)
    QA = Q @ A
    den = np.sum(np.abs(QA) ** 2, axis=0).real
    num = np.sum(np.abs(V.conj().T @ QA) ** 2, axis=0).real
    return np.divide(num, den, out=np.full_like(num, -np.inf), where=den > 128 * EPS)


def _quadratic_peak(S, it, ip, ta, pa):
    nt, np_ = S.shape
    if ip <= 0 or ip >= np_ - 1:
        return None
    span = (ta[-1] - ta[0]) + (ta[1] - ta[0])
    rows, vals = [], []
    for ti in ((it - 1) % nt, it, (it + 1) % nt):
        dt = float(ta[ti] - ta[it])
        if dt > span / 2: dt -= span
        if dt < -span / 2: dt += span
        for pj in (ip - 1, ip, ip + 1):
            dp = float(pa[pj] - pa[ip])
            rows.append([1, dt, dp, .5*dt*dt, dt*dp, .5*dp*dp])
            vals.append(float(S[ti, pj]))
    c, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(vals), rcond=None)
    H = np.array([[c[3], c[4]], [c[4], c[5]]])
    if not np.all(np.linalg.eigvalsh(H) < -1e-12):
        return None
    try:
        d = -np.linalg.solve(H, c[1:3])
    except np.linalg.LinAlgError:
        return None
    if abs(d[0]) > abs(ta[1]-ta[0]) or abs(d[1]) > abs(pa[1]-pa[0]):
        return None
    return np.array([ta[it] + d[0], pa[ip] + d[1]])


def _derivatives(c, pos, wavelength, eps_deg=1e-3):
    e1, e2 = np.array([eps_deg, 0.]), np.array([0., eps_deg])
    d1 = (steering(c+e1, pos, wavelength)[:,0] -
          steering(c-e1, pos, wavelength)[:,0]) / (2*np.deg2rad(eps_deg))
    d2 = (steering(c+e2, pos, wavelength)[:,0] -
          steering(c-e2, pos, wavelength)[:,0]) / (2*np.deg2rad(eps_deg))
    return np.c_[d1, d2]


def _dsd(D, Rt, pos, wavelength, tb, pb):
    K, M = len(D), len(pos)
    if K < 2:
        return D, False
    pairs = []
    for i in range(K):
        for j in range(i+1, K):
            A = steering(D[[i,j]], pos, wavelength)
            pairs.append((np.linalg.cond(A.conj().T @ A), i, j))
    kappa, i, j = max(pairs)
    if not np.isfinite(kappa) or kappa <= 10:
        return D, False

    other = np.delete(D, [i,j], axis=0)
    Q = np.eye(M) - (_projector(steering(other, pos, wavelength), M) if len(other) else 0)
    Rq = (Q @ Rt @ Q + (Q @ Rt @ Q).conj().T) / 2
    w, V = np.linalg.eigh(Rq)
    V = V[:, np.argsort(w)[::-1]]
    v1, v2 = V[:,0], V[:,1]

    th = float(np.angle(np.mean(np.exp(1j*np.deg2rad(D[[i,j],0]))), deg=True) % 360)
    c0 = np.array([th, float(np.mean(D[[i,j],1]))])

    def cfun(x):
        c = np.array([float(_wrap(x[0], *tb)), float(np.clip(x[1], *pb))])
        b = Q @ steering(c, pos, wavelength)[:,0]
        return -float(abs(np.vdot(v1,b))**2 / max(np.vdot(b,b).real, EPS))

    r = minimize(cfun, c0, method="Nelder-Mead",
                 options={"maxiter":20, "xatol":1e-4, "fatol":1e-8})
    c = np.array([float(_wrap(r.x[0], *tb)), float(np.clip(r.x[1], *pb))])
    b = Q @ steering(c, pos, wavelength)[:,0]
    Dc = Q @ _derivatives(c, pos, wavelength)
    Dp = (np.eye(M) - np.outer(b,b.conj())/max(np.vdot(b,b).real,EPS)) @ Dc
    z, *_ = np.linalg.lstsq(Dp, v2, rcond=None)
    psi = .5*np.angle(z.T@z) if abs(z.T@z) > 1e-14 else np.angle(z[np.argmax(abs(z))])
    u = np.real(z*np.exp(-1j*psi))
    nrm = math.sqrt((math.cos(math.radians(c[1]))*u[0])**2 + u[1]**2)
    if nrm <= 1e-12:
        return D, False
    u /= nrm

    dmax = max(np.linalg.norm(pos[a]-pos[b_]) for a in range(M) for b_ in range(a+1,M))
    smax = min(2.5, (180/np.pi)*wavelength/max(8*dmax, EPS))
    if smax <= .05:
        return D, False
    before = _loss(D, Rt, pos, wavelength)

    def sfun(s):
        C = D.copy()
        C[i], C[j] = c-s*u, c+s*u
        C[:,0], C[:,1] = _wrap(C[:,0], *tb), np.clip(C[:,1], *pb)
        return _loss(C, Rt, pos, wavelength)

    rr = minimize_scalar(sfun, bounds=(.05,smax), method="bounded",
                         options={"maxiter":15, "xatol":1e-4})
    if np.isfinite(rr.fun) and rr.fun < before:
        C = D.copy()
        C[i], C[j] = c-rr.x*u, c+rr.x*u
        C[:,0], C[:,1] = _wrap(C[:,0], *tb), np.clip(C[:,1], *pb)
        return C, True
    return D, False


def _seed(center, n, tb, pb, rng):
    n0 = max(1, n//3)
    base = np.c_[rng.uniform(tb[0],tb[1],n0), rng.uniform(pb[0],pb[1],n0)]
    opp = np.c_[tb[0]+tb[1]-base[:,0], pb[0]+pb[1]-base[:,1]]
    mid = np.array([(tb[0]+tb[1])/2, (pb[0]+pb[1])/2])
    qopp = mid + rng.random(base.shape)*(opp-mid)
    pop = np.vstack((np.asarray(center)[None,:], base, opp, qopp))[:n]
    if len(pop) < n:
        pop = np.vstack((pop, np.c_[rng.uniform(tb[0],tb[1],n-len(pop)),
                                    rng.uniform(pb[0],pb[1],n-len(pop))]))
    pop[:,0], pop[:,1] = _wrap(pop[:,0], *tb), np.clip(pop[:,1], *pb)
    return pop


def _conditional_values(points, Rt, Q, pos, wavelength):
    p = np.asarray(points,float)
    if p.ndim == 1: p = p[None,:]
    A = steering(p,pos,wavelength)
    QA = Q @ A
    den = np.sum(np.abs(QA)**2,axis=0).real
    num = np.sum(np.conj(QA)*(Rt@QA),axis=0).real
    gain = np.divide(num,den,out=np.full_like(num,-np.inf),where=den>128*EPS)
    return 1 - gain/max(np.trace(Rt).real,EPS)


def _rime(center, Rt, Q, pos, wavelength, tb, pb, B, rng):
    M = len(pos)
    TR = max(2, int(math.ceil(math.log2(M))))
    NR = max(4, int(B//(TR+3)))
    pop = _seed(center, NR, tb, pb, rng)
    f = _conditional_values(pop,Rt,Q,pos,wavelength)
    span = np.array([tb[1]-tb[0], pb[1]-pb[0]])
    W = 5.
    for t in range(1,TR+1):
        best = pop[np.argmin(f)].copy()
        r = ((2*rng.random()-1)*math.cos(math.pi*t/max(TR/10,1))*
             (1-round(t*W/TR)/W))
        E = math.sqrt(t/TR)
        P = pop.copy()
        P += (rng.random(pop.shape)<E)*(r*.08*span/(t+1))*rng.normal(size=pop.shape)
        q = (f.max()-f)/max(f.max()-f.min(),1e-12)
        hard = rng.random(pop.shape)<q[:,None]
        P = np.where(hard,best+.018*span/(t+1)*rng.normal(size=pop.shape),P)
        P[:,0],P[:,1] = _wrap(P[:,0],*tb),np.clip(P[:,1],*pb)
        fp = _conditional_values(P,Rt,Q,pos,wavelength)
        a = fp <= f
        pop[a],f[a] = P[a],fp[a]
    j = int(np.argmin(f))
    return pop[j], float(f[j])


def _covfit(D,R,pos,wavelength):
    A = steering(D,pos,wavelength)
    atoms = [np.outer(A[:,k],A[:,k].conj()) for k in range(A.shape[1])] + [np.eye(len(pos))]
    H = np.column_stack([np.r_[x.real.ravel(),x.imag.ravel()] for x in atoms])
    y = np.r_[R.real.ravel(),R.imag.ravel()]
    c,_ = nnls(H,y)
    F = sum(v*x for v,x in zip(c,atoms))
    loss = np.linalg.norm(R-F,"fro")**2/max(np.linalg.norm(R,"fro")**2,EPS)
    return float(loss), c[:-1], float(c[-1])


def _sml_form(D,Rt,pos,wavelength):
    A = steering(D,pos,wavelength)
    U,s,_ = np.linalg.svd(A,full_matrices=False)
    U = U[:,s>EPS*max(A.shape)*s[0]]
    if U.shape[1] != len(D): return 1e12
    C = U.conj().T@Rt@U
    sign,ld = np.linalg.slogdet((C+C.conj().T)/2 + 1e-12*np.eye(len(D)))
    if sign <= 0: return 1e12
    sig = max(float((np.trace(Rt).real-np.trace(C).real)/max(len(pos)-len(D),1)),1e-12)
    return float(ld + (len(pos)-len(D))*math.log(sig))


def irime(X,array_pos,wavelength,theta_grid,phi_grid,k_sources,*,h=2.5,B=896,
          seed=None,return_diagnostics=False):
    X,pos = np.asarray(X,complex),np.asarray(array_pos,float)
    M,L = X.shape
    K = int(k_sources)
    if pos.shape != (M,3) or not (1 <= K < M):
        raise ValueError("invalid array geometry or source count")
    tb = (float(np.min(theta_grid)),float(np.max(theta_grid)))
    pb = (float(np.min(phi_grid)),float(np.max(phi_grid)))
    ta = np.arange(tb[0],tb[1],h)
    pa = np.r_[np.arange(pb[0],pb[1],h),pb[1]]
    T,P = np.meshgrid(ta,pa,indexing="ij")
    grid = np.c_[T.ravel(),P.ravel()]

    R = X@X.conj().T/float(L)
    Rt,sigma0 = _stabilize(R,K)
    selected,evidence = [],[]
    for q in range(1,K+1):
        A0 = steering(np.asarray(selected),pos,wavelength) if selected else np.empty((M,0),complex)
        Q = np.eye(M)-_projector(A0,M)
        Rq = (Q@Rt@Q + (Q@Rt@Q).conj().T)/2
        w,V = np.linalg.eigh(Rq)
        V = V[:,np.argsort(w)[::-1][:K-q+1]]
        sc = _score(grid,V,Q,pos,wavelength).reshape(len(ta),len(pa))
        loc = sc == maximum_filter(sc,size=3,mode=("wrap","nearest"))
        peaks = sorted(((sc[i,j],i,j) for i,j in np.argwhere(loc)),reverse=True)
        _,it,ip = peaks[0]
        c = np.array([ta[it],pa[ip]])
        if K >= 3:
            z = _quadratic_peak(sc,it,ip,ta,pa)
            if z is not None:
                z[0],z[1] = _wrap(z[0],*tb),np.clip(z[1],*pb)
                if _score(z,V,Q,pos,wavelength)[0] > _score(c,V,Q,pos,wavelength)[0]:
                    c = z
        selected.append(c)
        evidence += [np.array([ta[i],pa[j]]) for _,i,j in peaks[1:6]]

    D,dsd_ok = _dsd(np.asarray(selected,float),Rt,pos,wavelength,tb,pb)
    rng = np.random.default_rng(seed)
    for _ in range(2):
        for k in range(K):
            other = np.delete(D,k,axis=0)
            Q = np.eye(M)-(_projector(steering(other,pos,wavelength),M) if len(other) else 0)
            f0 = _conditional_values(D[k],Rt,Q,pos,wavelength)[0]
            z,fz = _rime(D[k],Rt,Q,pos,wavelength,tb,pb,B,rng)
            if fz <= f0 + 64*EPS: D[k] = z

    if 2*K >= M:
        base,powers,noise = _covfit(D,R,pos,wavelength)
        if len(powers) and powers.min() <= noise and evidence:
            weak = int(np.argmin(powers))
            for e in evidence:
                C = D.copy(); C[weak] = e
                if np.linalg.matrix_rank(steering(C,pos,wavelength)) < K: continue
                f,_,_ = _covfit(C,R,pos,wavelength)
                if f < base: base,D = f,C
        criterion = lambda C:_sml_form(C,Rt,pos,wavelength)
        branch = "high-occupancy-sml-form"
    elif np.linalg.eigvalsh(R)[::-1][K-1].real <= (M/K)*sigma0 or dsd_ok:
        criterion = lambda C:_covfit(C,Rt,pos,wavelength)[0]
        branch = "weak-close-covariance-fit"
    else:
        criterion = lambda C:_loss(C,Rt,pos,wavelength)
        branch = "ordinary-dml-form"

    base = D.copy()
    def fun(delta):
        C = base + np.asarray(delta).reshape(K,2)
        C[:,0],C[:,1] = _wrap(C[:,0],*tb),np.clip(C[:,1],*pb)
        return criterion(C)
    bounds = []
    for k in range(K):
        bounds += [(-h,h),(max(-h,pb[0]-base[k,1]),min(h,pb[1]-base[k,1]))]
    r = minimize(fun,np.zeros(2*K),method="L-BFGS-B",bounds=bounds,options={"maxiter":150})
    if np.isfinite(r.fun) and r.fun <= fun(np.zeros(2*K)) + 64*EPS:
        D = base+r.x.reshape(K,2)
        D[:,0],D[:,1] = _wrap(D[:,0],*tb),np.clip(D[:,1],*pb)
    D = D[np.argsort(D[:,0])]
    out = (D[:,0].copy(),D[:,1].copy())
    if return_diagnostics:
        return (*out,{"h":h,"B":B,"dsd_accepted":bool(dsd_ok),"terminal_branch":branch})
    return out


__all__ = ["irime","steering"]
