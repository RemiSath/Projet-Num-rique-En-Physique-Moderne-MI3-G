# -*- coding: utf-8 -*-
"""
tunnel_barrier.py
=================
Résolution numérique de l'équation de Schrödinger 1D pour une barrière
rectangulaire de potentiel par la méthode de Crank-Nicolson.

Méthode  : Crank-Nicolson + Algorithme de Thomas (100% GausWP externe)
"""

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Importation de votre module externe contenant l'algorithme analytique
from PaquetOndeGauss_3G import GausWP

# ===========================================================================
# 1. PARAMÈTRES PHYSIQUES (unités atomiques : hbar = m = 1)
# ===========================================================================
@dataclass
class Params:
    # Particule
    hbar: float = 1.0
    m:    float = 1.0
    k0:   float = 3.0          # vecteur d'onde central
    sigma: float = 1.5         # largeur initiale du paquet (noté 'a' dans votre GausWP)

    # Barrière
    V0:   float = 3.0          # hauteur (V0 = 0 → particule libre)
    a:    float = 2.0          # largeur de la barrière de potentiel
    x_bar: float = 0.0         # bord gauche de la barrière

    # Grille
    x_min: float = -40.0
    x_max: float =  40.0
    nx:    int   = 1200        # points en espace
    t_max: float = 20.0
    nt:    int   = 8000        # pas de temps

    # Paquet initial
    x0:   float = -20.0       # position initiale du centre


def make_grids(p: Params):
    x  = np.linspace(p.x_min, p.x_max, p.nx)
    t  = np.linspace(0.0, p.t_max, p.nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    return x, t, dx, dt


# ===========================================================================
# 2. POTENTIEL
# ===========================================================================
def make_potential(x: np.ndarray, p: Params) -> np.ndarray:
    """Barrière rectangulaire [x_bar, x_bar + a] de hauteur V0."""
    V = np.zeros_like(x)
    mask = (x >= p.x_bar) & (x <= p.x_bar + p.a)
    V[mask] = p.V0
    return V


# ===========================================================================
# 3. MÉTHODE DE CRANK-NICOLSON (SOLVEUR THOMAS)
# ===========================================================================
def build_CN_matrices(V: np.ndarray, dx: float, dt: float, p: Params):
    n   = len(V)
    r   = 1j * p.hbar * dt / (4 * p.m * dx**2)
    d   = p.hbar * dt / (2 * p.hbar)

    diag_L  = 1 + 2*r + 1j * d / p.hbar * V
    off_L   = -r * np.ones(n - 1)

    ab = np.zeros((3, n), dtype=complex)
    ab[0, 1:]  = off_L
    ab[1, :]   = diag_L
    ab[2, :-1] = off_L

    diag_R = 1 - 2*r - 1j * d / p.hbar * V
    off_R  = r * np.ones(n - 1)

    return ab, diag_R, off_R


def apply_R(psi: np.ndarray, diag_R, off_R) -> np.ndarray:
    rhs         = diag_R * psi
    rhs[:-1]   += off_R * psi[1:]
    rhs[1:]    += off_R * psi[:-1]
    return rhs


def solve_thomas(lower, main, upper, d):
    n = len(d)
    c_prime = np.zeros(n-1, dtype=complex)
    d_prime = np.zeros(n, dtype=complex)

    c_prime[0] = upper[0] / main[0]
    d_prime[0] = d[0] / main[0]

    for i in range(1, n-1):
        m = main[i] - lower[i-1] * c_prime[i-1]
        c_prime[i] = upper[i] / m
        d_prime[i] = (d[i] - lower[i-1] * d_prime[i-1]) / m

    m = main[n-1] - lower[n-2] * c_prime[n-2]
    d_prime[n-1] = (d[n-1] - lower[n-2] * d_prime[n-2]) / m

    x = np.empty(n, dtype=complex)
    x[-1] = d_prime[-1]

    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]

    return x


def evolve_CN(psi0: np.ndarray, V: np.ndarray,
              dx: float, dt: float, nt: int, p: Params,
              save_every: int = 20):
    ab, diag_R, off_R = build_CN_matrices(V, dx, dt, p)
    upper = ab[0, 1:]
    main = ab[1, :]
    lower = ab[2, :-1]

    psi = psi0.copy()
    snapshots = []
    times_all = []

    for j in range(nt):
        rhs = apply_R(psi, diag_R, off_R)
        rhs[0] = rhs[-1] = 0.0
        psi = solve_thomas(lower, main, upper, rhs)
        psi[0] = psi[-1] = 0.0

        if j % save_every == 0:
            snapshots.append(np.abs(psi)**2)
            times_all.append(j * dt)

    return np.array(snapshots), np.array(times_all)


# ===========================================================================
# 4. CALCULS DE TEMPS DE TRAVERSÉE (τ)
# ===========================================================================
def compute_tau_current(prob_snapshots, times_all, x, x_detect, dx, p):
    i_det = np.argmin(np.abs(x - x_detect))
    prob_right = np.array([np.sum(snap[i_det:]) * dx for snap in prob_snapshots])
    flux = np.gradient(prob_right, times_all)
    flux = np.maximum(flux, 0)
    if flux.sum() == 0:
        return np.nan
    tau = np.sum(times_all * flux) / np.sum(flux)
    return tau


# ===========================================================================
# 5. VISUALISATION
# ===========================================================================
def plot_snapshots(x, prob_list, time_list, V, p, title=""):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    V_scaled = V / V.max() * 0.3 if V.max() > 0 else V
    ax.fill_between(x, 0, V_scaled, color="lightcoral", alpha=0.5, label="Barrière $V_0$")
    
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(time_list)))
    
    for prob, t_snap, col in zip(prob_list, time_list, colors):
        # Courbe numérique (continue)
        ax.plot(x, prob, color=col, lw=1.5, label=f"Num t = {t_snap:.1f}")
        
        # Superposition analytique avec l'algorithme externe GausWP (uniquement si V0=0)
        if p.V0 == 0.0:
            psi_exact = GausWP(k0=p.k0, a=p.sigma, x=x - p.x0, t=t_snap, hbar=p.hbar, m=p.m)
            ax.plot(x, np.abs(psi_exact)**2, color=col, linestyle="--", lw=1.2)

    ax.set_xlabel("x (u.a.)")
    ax.set_ylabel(r"$|\Psi(x,t)|^2$")
    ax.set_title(title or f"Propagation du paquet (V0={p.V0}, a={p.a})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_tau_vs_a(a_vals, tau0_vals, taut_vals, p, label=""):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(a_vals, tau0_vals, "o-", label=r"$\tau_{0,\mathrm{num}}$ (libre)")
    ax.plot(a_vals, taut_vals, "s--", label=r"$\tau_{t,\mathrm{num}}$ (tunnel)")
    tau0_th = p.m * np.array(a_vals) / (p.hbar * p.k0)
    ax.plot(a_vals, tau0_th, "k:", lw=1.5, label=r"$\tau_{0,\mathrm{th}} = ma/\hbar k_0$")
    ax.set_xlabel("a (u.a.)")
    ax.set_ylabel("τ (u.a.)")
    ax.set_title(label or "Temps de traversée vs longueur de barrière")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_tau_vs_V0(V0_vals, taut_vals):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(V0_vals, taut_vals, "D-", color="purple")
    ax.set_xlabel(r"$V_0$ (u.a.)")
    ax.set_ylabel(r"$\tau_{t,\mathrm{num}}$ (u.a.)")
    ax.set_title(r"Temps tunnel $\tau_t$ en fonction de $V_0$")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


# ===========================================================================
# 6. SIMULATION UNIQUE
# ===========================================================================
def run_simulation(p: Params, save_every: int = 10):
    x, t_arr, dx, dt = make_grids(p)
    V   = make_potential(x, p)
    
    # INITIALISATION EXCLUSIVE VIA VOTRE MODULE EXTERNE GausWP à t = 0.0
    # On applique le décalage spatial (x - p.x0) pour centrer correctement le paquet
    psi0 = GausWP(k0=p.k0, a=p.sigma, x=x - p.x0, t=0.0, hbar=p.hbar, m=p.m)

    print(f"  dx={dx:.4f}  dt={dt:.6f}  nx={p.nx}  nt={p.nt}")
    print(f"  Norme initiale = {np.sum(np.abs(psi0)**2)*dx:.6f}")

    prob_snaps, times_all = evolve_CN(psi0, V, dx, dt, p.nt, p, save_every=save_every)
    print(f"  Norme finale   = {np.sum(prob_snaps[-1])*dx:.6f}")
    return x, t_arr, dx, V, prob_snaps, times_all


# ===========================================================================
# 7. MAIN
# ===========================================================================
def main():
    import os
    os.makedirs("resultats", exist_ok=True)

    # ------------------------------------------------------------------ 1a/1b
    print("=== 1a-b : Particule LIBRE (V0=0) ===")
    p_free = Params(V0=0.0, a=4.0)
    x, _, dx, V, snaps, times = run_simulation(p_free, save_every=20)

    x_detect = p_free.x0 + p_free.a
    tau0_num = compute_tau_current(snaps, times, x, x_detect, dx, p_free)
    tau0_th  = p_free.m * p_free.a / (p_free.hbar * p_free.k0)
    print(f"  τ₀,num = {tau0_num:.4f}   τ₀,th = {tau0_th:.4f}  "
          f"(erreur relative = {abs(tau0_num-tau0_th)/tau0_th*100:.2f}%)\n")

    idx_show = np.linspace(0, len(times)-1, 6, dtype=int)
    fig = plot_snapshots(x, snaps[idx_show], times[idx_show], V, p_free,
                          title="Particule libre — Continu: Numérique | Pointillés: GausWP Externe")
    fig.savefig("resultats/libre.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ 1c
    print("=== 1c : Effet TUNNEL (V0>0) ===")
    p_tun = Params(V0=5.0, a=3.0)
    x, _, dx, V, snaps, times = run_simulation(p_tun, save_every=20)

    x_detect = p_tun.x_bar + p_tun.a + 1.0
    taut_num = compute_tau_current(snaps, times, x, x_detect, dx, p_tun)
    tau0_ref = p_tun.m * p_tun.a / (p_tun.hbar * p_tun.k0)
    print(f"  τₜ,num = {taut_num:.4f}   τ₀,ref = {tau0_ref:.4f}\n")

    idx_show = np.linspace(0, len(times)-1, 6, dtype=int)
    fig = plot_snapshots(x, snaps[idx_show], times[idx_show], V, p_tun,
                          title=f"Effet tunnel — $V_0={p_tun.V0}$, $a={p_tun.a}$")
    fig.savefig("resultats/tunnel.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ 1d
    print("=== 1d : Influence de a ===")
    a_vals   = [1.0, 2.0, 3.0, 4.0, 5.0]
    tau0_arr = []
    taut_arr = []

    for a_val in a_vals:
        pf = Params(V0=0.0, a=a_val, nt=6000)
        xf, _, dxf, Vf, sf, tf = run_simulation(pf, save_every=15)
        t0 = compute_tau_current(sf, tf, xf, pf.x0 + a_val, dxf, pf)
        tau0_arr.append(t0)

        pt = Params(V0=5.0, a=a_val, nt=6000)
        xt, _, dxt, Vt, st, tt = run_simulation(pt, save_every=15)
        tt_val = compute_tau_current(st, tt, xt, pt.x_bar + a_val + 1.0, dxt, pt)
        taut_arr.append(tt_val)
        print(f"  a={a_val:.1f}  High τ₀={t0:.4f}  τₜ={tt_val:.4f}")

    fig = plot_tau_vs_a(a_vals, tau0_arr, taut_arr, Params(), "Influence de a")
    fig.savefig("resultats/tau_vs_a.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ 1e
    print("\n=== 1e : Influence de V0 ===")
    V0_vals  = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
    taut_V0  = []

    for V0_val in V0_vals:
        pv = Params(V0=V0_val, a=3.0, nt=6000)
        xv, _, dxv, Vv, sv, tv = run_simulation(pv, save_every=15)
        tt_val = compute_tau_current(sv, tv, xv, pv.x_bar + pv.a + 1.0, dxv, pv)
        taut_V0.append(tt_val)
        print(f"  V0={V0_val:.1f}  High τₜ={tt_val:.4f}")

    fig = plot_tau_vs_V0(V0_vals, taut_V0)
    fig.savefig("resultats/tau_vs_V0.png", dpi=150)
    plt.close(fig)

    print("\n✓ Tous les graphiques sauvegardés dans resultats/")


if __name__ == "__main__":
    main()