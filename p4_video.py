####------Projet-partie-4-----###
#Équation de Schrödinger 1D + barrière + vidéo (Crank-Nicolson)
#auteur : Alexis DUMAIRE - Rémi SATHTHIRYAN - Alexandre BERNARD
#contributeur Claude de Anthropic AI pour consolider le code et ameliorer la representation graphique (21/06/2026 version non payante)
#date creation : 18 juin 2026
#version 1 : 21 juin 2026
###

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass, replace

from PaquetOndeGauss_3G import GausWP

@dataclass
class Params:
    """
        Classe pour poser les bases et pour facilité la modifications des données si nécessaire
    """
    hbar: float = 1.0
    m: float = 1.0
    k0: float = 3.0
    sigma: float = 1.5

    V0: float = 1.0 
    a: float = 1.0
    x_bar: float = 0.0

    x_min: float = -80.0
    x_max: float = 80.0
    nx: int = 2500
    t_max: float = 20.0
    nt: int = 5000

    x0: float = -20.0

def make_grids(p:Params) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
        Génère les grilles spatiale et temporelle de la simulation.
    """
    x = np.linspace(p.x_min, p.x_max, p.nx)
    t = np.linspace(0, p.t_max, p.nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    assert dx < 1.0, "dx trop grand, instable"
    assert dt < 0.01, "dt trop grand, instable"
    return x, t, dx, dt

def make_potential(x:np.ndarray, p:Params) -> tuple[np.ndarray]:
    """
        Construit un potentiel rectangulaire (barrière ou puits).
        Le potentiel vaut V0 dans l'intervalle [x_bar, x_bar + a],
        et 0 ailleurs.
    """
    V = np.zeros_like(x)
    mask = (x >= p.x_bar) & (x <= p.x_bar + p.a)
    V[mask] = p.V0
    return V

def build_CN_matrices(V, dx, dt, p) -> tuple[np.ndarray,np.ndarray]:
    """
        Construit les matrices du schéma de Crank-Nicolson
        pour l'équation de Schrödinger 1D.

        Discrétisation implicite du type :
            (L) ψ^{n+1} = (R) ψ^n
    """
    n = len(V)
    r = 1j * p.hbar * dt / (4 * p.m * dx**2)

    diag_L = 1 + 2*r + 1j * dt * V / (2 * p.hbar)
    off_L = -r * np.ones(n - 1)

    ab = np.zeros((3, n), dtype=complex)
    ab[0, 1:] = off_L
    ab[1, :] = diag_L
    ab[2, :-1] = off_L

    diag_R = 1 - 2*r - 1j * dt * V / (2 * p.hbar)
    off_R = r * np.ones(n - 1)

    return ab, diag_R, off_R

def apply_R(psi, diag_R, off_R)-> tuple[np.ndarray]:
    """
        Applique l'opérateur de droite R à ψ^n.

        Calcule le membre de droite du schéma Crank-Nicolson :
            rhs = R ψ
    """
    rhs = diag_R * psi
    rhs[:-1] += off_R * psi[1:]
    rhs[1:] += off_R * psi[:-1]
    return rhs

def solve_thomas(lower, main, upper, d)-> tuple[np.ndarray,np.ndarray]:
    """
        Résout un système tridiagonal par l'algorithme de Thomas.
        Résout Ax = d où A est tridiagonale.
    """
    n = len(d)
    c = np.zeros(n-1, dtype=complex)
    d_ = np.zeros(n, dtype=complex)

    c[0] = upper[0] / main[0]
    d_[0] = d[0] / main[0]
    for i in range(1, n):
        denom = main[i] - lower[i-1] * c[i-1]
        if i < n - 1:
            c[i] = upper[i] / denom
        d_[i] = (d[i] - lower[i-1] * d_[i-1]) / denom
    x = np.zeros(n, dtype=complex)
    x[-1] = d_[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_[i] - c[i] * x[i+1]
    return x

def evolve_CN(psi0, V, dx, dt, nt, p, save_every=10):
    """
            Fait évoluer la fonction d'onde dans le temps avec le schéma de Crank-Nicolson.
        Chaque pas de itération :
        1. Calcul du second membre
        2. Application des conditions aux bords
        3. Résolution du système tridiagonal
        4. Stockage périodique des densités de probabilité
    """
    ab, diag_R, off_R = build_CN_matrices(V, dx, dt, p)
    upper = ab[0, 1:]
    main = ab[1, :]
    lower = ab[2, :-1]
    psi = psi0.copy()
    snaps = []
    times = []
    for i in range(nt):
        rhs = apply_R(psi, diag_R, off_R)
        rhs[0] = rhs[-1] = 0
        psi = solve_thomas(lower, main, upper, rhs)
        psi[0] = psi[-1] = 0
        norm = np.sum(np.abs(psi)**2) * dx
        if i % 100 == 0:
            print(f"t = {i*dt:.3f} | norm = {norm:.6f}")
        if i % save_every == 0:
            snaps.append(np.abs(psi)**2)
            times.append(i * dt)
    return np.array(snaps), np.array(times)

def run_simulation(p):
    """
        Initialise la simulation complète :
        - grille
        - potentiel
        - état initial (paquet d'onde gaussien)
        - évolution temporelle
    """
    x, t, dx, dt = make_grids(p)
    V = make_potential(x, p)
    psi0 = GausWP(
        k0=p.k0,
        a=p.sigma,
        x=x - p.x0,
        t=0.0,
        hbar=p.hbar,
        m=p.m
    )
    print("Norme initiale:", np.sum(np.abs(psi0)**2) * dx)
    snaps, times = evolve_CN(psi0, V, dx, dt, p.nt, p, save_every=10)
    return x, V, snaps, times

def make_video(x, snaps, V, times, p, filename="tunnel.gif"):
    """
        Génère une animation de l'évolution de |ψ(x,t)|²
        avec affichage de la barrière de potentiel.
        Sauvegarde un fichier GIF.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    # Barrière de potentiel
    V_plot = np.zeros_like(V)
    if np.max(V) > 0:
        V_plot = V / np.max(V) * (0.3 * np.max(snaps))  # mise à l'échelle propre
    ax.fill_between(
        x,
        0,
        V_plot,
        color="lightcoral",
        alpha=0.5,
        label=r"Barrière $V_0$"
    )
    # Courbe initiale
    line, = ax.plot(
        x,
        snaps[0],
        lw=2,
        color="#3b2c7a",
        label=r"Num $t = 0.0$"
    )
    # Texte temps (propre)
    time_text = ax.text(
        0.02, 0.95,
        "",
        transform=ax.transAxes
    )
    # Axes style
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0.0,0.8)

    ax.set_xlabel("x (u.a.)")
    ax.set_ylabel(r"$|\Psi(x,t)|^2$")
    ax.set_title("État initial uniquement (t=0)")

    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    # Animation
    def update(i):
        line.set_data(x, snaps[i])
        time_text.set_text(f"t = {times[i]:.2f}")
        return line, time_text
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(snaps),
        interval=30,
        blit=False
    )
    ani.save(filename, writer="pillow", fps=30)
    plt.close()

def compute_arrival_time(prob_snapshots, times, x, x_detect):
    """
    Calcule un temps moyen d'arrivée à un détecteur.
    On calcule la probabilité située à droite du détecteur,
    puis sa dérivée temporelle. Cette dérivée joue le rôle
    d'un flux de probabilité.
    """
    dx = x[1] - x[0]
    i_detect = np.argmin(np.abs(x - x_detect))
    # Probabilité située à droite du détecteur
    prob_right = np.sum(prob_snapshots[:, i_detect:], axis=1) * dx
    # Variation de cette probabilité dans le temps
    flux = np.gradient(prob_right, times)
    # On conserve seulement le flux vers la droite
    flux_positive = np.maximum(flux, 0.0)
    flux_total = np.trapz(flux_positive, times)
    if flux_total < 1e-10:
        return np.nan
    tau = np.trapz(times * flux_positive, times) / flux_total
    return tau


def transmission_finale(prob_snapshots, x, p):
    """
    Estime la probabilité transmise à la fin de la simulation.
    """
    dx = x[1] - x[0]
    mask = x > (p.x_bar + p.a)
    return np.sum(prob_snapshots[-1, mask]) * dx


def make_free_reference(p, save_every=5, detector_offset=1.0):
    """
    Calcule les temps dans le cas libre V0 = 0.
    Cette référence est utilisée pour tau0,num et tau_t,num.
    """
    p_free = replace(p, V0=0.0)
    x, V, snaps, times = run_simulation(p_free)
    # Détecteurs exactement séparés par la distance a
    t_x0 = compute_arrival_time(snaps, times, x, p_free.x_bar)
    t_xa = compute_arrival_time(
        snaps,
        times,
        x,
        p_free.x_bar + p_free.a
    )
    tau0_num = t_xa - t_x0
    # Détecteurs légèrement avant et après la barrière
    x_left = p_free.x_bar - detector_offset
    x_right = p_free.x_bar + p_free.a + detector_offset
    t_left_free = compute_arrival_time(snaps, times, x, x_left)
    t_right_free = compute_arrival_time(snaps, times, x, x_right)
    tau_path_free = t_right_free - t_left_free
    return {
        "tau0_num": tau0_num,
        "tau_path_free": tau_path_free
    }


def measure_tunnel_time(p, free_reference=None,
                        save_every=5, detector_offset=1.0):
    """
    Mesure tau_t,num avec deux détecteurs.

    On compare le temps avec barrière au temps libre afin
    d'isoler l'effet de la barrière.
    """
    if free_reference is None:
        free_reference = make_free_reference(p,detector_offset=detector_offset) 
    x, V, snaps, times = run_simulation(p)
    x_left = p.x_bar - detector_offset
    x_right = p.x_bar + p.a + detector_offset
    t_left = compute_arrival_time(snaps, times, x, x_left)
    t_right = compute_arrival_time(snaps, times, x, x_right)
    tau_path_barrier = t_right - t_left
    tau0_num = free_reference["tau0_num"]
    tau_path_free = free_reference["tau_path_free"]
    # Temps attribué à la traversée de la barrière
    tau_t_num = tau0_num + (tau_path_barrier - tau_path_free)
    T_num = transmission_finale(snaps, x, p)

    return {
        "tau_t_num": tau_t_num,
        "tau0_num": tau0_num,
        "tau_path_barrier": tau_path_barrier,
        "tau_path_free": tau_path_free,
        "T_num": T_num
    }


def plot_tau_vs_a(a_vals, tau0_vals, taut_vals, p):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(a_vals, tau0_vals, "o-", label="tau0,num")
    ax.plot(a_vals, taut_vals, "s--", label="tau_t,num")
    tau0_th = p.m * np.array(a_vals) / (p.hbar * p.k0)
    ax.plot(
        a_vals,
        tau0_th,
        ":",
        label="tau0,th"
    )
    ax.set_xlabel("a")
    ax.set_ylabel("Temps")
    ax.set_title("Influence de la largeur a")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_tau_vs_V0(V0_vals, taut_vals):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(V0_vals, taut_vals, "D-")
    ax.set_xlabel("V0")
    ax.set_ylabel("tau_t,num")
    ax.set_title("Influence de V0 sur le temps de traversée")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_transmission_vs_V0(V0_vals, T_vals):
    fig, ax = plt.subplots(figsize=(7, 4))
    T_vals = np.maximum(np.array(T_vals), 1e-16)
    ax.semilogy(V0_vals, T_vals, "o-")
    ax.set_xlabel("V0")
    ax.set_ylabel("T_num")
    ax.set_title("Transmission en fonction de V0")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return fig

def main():
    import os
    os.makedirs("resultats", exist_ok=True)
    print("\n=== 1.a : Propagation du paquet et effet tunnel ===")
    p_demo = Params(V0=6.0, a=2.0)
    x, V, snaps, times = run_simulation(p_demo)
    make_video(
        x,
        snaps,
        V,
        times,
        p_demo,
        "resultats/tunnel.gif"
    )
    print("GIF créé : resultats/tunnel.gif")
    print("\n=== 1.b : Temps libre tau0,num ===")
    p_free = Params(V0=0.0, a=4.0, nt=6000)
    free_ref = make_free_reference(p_free)
    tau0_num = free_ref["tau0_num"]
    tau0_th = (
        p_free.m * p_free.a
        / (p_free.hbar * p_free.k0)
    )
    print(f"tau0,num = {tau0_num:.4f}")
    print(f"tau0,th  = {tau0_th:.4f}")
    print("\n=== 1.c : Temps tunnel tau_t,num ===")
    p_tun = Params(V0=6.0, a=2.0, nt=6000)
    free_ref_tun = make_free_reference(
        p_tun,
        save_every=5
    )
    result_tun = measure_tunnel_time(
        p_tun,
        free_reference=free_ref_tun
    )
    print(f"tau0,num = {result_tun['tau0_num']:.4f}")
    print(f"tau_t,num = {result_tun['tau_t_num']:.4f}")
    print(f"T_num = {result_tun['T_num']:.8f}")
    print("\n=== 1.d : Influence de a ===")
    a_vals = [1.0, 1.5, 2.0, 2.5, 3.0]
    tau0_vals = []
    taut_vals = []
    for a_val in a_vals:
        p_a = Params(
            V0=6.0,
            a=a_val,
            nt=6000
        )
        reference_a = make_free_reference(
            p_a,
            save_every=5
        )
        result_a = measure_tunnel_time(
            p_a,
            free_reference=reference_a,
            save_every=5
        )
        tau0_vals.append(result_a["tau0_num"])
        taut_vals.append(result_a["tau_t_num"])
        print(
            f"a = {a_val:.1f} | "
            f"tau0,num = {result_a['tau0_num']:.4f} | "
            f"tau_t,num = {result_a['tau_t_num']:.4f}"
        )
    fig = plot_tau_vs_a(
        a_vals,
        tau0_vals,
        taut_vals,
        Params()
    )
    fig.savefig("resultats/tau_vs_a.png", dpi=150)
    plt.close(fig)
    print("\n=== 1.e : Influence de V0 ===")
    a_etude = 2.0
    p_reference = Params(
        V0=0.0,
        a=a_etude,
        nt=6000
    )
    E0 = (
        p_reference.hbar**2
        * p_reference.k0**2
        / (2 * p_reference.m)
    )
    print(f"E0 = {E0:.2f}")
    V0_vals = [
        1.0, 2.0, 3.0, 4.0,
        4.5, 5.0, 6.0, 7.0, 8.0
    ]
    reference_V0 = make_free_reference(
        p_reference,
        save_every=5
    )
    taut_V0 = []
    T_V0 = []
    for V0_val in V0_vals:
        p_V0 = Params(
            V0=V0_val,
            a=a_etude,
            nt=6000
        )
        result_V0 = measure_tunnel_time(
            p_V0,
            free_reference=reference_V0,
            save_every=5
        )
        taut_V0.append(result_V0["tau_t_num"])
        T_V0.append(result_V0["T_num"])
        print(
            f"V0 = {V0_val:.1f} | "
            f"V0/E0 = {V0_val / E0:.2f} | "
            f"tau_t,num = {result_V0['tau_t_num']:.4f} | "
            f"T_num = {result_V0['T_num']:.8f}"
        )
    fig = plot_tau_vs_V0(V0_vals, taut_V0)
    fig.savefig("resultats/tau_vs_V0.png", dpi=150)
    plt.close(fig)
    fig = plot_transmission_vs_V0(V0_vals, T_V0)
    fig.savefig(
        "resultats/transmission_vs_V0.png",
        dpi=150
    )
    plt.close(fig)
    print("\nTous les résultats sont dans le dossier resultats/")

if __name__ == "__main__":
    main()
