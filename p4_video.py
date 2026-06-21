import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass

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

def main():
    import os
    os.makedirs("resultats", exist_ok=True)

    V0 = float(input("Quelle valeur de V0 voulez-vous ?\n"))
    while(V0 < 0.0):
        V0 = input("V0 dois être supérieur ou égal à 0")
    print("V0 =", V0)

    p = Params(V0=V0)

    x, V, snaps, times = run_simulation(p)

    make_video(x, snaps, V, times, p, "resultats/tunnel.gif")

    print("✓ GIF généré : resultats/tunnel.gif")

if __name__ == "__main__":
    main()