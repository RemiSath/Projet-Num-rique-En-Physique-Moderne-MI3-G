import numpy as np
from numpy import pi, linspace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PaquetOndeGauss_3G import GausWP, H_BARRE, M

# ------------------ Def ------------------
nx = 400
nt = 2000
x_min, x_max = 0, 10
t_min, t_max = 0.0, 2.0

x = np.linspace(x_min, x_max, nx)
t = np.linspace(t_min, t_max, nt)

dx = x[1] - x[0]
dt = t[1] - t[0]

# Parametres du paquet d'ondes initial (definis ICI, pas importes)
k0 = 5.0
a = 0.5
x_0 = 0
V0 = 0.0


# ------------------ Algorithme de derivation ------------------
def derives_npt(f, dx):
    npts = len(f)
    df = np.zeros(npts, dtype=f.dtype)
    for i in range(1, npts - 1):
        df[i] = (f[i + 1] - f[i - 1]) / (2 * dx)
    df[0] = (f[1] - f[0]) / dx
    df[-1] = (f[-1] - f[-2]) / dx
    return df


def derivee_seconde(f, dx):
    npts = len(f)
    d2f = np.zeros(npts, dtype=f.dtype)
    for i in range(1, npts - 1):
        d2f[i] = (f[i + 1] - 2 * f[i] + f[i - 1]) / dx**2
    d2f[0] = d2f[1]
    d2f[-1] = d2f[-2]
    return d2f


# ------------------ Algorithme pour l'equation de Schrodinger ------------------
def Ond_2d(nx_: int, nt_: int, x_, k0_, a_, x_0_) -> np.ndarray:
    """Tableau 2D (nx, nt) : colonne 0 = paquet gaussien initial, reste = 0."""
    tab = np.empty((nx_, nt_), dtype=complex)
    tab[:, 0] = GausWP(k0_, a_, x_, 0.0, x_0_)   # t=0 scalaire, pas le tableau t
    tab[:, 1:] = 0.0
    return tab


def algo_etude_t(psi, x, t, V0=0.0, H_BARRE=H_BARRE, m=M):
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    nt = len(t)
    for j in range(nt - 1):
        d2_psi = derivee_seconde(psi[:, j], dx)
        psi[:, j + 1] = psi[:, j] + dt * (
            1j * H_BARRE / (2 * m) * d2_psi
            - 1j * V0 / H_BARRE * psi[:, j]
        )
        psi[0, j + 1] = 0.0
        psi[-1, j + 1] = 0.0
    return psi


# ------------------ Main ------------------
def main():
    psi = Ond_2d(nx, nt, x, k0, a, x_0)
    psi_num = algo_etude_t(psi, x, t, V0)

    print(f"dx = {dx:.4f}, dt = {dt:.6f}")
    print(f"Critere de stabilite (dt << m*dx^2/hbar) : "
          f"m*dx^2/hbar = {M*dx**2/H_BARRE:.4f}  (dt = {dt:.6f})")
    print()

    indices = [0, nt // 4, nt // 2, nt - 1]

    for j in indices:
        tj = t[j]

        col = psi_num[:, j]
        norme_num = np.sum(np.abs(col)**2) * dx
        x_moy_num = (np.sum(x * np.abs(col)**2) * dx / norme_num
                      if norme_num != 0 else np.nan)

        psi_th = GausWP(k0, a, x, tj, x_0)
        norme_th = np.sum(np.abs(psi_th)**2) * dx
        x_moy_th = np.sum(x * np.abs(psi_th)**2) * dx / norme_th

        print(f"t = {tj:.4f}")
        print(f"  Numerique  : norme = {norme_num:10.4f}, <x> = {x_moy_num:8.4f}")
        print(f"  Theorique  : norme = {norme_th:10.4f}, <x> = {x_moy_th:8.4f}")
        print(f"  v_g attendu (x0 + vg*t) = {x_0 + (H_BARRE*k0/M)*tj:.4f}")
        print()

    plt.figure()
    plt.plot(x, np.abs(psi_num[:, 0])**2, label="t=0")
    plt.plot(x, np.abs(psi_num[:, -1])**2, label="t=tf")
    plt.xlabel("x")
    plt.ylabel(r"$|\Psi(x,t)|^2$")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("resultats/PaquetOndeGauss1_3G.png", dpi=150)


if __name__ == "__main__":
    main()