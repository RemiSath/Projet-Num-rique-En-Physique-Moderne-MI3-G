import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
# ------------------ Def ------------------
H_BARRE : float = 1 #= 6.626e-34
M: float = 1 #= 9.109e-31
k0 = 3.0       # nombre d'onde central
a = -15.0        # largeur initiale
x_0 = 0.0      # position initiale du centre
V0 = 0.0

npts = 1000
x = np.linspace(-20, 20, npts)
dx = x[1] - x[0]

t = 0.0
# ------------------ Algorithme GausWP ------------------
def GausWP(k0 : float, 
           a : float, 
           x : np.ndarray, 
           t : float,
           hbar=H_BARRE,
           m=M):
    """
    Paquet d'ondes gaussien libre.
    Paramètres
    ----------
    k0 : float
        Nombre d'onde moyen.
    a : float
        Largeur initiale.
    x : float ou ndarray
        Position.
    t : float
        Temps.

    Retour
    ------
    complex
        Valeur de Ψ(x,t).
    """

    omega = a * np.sqrt(1 + (2*hbar*t/(m*a**2))**2)
    vg = hbar * k0 / m

    amplitude = (2/(np.pi*omega**2))**0.25

    # phase simplifiée
    phase = np.exp(1j*(k0*x - hbar*k0**2*t/(2*m)))

    return amplitude * np.exp(-(x-vg*t)**2/omega**2) * phase

if __name__ == "__main__":
    Psi = GausWP(k0, a, x, t, x_0)
    plt.figure(figsize=(8, 5))
    plt.plot(x, Psi.real, label=r"$\Re(\Psi(x,0))$")
    plt.plot(x, Psi.imag, label=r"$\Im(\Psi(x,0))$")
    plt.xlabel("x")
    plt.ylabel(r"$\Psi(x,0)$")
    plt.title("Paquet d'ondes gaussien a t = 0")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("resultats/PaquetOndeGauss1_3G.png", dpi=150)

    print("Valeur de Psi(0,0) =", Psi[npts // 2])
    print("Norme de Psi (verification) =", np.sum(np.abs(Psi)**2) * dx)

