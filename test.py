import numpy as np

def derives_npt(f,dx):
	"""
	Entrées : 
		f : tableau
		dx : distance entre les points
	Sortie :
		df : dérivée de f
	"""
	npts = len(f)
	df = np.zeros(npts)
	for i in range(1,npts-1):
		df[i] = (f[i+1] - f[i-1])/2*dx
	df[0] = (f[1] - f[0])/dx
	df[-1] = (f[-1] - f[-2])/dx
	return df
	
def carre(x):
	return x**2

def deux_x(x):
	return 2*x

def fonction_bis(x):
	return 1/(np.exp(-x)+1)

def main():
	x = np.linspace(0,10,11)
	f = fonction_bis(x)
	d_f = derives_npt(f,1)
	for i in range(len(d_f)):
		print(d_f[i])

main()