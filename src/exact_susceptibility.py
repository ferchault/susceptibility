import numpy as np
import math
from scipy.special import expi

def h_atom_groundstate(r, Z):
    return np.power(Z,3/2)/np.sqrt(np.pi) * np.exp(-Z*r)

r = np.asarray( (1, 0, 0) )
rp = np.asarray( (0, 1, 0) )
Z = 1

dist = math.dist( r, rp )

u = Z*( np.linalg.norm(r) + np.linalg.norm(rp) + dist )
v = Z*( np.linalg.norm(r) + np.linalg.norm(rp) - dist )

prefactor = np.exp( 0.5*(u + v) )/np.pi
first_term = 2*np.euler_gamma + 5/2 + 0.5*(u + v) + np.log( u * v ) + expi(v)
second_term = np.exp( 0.5*(u - v) )/np.pi/(u - v)
chi = h_atom_groundstate(np.linalg.norm(r), Z) *  h_atom_groundstate(np.linalg.norm(rp), Z) * (prefactor * first_term + second_term)

print( chi )
