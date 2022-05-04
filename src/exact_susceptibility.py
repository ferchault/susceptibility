import numpy as np
import math
from scipy.special import expi
from scipy.integrate import nquad

def h_atom_groundstate(r, Z):
    return np.power(Z,3/2)/np.sqrt(np.pi) * np.exp(-Z*r)

def nonlocal_susceptibility(r, rp, Z):
    dist = math.dist( r, rp )

    u = Z*( np.linalg.norm(r) + np.linalg.norm(rp) + dist )
    v = Z*( np.linalg.norm(r) + np.linalg.norm(rp) - dist )

    prefactor = np.exp( 0.5*(u + v) )/np.pi
    first_term = 2*np.euler_gamma + 5/2 + 0.5*(u + v) + np.log( u * v ) + expi(v)
    second_term = np.exp( 0.5*(u - v) )/np.pi/(u - v)
    return h_atom_groundstate(np.linalg.norm(r), Z) *  h_atom_groundstate(np.linalg.norm(rp), Z) * (prefactor * first_term + second_term)

def integrate_me(x1, y1, z1, x2, y2, z2):
    r = np.asarray( (x1, y1, z1) )
    rp = np.asarray( (x2, y2, z2) )
    return nonlocal_susceptibility( r, rp, 1) 


#r = np.asarray( (1, 0, 0) )
#rp = np.asarray( (0, 1, 0) )
#Z = 1

#print( nonlocal_susceptibility(r, rp, Z) )
print( nquad(integrate_me, [[-10, 10],[-10, 10], [-10, 10], [-10, 10], [-10, 10], [-10, 10] ]) )
