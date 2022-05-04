import numpy as np
import math
from scipy.special import expi
from scipy.integrate import nquad, quad

def h_atom_groundstate(r, Z):
    return np.power(Z,3/2)/np.sqrt(np.pi) * np.exp(-Z*r)

def nonlocal_susceptibility(r, rp, Z):
    dist = math.dist( r, rp )
    # possible typo in the paper? if I do this, then all the formulas agree, but it's weird...
    u = Z*( np.linalg.norm(r) + np.linalg.norm(rp) + dist )/0.5
    v = Z*( np.linalg.norm(r) + np.linalg.norm(rp) - dist )/0.5

    prefactor = np.exp( -0.5*(u + v) )/np.pi 
    first_term = 2*np.euler_gamma - 5/2 + 0.5*(u + v) + np.log( u * v ) - expi(v)
    second_term = np.exp( -0.5*(u - v) )/(np.pi*(u - v))
    return h_atom_groundstate(np.linalg.norm(r), Z) *  h_atom_groundstate(np.linalg.norm(rp), Z) * (prefactor * first_term - second_term)

def integrate_me(x1, y1, z1, x2, y2, z2):
    r = np.asarray( (x1, y1, z1) )
    rp = np.asarray( (x2, y2, z2) )
    return nonlocal_susceptibility( r, rp, 1) 

def white_formula(r, rp, Z):
    r1 = np.linalg.norm(r)
    r2 = np.linalg.norm(rp)
    r12 = math.dist(r,rp)

    first_term = -np.exp(-Z * r12)/(r12 * 4* np.pi) 

    second_term_prefactor = Z * np.exp(- Z * (r1 + r2) ) / (2* np.pi)
    f = lambda x: (np.exp(x) - 1)/x
    integral, err = quad(f, 0, Z*(r1 + r2 - r12))
    second_term = np.euler_gamma - 5/2 + Z*(r1+r2) + np.log( Z*(r1 + r2 +r12 )) - integral


    return h_atom_groundstate(r1, Z) *  h_atom_groundstate(r2, Z) *(first_term + second_term_prefactor * second_term)

def hostler_formula(r, rp, Z):
    dist = math.dist( r, rp )
    u = Z*( np.linalg.norm(r) + np.linalg.norm(rp) + dist )
    v = Z*( np.linalg.norm(r) + np.linalg.norm(rp) - dist )

    f = lambda x: (np.exp(x) - 1)/x
    integral, err = quad(f, 0, v)

    first_term = - np.exp( - Z * dist)/(4*np.pi*dist)
    second_term_prefactor = np.exp(-0.5*(u+v))/(2*np.pi*Z)
    second_term = -integral + np.log(u) + 0.5*(u+v)-5/2+np.euler_gamma


    return h_atom_groundstate(np.linalg.norm(r), Z) *  h_atom_groundstate(np.linalg.norm(rp), Z) *(first_term + second_term_prefactor * second_term)
    



r = np.asarray( (1, 0, 0) )
rp = np.asarray( (1.0001, 0, 0) )
Z = 1

print( nonlocal_susceptibility(r, rp, Z) )
print( white_formula(r, rp, Z) )
print( hostler_formula(r, rp, Z) )
#print( nquad(integrate_me, [[-10, 10],[-10, 10], [-10, 10], [-10, 10], [-10, 10], [-10, 10] ]) )
