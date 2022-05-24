from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import assoc_laguerre, sph_harm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.interpolate
plt.style.use('seaborn-whitegrid')

def h_atom_density(r,theta,phi,n1,l1,m1,n2,l2,m2):
    return h_atom_wfn(r,theta,phi,n1,l1,m1)*h_atom_wfn(r,theta,phi,n2,l2,m2)

def h_atom_wfn(r,theta,phi,n,l,m):
    rho=2*r/n
    prefactor=np.sqrt( (2/n)**3 * np.math.factorial(n-l-1)/(2*n*np.math.factorial(n+l) ) )
    radial=np.exp(-rho/2) *(rho**l)* assoc_laguerre(rho,n-l-1,2*l+1)
    angular=sph_harm(m,l, phi, theta) #Note: convention on phi and theta is opposite of Mathematica/textbook!!!
    return prefactor*radial*angular

def h_atom_energy(n):
    return -1/(2*n**2)
  
  ## WARNING!!!
# Individual P(r) functions are imaginary, but when we build alpha, the imaginary part cancels.
# I only return the real part to save computational time
# It is not the true P(r) that's returned
# You have been WARNED

def integrand(par,r,theta,phi,quantum_numbers_init,quantum_numbers_final):
    n1, l1, m1 = quantum_numbers_init #unpacking
    n2, l2, m2 = quantum_numbers_final
    return np.real(h_atom_density(r/par,theta,phi,n1,l1,m1,n2,l2,m2)/(par**4.0))

def transient_polarization(r, theta, phi,quantum_numbers_init, quantum_numbers_final ):
    partial_int = lambda r, theta, phi: quad(integrand, 0,1, args=(r,theta,phi,quantum_numbers_init,quantum_numbers_final))
    return r * partial_int(r, theta, phi)[0]
  
  def cart2sph(cartesians):
    x, y, z = cartesians
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r #TODO verify consistency of angles

def nlalpha(coord1, coord2, max_principal):
    phi1, theta1, r1 = cart2sph(coord1)
    phi2, theta2, r2 = cart2sph(coord2)
    alpha=0
    for n in range(max_principal+1):
        if n>1:
            for i in range(n):
                m_min = -i
                m_max = i+1 # dumb dumb Python numbering
                for j in range(m_min, m_max):
                    quantum_numbers= (n, i ,j)
                    alpha+= ( transient_polarization( r1, theta1, phi1, (1,0,0), quantum_numbers  ) * 
                    transient_polarization(r2, theta2, phi2, quantum_numbers, (1,0,0)) ) / (h_atom_energy(n)-0.5)
                    #print("principal: ",n," angular: ",i," magnetic: ",j)
    return alpha
  
  axes_range_x = np.linspace(-3, 3, num=50)
axes_range_y = np.linspace(-3, 3, num=50)

alphas = np.zeros((len(axes_range_x), len(axes_range_y)))

for xcoord in range(len(axes_range_x)):
    for ycoord in range(len(axes_range_y)):
        alphas[xcoord, ycoord] = nlalpha((1,1,axes_range_x[xcoord]),(1,1,axes_range_y[ycoord]),3)    

X,Y = np.meshgrid(axes_range_x, axes_range_y)
        
fig = plt.figure()
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#fig, ax = plt.subplots()
#CS = ax.contour(X, Y, alphas)
#surf = ax.plot_surface(X, Y, alphas, cmap=cm.coolwarm,
 #                      linewidth=0, antialiased=False)
  
print(nlalpha((1,1,1),(1,1,1),3))
