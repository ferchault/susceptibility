import numpy as np

# Approximate non-local polarizability for hydrogen atom
# 2 coordinate function, assuming \theta=\phi=\theta'=\phi'=0

def hydrogen_polarizability(x,y): # second order approximation
    nlalpha = 2654208.0 *  x**3 * y**3 + 18432.0 * np.exp((x + y)/6) * (-6 + x) * x**3 * (-6 + y) *  y**3
    nlalpha += 8192 * (32 + 6 * x * (8 + 3 * x * (2 + x))) * (32 + 6 * y * (8 + 3 * y * (2 + y)))
    nlalpha += 27 * np.exp((x +y)/6) *  (243 + 4 * x * (81 + 2 * x * (27 - 4 * (-3 + x) * x))) *  (243 + 4 * y * (81 + 2 * y * (27 - 4 * (-3 + y) * y)))
    nlalpha += np.exp((x + y)/6) * (729 + 12 * x * (81 + 2 * x * (27 + 4 * x * (3 + x)))) * (729 + 12 * y * (81 + 2 * y * (27 + 4 * y * (3 + y))))
    nlalpha = nlalpha * np.exp(-3.0*(x+y)/2.0)/(71663616.0 * np.pi**2 * x**2 * y**2 )
    return nlalpha

a = hydrogen_polarizability(2,3)
print(a)
