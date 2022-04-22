#%%
import pyscf.scf
import pyscf.gto
import pyscf.dft
import scipy.integrate as sci
import scipy.interpolate as scii
import pyscf.qmmm
import tqdm
import numpy as np
import quadpy
from scipy.linalg import lstsq as sp_lstsq
import matplotlib.pyplot as plt


class ResponseCalculator:
    """Implements a generalized case of 10.1021/ct1004577, section 2."""

    def __init__(self, mol, calc, coords, centers, sigmas):
        self._mol = mol
        self._calc = calc
        # point charge
        self._q = 1e-1
        # softness parameter
        self._a = 1e-5
        calc = self._calc(self._mol)
        calc.kernel()
        self._center = calc.energy_elec()[0]
        self._centers = centers
        self._sigmas = sigmas

        self._build_cache(coords)
        self._build_numerical_integral()

    def _build_numerical_integral(self):
        def integrand(cylr, cylz, sigma, d, a):
            """2D version of integral"""
            p = np.sqrt(cylr**2 + cylz**2)
            q = np.sqrt(cylr**2 + (cylz - d) ** 2)
            return (
                2
                * np.pi
                * cylr
                * np.exp(-0.5 * (p / sigma) ** 2)
                / ((sigma * np.sqrt(2 * np.pi)))
                / (a + q)
            )

        self._integral = {}
        for sigma in tqdm.tqdm(set(self._sigmas)):
            yss = []
            xss = np.logspace(1e-5, 5, 100)
            for d in xss:
                # radius where the gaussian falls below 1e-10
                limit = np.sqrt(-2 * sigma**2 * np.log(1e-10))
                yss.append(
                    sci.nquad(
                        integrand,
                        ((0, limit), (-limit, limit)),
                        args=(sigma, d, self._a),
                        opts={
                            "limit": 1000,
                            "points": [
                                d,
                            ],
                        },
                    )[0]
                )
            self._integral[sigma] = scii.interp1d(
                xss, yss, "linear", fill_value=(yss[0], yss[-1]), bounds_error=False
            )

    def _build_cache(self, coords):
        try:
            cache = np.load("cache.npz")["Es"]
            if np.allclose(cache[:, :3], coords):
                self._cache = cache[:, 3]
                return
        except:
            pass

        Es = []
        for pos in tqdm.tqdm(coords):
            Es.append(list(pos) + [self.get_energy_derivative(pos)])
        Es = np.array(Es)
        np.savez("cache.npz", Es=Es)
        self._cache = Es[:, 3]

    def get_energy_derivative(self, pos):
        up = pyscf.qmmm.mm_charge(
            self._calc(self._mol), np.array((pos,)), np.array((self._q,))
        )
        up.conv_tol = 1e-13
        up.kernel()
        dn = pyscf.qmmm.mm_charge(
            self._calc(self._mol), np.array((pos,)), np.array((-self._q,))
        )
        dn.conv_tol = 1e-13
        dn.kernel()
        assert up.converged and dn.converged
        return up.energy_elec()[0] + dn.energy_elec()[0] - 2 * self._center


# def do_case():
mol = pyscf.gto.M(
    atom=f"He 0 0 0",
    basis="aug-cc-pVDZ",
    verbose=0,
)

scheme = quadpy.u3.schemes["lebedev_053"]()
radial = np.linspace(0.3, 3, 4)
coords = np.concatenate([scheme.points.T * _ for _ in radial])
np.random.seed(42)
coords += np.random.random(len(coords) * 3).reshape(-1, 3) * 0.1

# collect data
nsigmas = 20
minsigma = -10
sigmas = 1.2 ** np.arange(minsigma, minsigma + nsigmas)
centers = np.zeros((nsigmas, 3))
rc = ResponseCalculator(mol, pyscf.scf.RHF, coords, centers, sigmas)

A = []
for pos in tqdm.tqdm(coords[: nsigmas**2]):
    line = np.zeros((nsigmas, nsigmas))
    distances = np.linalg.norm(pos - centers, axis=1)
    for i in range(nsigmas):
        g_i = rc._integral[sigmas[i]](distances[i])
        for j in range(nsigmas):
            line[i, j] = g_i * rc._integral[sigmas[j]](distances[j])
    A.append(line.reshape(-1))
#     plt.plot(line.reshape(-1))
#     if len(A) == 1:
#         break
# return

A = np.array(A)
plt.imshow(A)
plt.show()
y = rc._cache / rc._q**2
y = y[: nsigmas**2]
res = sp_lstsq(A, y, lapack_driver="gelsy")[0]
residuals = y - A @ res
print("Relative residual [%]", abs(residuals).mean() / abs(y).mean() * 100)

res = res.reshape(nsigmas, nsigmas)
plt.imshow(res)


# do_case()

# %%
plt.plot(sorted(residuals))
plt.plot(sorted(y))
# %%