import pyscf.scf
import pyscf.gto
import pyscf.dft
import scipy.integrate as sci
import pyscf.qmmm
import tqdm
import numpy as np
import quadpy
from scipy.linalg import lstsq as sp_lstsq


class ResponseCalculator:
    """Implements a generalized case of 10.1021/ct1004577, section 2."""

    def __init__(self, mol, calc, coords, b):
        self._mol = mol
        self._calc = calc
        # point charge
        self._q = 1e-1
        # softness parameter
        self._a = 1e-5
        # gaussian width parameter
        self._b = b
        calc = self._calc(self._mol)
        calc.kernel()
        self._center = calc.energy_elec()[0]

        self._build_cache(coords)
        self._build_numerical_integral()
        self._build_grid()

    def _build_numerical_integral(self):
        xss = np.logspace(1e-4, 8, 100)
        yss = []
        for x1 in xss:
            yss.append(
                sci.quad(
                    lambda xs: np.exp(-self._b * xs**2) / (self._a + np.abs(xs - x1)),
                    -100,
                    100,
                    limit=1000,
                    points=x1,
                )[0]
            )
        self._integral = lambda x: np.interp(x, xss, yss, right=np.nan)

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

    def _build_grid(self):
        # xs = np.linspace(-3, 3, 5)
        # self._centers = np.array(np.meshgrid(*[xs] * 6)).reshape(6, -1).T
        self._centers = np.zeros((70, 6))


def do_case(b):
    mol = pyscf.gto.M(
        atom=f"He 0 0 0",
        basis="aug-cc-pVDZ",
        verbose=0,
    )

    scheme = quadpy.u3.schemes["lebedev_053"]()
    radial = np.linspace(0.3, 3, 10)
    coords = np.concatenate([scheme.points.T * _ for _ in radial])

    # collect data
    rc = ResponseCalculator(mol, pyscf.scf.RHF, coords, b)

    A = []
    scales = 1.2 ** np.arange(-50, 20)
    for pos in coords:
        coeffs = 1
        for dim in range(3):
            coeffs *= rc._integral((abs(pos[dim] - rc._centers[:, dim])) / scales)
            coeffs *= rc._integral((abs(pos[dim] - rc._centers[:, dim + 3])) / scales)
        A.append(coeffs)
    A = np.array(A)
    res = sp_lstsq(A, rc._cache, lapack_driver="gelsy")[0]
    # print(res)
    residuals = rc._cache - A @ res
    return abs(residuals).mean() / abs(rc._cache).mean()
    # np.savez("res.npz", res=res)


if __name__ == "__main__":
    # for b in 1.2 ** np.arange(-50, 20):
    #     print(f"{b:e} {do_case(b)*100:5.3f}")
    b = 1e-2
    print(do_case(b) * 100)
    # plot this, fix the prefactors
    # TODO: tighter SCF convergence?
