# %%
import pyscf.scf
import pyscf.gto
import pyscf.dft
import pyscf.qmmm
import numpy as np
import itertools as it
import tqdm
from scipy.sparse.linalg import lsqr
from os.path import exists as file_exists


def regularized_least_squares(A, y, lamb=0):
    n_col = A.shape[1]
    return np.linalg.lstsq(
        A.T.dot(A) + lamb * np.identity(n_col), A.T.dot(y), rcond=None
    )


class ResponseCalculator:
    """Implements a generalized case of 10.1021/ct1004577, section 2."""

    def __init__(self, mol, calc):
        self._mol = mol
        self._calc = calc
        self._q = 0.001
        self._grid = pyscf.dft.gen_grid.Grids(self._mol)
        self._grid.level = 8
        self._grid.build()
        calc = self._calc(self._mol)
        calc.kernel()
        self._center = calc.energy_elec()[0]

    def get_energy_derivative(self, pos):
        up = pyscf.qmmm.mm_charge(
            self._calc(self._mol), np.array((pos,)), np.array((self._q,))
        )
        up.kernel()
        dn = pyscf.qmmm.mm_charge(
            self._calc(self._mol), np.array((pos,)), np.array((-self._q,))
        )
        dn.kernel()
        assert up.converged and dn.converged
        return up.energy_elec()[0] + dn.energy_elec()[0] - 2 * self._center

    def get_derivative(self, pos: np.ndarray):
        d = np.linalg.norm(self._grid.coords - pos, axis=1)
        combined = self._q * self._grid.weights / d
        ao_value = pyscf.dft.numint.eval_ao(self._molresp, self._grid.coords, deriv=0)
        integrals = np.dot(ao_value.T, combined.T)

        B_j = np.outer(integrals, integrals).reshape(-1)
        D_j = self.get_energy_derivative(pos)

        return D_j, B_j

    def build_susceptibility(self, coords: np.ndarray, fit_mol):
        self._molresp = fit_mol
        if file_exists("chi.npy"):
            self._Qvec = np.load("chi.npy")
        else:
            D = []
            B = []

            if len(coords) < self._molresp.nao:
                # print("Would be underdetermined. Aborting.")
                raise ValueError("Underdetermined")
            for coord in tqdm.tqdm(coords, desc="Chi", leave=False):
                # for coord in coords:
                D_j, B_j = self.get_derivative(coord)
                D.append(D_j)
                B.append(B_j)
            D = np.array(D)
            B = np.array(B)

            lstsq = lsqr(B, D)
            res = (np.sqrt((D - B @ lstsq[0]) ** 2).mean()) / np.abs(D).mean()
            print(f"Chi:   Average relative residual {res*100:8.3f} %")
            self._Q = lstsq[0].reshape(self._molresp.nao, self._molresp.nao)
            self._Qvec = lstsq[0]
            np.save("chi.npy", self._Qvec)
            np.save("B.npy", B)
            np.save("D.npy", D)
            return res

    def evaluate_susceptibility(self, r: np.ndarray, rprime: np.ndarray) -> float:
        coords = np.array((r, rprime))
        beta_k, beta_l = pyscf.dft.numint.eval_ao(self._molresp, coords, deriv=0)

        # return np.sum(self._Q * np.outer(beta_k, beta_l))
        return np.dot(self._Qvec, np.outer(beta_k, beta_l).reshape(-1))

    def build_polarizability(self, coords: np.ndarray, alpha_mol):
        self._A = np.zeros((3, 3, alpha_mol.nao, alpha_mol.nao))
        derivs = pyscf.dft.numint.eval_ao(
            alpha_mol, coords, deriv=1
        )  # [0, x, y, z]:[pts]:[nao]

        ncoords = len(coords)
        for i, j in it.product(range(3), range(3)):
            B = []
            D = []
            label = f"A_{i},{j}"
            for r, rprime in tqdm.tqdm(
                it.product(range(ncoords), range(ncoords)),
                total=ncoords**2,
                desc=label,
                leave=False,
            ):
                D.append(self.evaluate_susceptibility(coords[r], coords[rprime]))

                left = derivs[i + 1, r, :]
                right = derivs[j + 1, rprime, :]
                B.append(np.outer(left, right).reshape(-1))

            # lstsq = npl.lstsq(B, D, rcond=None)
            B = np.array(B)
            D = np.array(D)
            lstsq = regularized_least_squares(B, D, 1e-7)
            res = (np.sqrt((D - B @ lstsq[0]) ** 2).mean()) / np.abs(D).mean()
            print(f"{label}: Average relative residual {res*100:8.3f} %")
            A = lstsq[0].reshape(alpha_mol.nao, alpha_mol.nao)
            self._A[i, j, :, :] = A
            return res * 100

    def evaluate_polarizability(self, r: np.ndarray, rprime: np.ndarray) -> float:
        coords = np.array((r, rprime))
        beta_k, beta_l = pyscf.dft.numint.eval_ao(self._molresp, coords, deriv=0)

        return np.sum(self._A * np.outer(beta_k, beta_l), axis=(2, 3))

    def get_ao_integrals(self) -> float:
        basis_set_values = pyscf.dft.numint.eval_ao(
            self._molresp, self._grid.coords, deriv=0
        )
        ao_integrals = np.dot(self._grid.weights, basis_set_values)
        return ao_integrals


def real_space_scan():
    mol = pyscf.gto.M(
        atom=f"He 0 0 0",
        basis="def2-TZVP",
        verbose=0,
    )
    chi_mol = pyscf.gto.M(
        atom=f"He 0 0 0",
        basis="unc-def2-TZVPP",
        verbose=0,
    )
    # basis = [
    #    [0, [2.0 ** bs[0], 1]],
    #    [0, [2.0 ** bs[1], 1]],
    #    [0, [2.0 ** bs[2], 1]],
    #    [0, [2.0 ** bs[3], 1]],
    #    [1, [2.0 ** bs[4], 1]],
    #    [1, [2.0 ** bs[5], 1]],
    #    [1, [2.0 ** bs[6], 1]],
    #    [1, [2.0 ** bs[7], 1]],
    #    [2, [2.0 ** bs[8], 1]],
    #    [2, [2.0 ** bs[9], 1]],
    #    [2, [2.0 ** bs[10], 1]],
    #    [2, [2.0 ** bs[11], 1]],
    # ]
    basis = [
        [1, [2.0**3, 1]],
        [1, [2.0**5, 1]],
        [1, [2.0**8, 1]],
        [1, [2.0**12, 1]],
        # [1, [2.0**14, 1]],
        [3, [2.0**3, 1]],
        [3, [2.0**5, 1]],
        [3, [2.0**8, 1]],
        [3, [2.0**12, 1]],
        # [3, [2.0**14, 1]],
        [5, [2.0**3, 1]],
        [5, [2.0**5, 1]],
        [5, [2.0**8, 1]],
        [5, [2.0**12, 1]],
        # [5, [2.0**14, 1]],
        # [3, [2.0**5, 1]],
        # [3, [2.0**1, 1]],
    ]
    basis = [
        [0, [2.0**3, 1]],
        [0, [2.0**5, 1]],
        [0, [2.0**8, 1]],
        [0, [2.0**12, 1]],
        # [1, [2.0**14, 1]],
        [2, [2.0**3, 1]],
        [2, [2.0**5, 1]],
        [2, [2.0**8, 1]],
        [2, [2.0**12, 1]],
        # [3, [2.0**14, 1]],
        [4, [2.0**3, 1]],
        [4, [2.0**5, 1]],
        [4, [2.0**8, 1]],
        [4, [2.0**12, 1]],
        # [5, [2.0**14, 1]],
        # [3, [2.0**5, 1]],
        # [3, [2.0**1, 1]],
    ]
    alpha_mol = pyscf.gto.M(
        atom=f"H 0 0 0",
        basis={"H": basis},
        spin=1,
        verbose=0,
    )

    grid = pyscf.dft.gen_grid.Grids(mol)
    grid.level = 1
    grid.build()

    rc = ResponseCalculator(mol, pyscf.scf.RHF)

    # random grid subset
    npts = grid.coords.shape[0]
    idx = np.random.choice(npts, 200, replace=False)

    rc.build_susceptibility(grid.coords[idx, :], chi_mol)
    return rc.build_polarizability(grid.coords[idx, :], alpha_mol)


real_space_scan()

best = 10
best_point = None


def avgs(x0):
    global best, best_point
    x = np.average([real_space_scan(x0) for _ in range(5)])
    if x < best:
        best = x
        best_point = x0
    print(best, best_point)
    return x


from scipy.optimize import differential_evolution

# x0 = (14, 12, 5, 3, 14, 10, 5, 3, 14, 12, 10, 8)
# res = minimize(avgs, x0, options={"maxiter": 10})
# print(res)

# bounds = [(2, 8)] * 12
# result = differential_evolution(avgs, bounds, maxiter=10, workers=1)
# print(result)

# print(real_space_scan((3, 4, 5, 6, 8, 4, 5, 6, 3, 4, 5, 12)))
# %%
# separate mol basis for chi and alpha
# compare derivatives of orbitals to FD
[
    [0, [27150.699364, 1]],
    [0, [4070.466736, 1]],
    [0, [926.45124718, 1]],
    [0, [262.40039196, 1]],
    [0, [85.706031782, 1]],
    [0, [31.168371532, 1]],
    [0, [12.4134277016, 1]],
    [0, [5.1529793054, 1]],
    [0, [1.15392678838, 1]],
    [0, [0.45945662716, 1]],
    [0, [0.190328880056, 1]],
    [1, [27150.699364, 1]],
    [1, [4070.466736, 1]],
    [1, [926.45124718, 1]],
    [1, [262.40039196, 1]],
    [1, [85.706031782, 1]],
    [1, [31.168371532, 1]],
    [1, [12.4134277016, 1]],
    [1, [5.1529793054, 1]],
    [1, [1.15392678838, 1]],
    [1, [0.45945662716, 1]],
    [1, [0.190328880056, 1]],
    [2, [2.194, 1]],
    [2, [0.636, 1]],
    [3, [1.522, 1]],
]

# %%
# [11.86094665  5.06935174  8.46099175 10.52933808
# 6.80437205  5.06042316 3.6513395   2.47137537
#   4.73135485  8.1781946   1.72571595  6.98567145]

# [ 6.86462741  7.50794078  8.46099175 10.15008919
# 6.80437205  5.43153594  3.29936681  3.19266769
#   10.1487846   6.48407154  8.4297509   2.87228208]
