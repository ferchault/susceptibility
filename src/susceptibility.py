import pyscf.scf
import pyscf.gto
import pyscf.dft
import pyscf.qmmm
import numpy as np
import itertools as it
import numpy.linalg as npl
import tqdm


class ResponseCalculator:
    """Implements a generalized case of 10.1021/ct1004577, section 2."""

    def __init__(self, mol, calc):
        self._mol = mol
        self._calc = calc
        self._q = 0.01
        self._grid = pyscf.dft.gen_grid.Grids(self._mol)
        self._grid.level = 1
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
        return up.energy_elec()[0] + dn.energy_elec()[0] - 2 * self._center

    def get_derivative(self, pos: np.ndarray):
        coords = self._grid.coords - pos
        d = np.linalg.norm(coords, axis=1)
        combined = self._q * self._grid.weights / d

        ao_value = pyscf.dft.numint.eval_ao(self._mol, coords, deriv=0)
        integrals = np.dot(ao_value.T, combined.T)
        B_j = np.outer(integrals, integrals).reshape(-1)
        D_j = self.get_energy_derivative(pos)

        return D_j, B_j

    def build_susceptibility(self, coords: np.ndarray):
        D = []
        B = []
        for coord in tqdm.tqdm(coords):
            D_j, B_j = self.get_derivative(coord)
            D.append(D_j)
            B.append(B_j)
        D = np.array(D)
        B = np.array(B)

        self._Q = npl.lstsq(B, D, rcond=None)[0].reshape(self._mol.nao, self._mol.nao)

    def evaluate_susceptibility(self, r: np.ndarray, rprime: np.ndarray) -> float:
        coords = np.array((r, rprime))
        beta_k, beta_l = pyscf.dft.numint.eval_ao(self._mol, coords, deriv=0)

        return np.sum(self._Q * np.outer(beta_k, beta_l))

    def build_polarizability(self, coords: np.ndarray):
        self._A = np.zeros((3, 3, self._mol.nao, self._mol.nao))
        derivs = pyscf.dft.numint.eval_ao(
            self._mol, coords, deriv=1
        )  # [0, x, y, z]:[pts]:[nao]

        for i, j in it.product(range(3), range(3)):
            B = []
            D = []
            for r, rprime in it.product(range(len(coords)), range(len(coords))):
                D.append(self.evaluate_susceptibility(coords[r], coords[rprime]))

                left = derivs[i + 1, r, :]
                right = derivs[j + 1, rprime, :]
                B.append(np.outer(left, right).reshape(-1))

            A = npl.lstsq(B, D, rcond=None)[0].reshape(self._mol.nao, self._mol.nao)
            self._A[i, j, :, :] = A

    def evaluate_polarizability(self, r: np.ndarray, rprime: np.ndarray) -> float:
        coords = np.array((r, rprime))
        beta_k, beta_l = pyscf.dft.numint.eval_ao(self._mol, coords, deriv=0)

        return np.sum(self._A * np.outer(beta_k, beta_l), axis=(2, 3))


if __name__ == "__main__":
    # define molecule
    mol = pyscf.gto.M(
        atom=f"He 0 0 0",
        # atom=f"N 0 0 0; N 0 0 1",
        basis="6-31G",
        verbose=0,
    )

    # regression grid
    N = 110
    coords = np.zeros((N, 3))
    coords[:, 0] = np.linspace(0.1, 5, N)

    # collect data
    rc = ResponseCalculator(mol, pyscf.scf.RHF)
    rc.build_susceptibility(coords)
    rc.build_polarizability(coords)

    print(
        "chitest",
        rc.evaluate_susceptibility(np.array((0, 0, 0)), np.array((0, 0, 0.1))),
    )
    print(
        "alphatest",
        rc.evaluate_polarizability(np.array((0, 0, 0)), np.array((0, 0, 0.1))),
    )
