import pyscf.scf
import pyscf.gto
import pyscf.dft
import pyscf.qmmm
import numpy as np
import numpy.linalg as npl
import tqdm


class Susceptibility:
    """Implements a generalized case of 10.1021/ct1004577, section 2."""

    def __init__(self, mol, calc):
        self._mol = mol
        self._calc = calc
        self._q = 0.01
        self._grid = pyscf.dft.gen_grid.Grids(self._mol)
        self._grid.level = 3
        self._grid.build()
        self._ao_value = pyscf.dft.numint.eval_ao(self._mol, self._grid.coords, deriv=0)
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
        # beta_k, beta_l
        d = np.linalg.norm(self._grid.coords - pos, axis=1)
        combined = self._grid.weights / d

        integrals = np.dot(self._ao_value.T, combined.T)
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
        self._Q = npl.lstsq(B, D)[0].reshape(self._mol.nbas, self._mol.nbas)

    def query(self, r: np.ndarray, rprime: np.ndarray) -> float:
        coords = np.array((r, rprime))
        beta_k, beta_l = pyscf.dft.numint.eval_ao(self._mol, coords, deriv=0)

        return np.sum(self._Q * np.outer(beta_k, beta_l))


if __name__ == "__main__":
    # define molecule
    mol = pyscf.gto.M(
        atom=f"He 0 0 0",
        basis="6-31G",
        verbose=0,
    )

    # regression grid
    N = 100
    coords = np.zeros((N, 3))
    coords[:, 0] = np.linspace(0.1, 5, N)

    # collect data
    s = Susceptibility(mol, pyscf.scf.RHF)
    s.build_susceptibility(coords)

    print(s.query(np.array((0, 0, 0)), np.array((0, 0, 0.1))))
