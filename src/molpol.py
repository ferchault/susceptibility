# %%
import pyscf.scf
import pyscf.gto
import pyscf.dft
import pyscf.qmmm
import numpy as np
import itertools as it
from scipy.sparse.linalg import lsqr
from os.path import exists as file_exists
import time

def regularized_least_squares(A, y, lamb=0):
    n_col = A.shape[1]
    return np.linalg.lstsq(
        A.T.dot(A) + lamb * np.identity(n_col), A.T.dot(y), rcond=None
    )

def CutOutFar(gridcoords, atomobject, radius): #BUGGED
    notFarCoords = []
    for gridpoint in gridcoords:
        for atompoint in atomobject:
            if np.linalg.norm(gridpoint - atompoint[1]) < radius:
                notFarCoords.append(gridpoint)
    return np.asarray(notFarCoords)

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
            #for coord in tqdm.tqdm(coords, desc="Chi", leave=False):
            for coord in coords:
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
        self._polresp = alpha_mol
        if file_exists("A.npy"):
            self._A = np.load("A.npy")
        else:
            self._A = np.zeros((3, 3, alpha_mol.nao, alpha_mol.nao))
            derivs = pyscf.dft.numint.eval_ao(
                alpha_mol, coords, deriv=1
            )  # [0, x, y, z]:[pts]:[nao]
            #print(derivs.shape, alpha_mol.nao)

            ncoords = len(coords)
            for i, j in it.product(range(3), range(3)):
                B = []
                D = []
                label = f"A_{i},{j}"
            #for r, rprime in tqdm.tqdm(
            #    it.product(range(ncoords), range(ncoords)),
            #    total=ncoords**2,
            #    desc=label,
            #    leave=False,
            #):
                for r, rprime in it.product(range(ncoords), range(ncoords)):
                    D.append(self.evaluate_susceptibility(coords[r], coords[rprime]))

                    left = derivs[i + 1, r, :]
                    right = derivs[j + 1, rprime, :]
                    B.append(np.outer(left, right).reshape(-1))

            # lstsq = npl.lstsq(B, D, rcond=None)
                B = np.array(B)
                D = np.array(D)
                #print(B.shape, D.shape)
                lstsq = regularized_least_squares(B, D, 1e-7)
                res = (np.sqrt((D - B @ lstsq[0]) ** 2).mean()) / np.abs(D).mean()
                print(f"{label}: Average relative residual {res*100:8.3f} %")
                A = lstsq[0].reshape(alpha_mol.nao, alpha_mol.nao)
                self._A[i, j, :, :] = A
            np.save("A.npy",self._A)
            return res * 100

    def evaluate_polarizability(self, r: np.ndarray, rprime: np.ndarray) -> float:
        coords = np.array((r, rprime))
        beta_k, beta_l = pyscf.dft.numint.eval_ao(self._polresp, coords, deriv=0)

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
    basis = [
        [1, [2.0**2, 1]],
        [1, [2.0**3, 1]],
        [1, [2.0**4, 1]],
        [1, [2.0**5, 1]],
        [1, [2.0**6, 1]],

        [3, [2.0**2, 1]],
        [3, [2.0**3, 1]],
        [3, [2.0**4, 1]],
        [3, [2.0**5, 1]],
        [3, [2.0**6, 1]],

        [5, [2.0**2, 1]],
        [5, [2.0**3, 1]],
        [5, [2.0**4, 1]],
        [5, [2.0**5, 1]],
        [5, [2.0**6, 1]],
    ]
    #print(basis)
    alpha_mol = pyscf.gto.M(
        atom=f"H 0 0 0",
        basis={"H": basis},
        spin=1,
        verbose=0,
    )

    grid = pyscf.dft.gen_grid.Grids(mol)
    grid.level = 3
    grid.build()

    coords = CutOutFar(grid.coords, alpha_mol._atom, 5)
    #coords = grid.coords[np.linalg.norm(grid.coords, axis=1) < 5]

    rc = ResponseCalculator(mol, pyscf.scf.RHF)

    # random grid subset
    npts = coords.shape[0]
    idx = np.random.choice(npts, 200, replace=False)

    rc.build_susceptibility(grid.coords[idx, :], chi_mol)
    rc.build_polarizability(coords[idx, :], alpha_mol)
    return rc


##############################################################################

start_time = time.time()

nlPol = real_space_scan() # Sets up the response calculator

# NLPol scan for an atom
angle = np.pi/4.0
UnitVec1 = (1.0, 0.0, 0.0)
UnitVec2 = (np.cos(angle), np.sin(angle),0)
distance_range = np.linspace(start=0.2, stop=2) #50 pts

#for pairs in it.product(distance_range, repeat=2):
#    d1, d2 = np.multiply(UnitVec1,pairs[0]), np.multiply(UnitVec2,pairs[1])
#    pol = nlPol.evaluate_polarizability(d1, d2)
#    print(pairs[0], pairs[1], np.trace(pol))


print(nlPol.evaluate_polarizability((0.0, 0.1, 0.5), (0.1, 0.3, 0.8)))

print(time.time() - start_time, "seconds")

