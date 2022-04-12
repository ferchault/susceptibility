import pyscf.scf
import pyscf.gto
import pyscf.dft
import pyscf.qmmm
import numpy as np
import itertools as it
import numpy.linalg as npl
import tqdm
import sys
from scipy import integrate
from scipy.sparse.linalg import lsqr
from scipy.linalg import lstsq as scipy_lstsq

def transformed_coulomb(s, coord, pos):
    return np.exp(-s**2 * np.linalg.norm(coord-pos)**2)

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
        return up.energy_elec()[0] + dn.energy_elec()[0] - 2 * self._center


    def get_derivative(self, pos: np.ndarray):
        coords = self._gridresp.coords - pos
        d = np.linalg.norm(coords, axis=1)
        combined = self._q * self._gridresp.weights / d
        ao_value = pyscf.dft.numint.eval_ao(self._molresp, coords, deriv=0)
        integrals = np.dot(ao_value.T, combined.T)

        # alternative integral scheme

        #weights = self._q * self._grid.weights * 2.0 / np.sqrt(np.pi) 
        #b_integral= []
        #for ii in range(len(self._grid.coords)):
        #    b_integral.append( integrate.quad(transformed_coulomb, 0.0, np.inf, args=(self._grid.coords[ii], pos))[0] )
        #weights = weights * b_integral
        #integral2 = np.dot(ao_value.T, weights.T)

        # compare the two integrals
        #print("new integral: ",integral2)
        #print("old integral: ",integrals)


        B_j = np.outer(integrals, integrals).reshape(-1)
        D_j = self.get_energy_derivative(pos)

        return D_j, B_j

    def build_susceptibility(self, coords: np.ndarray):
        D = []
        B = []

        if len(coords) < self._molresp.nao:
            print("Would be underdetermined. Aborting.")
            sys.exit(1)
        for coord in tqdm.tqdm(coords, desc="Chi", leave=False):
            D_j, B_j = self.get_derivative(coord)
            D.append(D_j)
            B.append(B_j)
        D = np.array(D)
        B = np.array(B)

        lstsq = npl.lstsq(B, D, rcond=None)
        #lstsq = lsqr(B, D)
        #lstsq = scipy_lstsq(B, D)
        res = (np.sqrt((D - B @ lstsq[0]) ** 2).mean()) / np.abs(D).mean()
        print(f"Chi:   Average relative residual {res*100:8.3f} %")
        self._Q = lstsq[0].reshape(self._molresp.nao, self._molresp.nao)
        self._Qvec = lstsq[0]

    def evaluate_susceptibility(self, r: np.ndarray, rprime: np.ndarray) -> float:
        coords = np.array((r, rprime))
        beta_k, beta_l = pyscf.dft.numint.eval_ao(self._molresp, coords, deriv=0)

        #return np.sum(self._Q * np.outer(beta_k, beta_l))
        return np.dot(self._Qvec,np.outer(beta_k, beta_l).reshape(-1))

    def build_polarizability(self, coords: np.ndarray):
        self._A = np.zeros((3, 3, self._mol.nao, self._mol.nao))
        derivs = pyscf.dft.numint.eval_ao(
            self._mol, coords, deriv=1
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

            lstsq = npl.lstsq(B, D, rcond=None)
            res = (np.sqrt((D - B @ lstsq[0]) ** 2).mean()) / np.abs(D).mean()
            print(f"{label}: Average relative residual {res*100:8.3f} %")
            A = lstsq[0].reshape(self._molresp.nao, self._molresp.nao)
            self._A[i, j, :, :] = A

    def evaluate_polarizability(self, r: np.ndarray, rprime: np.ndarray) -> float:
        coords = np.array((r, rprime))
        beta_k, beta_l = pyscf.dft.numint.eval_ao(self._molresp, coords, deriv=0)

        return np.sum(self._A * np.outer(beta_k, beta_l), axis=(2, 3))

    def get_ao_integrals(self) -> float:
        basis_set_values = pyscf.dft.numint.eval_ao(self._molresp, self._gridresp.coords, deriv=0)
        ao_integrals = np.dot(self._gridresp.weights,basis_set_values)
        return ao_integrals

    def response_basis_set(self):
        molresp = pyscf.gto.M(
            atom=f"H 0 0 0",
            #atom=f"N 0 0 0; N 0 0 1",
            #basis="unc-def2-TZVP",
            basis="unc-aug-cc-pVTZ",
            spin=1,
            verbose=0,
         )
        self._molresp = molresp
        self._calcresp = pyscf.scf.RHF
        self._gridresp = pyscf.dft.gen_grid.Grids(self._molresp)
        self._gridresp.level = 8
        self._gridresp.build()
        calc = self._calcresp(self._molresp)
        calc.kernel()


if __name__ == "__main__":
    # define molecule
    mol = pyscf.gto.M(
        atom=f"He 0 0 0",
        #atom=f"N 0 0 0; N 0 0 1",
        basis="unc-def2-TZVP",
        #basis="unc-aug-cc-pVTZ",
        verbose=0,
    )


    # dft integration grid
    grid = pyscf.dft.gen_grid.Grids(mol)
    grid.level = 4
    grid.build()


    # uniform cubic grid
    N = 10
    x = np.linspace(-3,3,N)
    y = np.linspace(-3,3,N)
    z = np.linspace(-3,3,N)
    xx, yy, zz = np.meshgrid(x, y, z)
    coords=[]
    for ii in range(xx.shape[0]):
        for jj in range(xx.shape[1]):
            for kk in range(xx.shape[0]):
                coords.append(np.asarray([xx[ii][jj][kk], yy[ii][jj][kk], zz[ii][jj][kk]])) #is there a need to make sure theres no grid at (0,0,0)?

    # collect data
    rc = ResponseCalculator(mol, pyscf.scf.RHF)
    rc.response_basis_set()
    #rc.build_susceptibility(grid.coords)
    rc.build_susceptibility(coords)
    #rc.build_polarizability(coords)

    #print(rc._Q.shape)
    print(" sum of q: ",np.sum(rc._Q))
    chi = np.dot(rc._Qvec,np.outer(rc.get_ao_integrals(), rc.get_ao_integrals()).reshape(-1))
    print("chi = :",chi, " . If this value is not close to zero, then something is wrong! ")

    print(
     "chitest",
     rc.evaluate_susceptibility(np.array((0, 0, 0)), np.array((0, 0, 0.1))),
    )
    #print(
    #    "alphatest",
    #    rc.evaluate_polarizability(np.array((0, 0, 0)), np.array((0, 0, 0.1))),
   # )
