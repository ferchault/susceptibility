#!/usr/bin/env
#%%
import numpy as np
import pyscf.scf
import pyscf.dft
import pyscf.gto
import itertools as it


class DeltaUHF(pyscf.scf.uhf.UHF):
    def set_delta(self, pos):
        self._pos = [pos]

    def get_hcore(self, mol):
        base = pyscf.scf.uhf.UHF.get_hcore(self, mol)
        ao_at_delta = pyscf.dft.numint.eval_ao(mol, self._pos, deriv=0)
        return base + 0.001 * np.outer(ao_at_delta, ao_at_delta)


def susceptibility_system(r: np.ndarray, rp: np.ndarray, mol) -> float:

    rangstrom = np.array(r) * 0.529177249
    mol.atom.extend( [[ "X", (rangstrom) ]] )

    calc = pyscf.scf.UHF(mol)
    calc.kernel()
    dm1 = calc.make_rdm1()

    ao_value = pyscf.dft.numint.eval_ao(mol, [rp], deriv=0)

    rho1 = pyscf.dft.numint.eval_rho(
        mol, ao_value, dm1[0], xctype="LDA"
    ) + pyscf.dft.numint.eval_rho(mol, ao_value, dm1[1], xctype="LDA")

    # get perturbing delta function
    calc = DeltaUHF(mol)
    calc.set_delta(r)
    calc.kernel()
    dm2 = calc.make_rdm1()
    ao_value = pyscf.dft.numint.eval_ao(mol, [rp], deriv=0)

    rho2 = pyscf.dft.numint.eval_rho(
        mol, ao_value, dm2[0], xctype="LDA"
    ) + pyscf.dft.numint.eval_rho(mol, ao_value, dm2[1], xctype="LDA")

    return (rho2[0] - rho1[0]) / 0.001

def write_to_file():
    mol = setup_mol()
    outfile= open("2watah.txt", "w")

    for xx in np.linspace(-3, 3, 50):
        for yy in np.linspace(-3, 3, 50):
            r = [xx, 0,0]
            rp = [0, yy, 0]
            outfile.write(str(xx) + "  "+str( yy) + "   "+ str(susceptibility_system(r, rp, mol)) + "\n")
            outfile.flush()

def setup_mol():
    basis = {"H": pyscf.gto.uncontract(pyscf.gto.load("def2-TZVPP", "H")), "O": pyscf.gto.uncontract(pyscf.gto.load("def2-TZVPP", "O"))}
    basis["X"] = basis = [[0, [_, 1.0]] for _ in 2.0 ** np.linspace(-10, 20)]

    mol = pyscf.gto.Mole()
    mol.atom = [['O',(0, 0, 0)], ['H', (0.00000,  0.75545, -0.47116)] , ['H', (0.00000, -0.75545,  -0.47116)] ]
    mol.basis=basis
    mol.verbose=0
    mol.spin=0
    mol.symmetry = False
    return mol

def mol_sus(r, rp):
    mol = setup_mol()
    return susceptibility_system(r, rp, mol)


####

#write_to_file()
