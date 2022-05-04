#!/usr/bin/env
#%%
import numpy as np
import pyscf.scf
import pyscf.dft
import pyscf.gto
import exact_susceptibility as exact
import itertools as it


class DeltaUHF(pyscf.scf.uhf.UHF):
    def set_delta(self, pos):
        self._pos = [pos]

    def get_hcore(self, mol):
        base = pyscf.scf.uhf.UHF.get_hcore(self, mol)
        ao_at_delta = pyscf.dft.numint.eval_ao(mol, self._pos, deriv=0)
        return base + 0.001 * np.outer(ao_at_delta, ao_at_delta)


def atom(r: np.ndarray, rp: np.ndarray, Z: float) -> float:
    if Z != 1:
        raise NotImplementedError()

    basis = {"H": pyscf.gto.uncontract(pyscf.gto.load("def2-TZVPP", "H"))}
    basis["X"] = basis = [[0, [_, 1.0]] for _ in 2.0 ** np.linspace(-10, 20)]

    rangstrom = np.array(r) * 0.529177249
    mol = pyscf.gto.M(
        atom=f"H 0 0 0; X {rangstrom[0]} {rangstrom[1]} {rangstrom[2]}",
        basis=basis,
        verbose=0,
        spin=1,
    )
    mol.symmetry = False
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


# r = (0, 0, 0.2)
# ex1 = []
# a = []
# xs = np.linspace(0.1, 0.3, 10)
# for rp in xs:
#     ex1.append(exact.hostler_formula(r, (0, 0, rp), 1))
#     a.append(atom(r, (0, 0, rp), 1))
# #%%
# plt.plot(xs, ex1)
# plt.plot(xs, np.array(a)/2.5)


#%%


#%%
if __name__ == "__main__":
    rs = np.linspace(0.1, 4, 10)
    alphas = np.linspace(0, np.pi, 5)
    for ra, rb, alpha in it.product(rs, rs, alphas):
        r = (0, 0, ra)
        rp = (0, rb * np.sin(alpha), rb * np.cos(alpha))
        numeric = atom(r, rp, 1)
        print(ra, rb, alpha, numeric, exact.nonlocal_susceptibility(r, rp, 1))
    # ex1 = exact.hostler_formula(r, rp, 1)
    # ex2 = exact.nonlocal_susceptibility(r, rp, 1)
    # ex3 = exact.white_formula(r, rp, 1)
    # print(ex1, ex2, ex3, numeric)

# %%
