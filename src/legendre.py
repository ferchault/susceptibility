#!/usr/bin/env python
#%%
"""Explores whether a 6d-expansion in Legendre polynomials yields an efficient description of the polarizability.
"""
from typing import Callable, Iterable
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import exact_susceptibility as exact
import quadpy


class Legendre1D:
    def __init__(self, max_degree: int):
        self._max_degree = max_degree

    def _scale_forward(self, xs: Iterable[float]):
        return (np.copy(xs) - self._shift) / self._scale

    def fit(self, xs: Iterable[float], ys: Iterable[float]):
        # shift and center to get to domain [-1, 1]
        minval, maxval = min(xs), max(xs)
        self._scale = 2 / (maxval - minval)
        self._shift = (maxval + minval) / 2
        xs = self._scale_forward(xs)

        self._coefficients = np.polynomial.legendre.legfit(xs, ys, self._max_degree)
        print(self._coefficients)

    def evaluate(self, xs: Iterable[float]):
        xs = self._scale_forward(xs)
        return np.polynomial.legendre.legval(xs, self._coefficients)

    @staticmethod
    def test():
        L = Legendre1D(8)
        xs = np.linspace(-1, 1, 50)
        ys = np.exp(-(5 * xs**2))
        L.fit(xs, ys)
        plt.plot(xs, ys)
        plt.plot(xs, L.evaluate(xs))


import pyscf.dft
import pyscf.gto
import tqdm


class PolarizabilityBasis:
    def __init__(
        self,
        basis: Iterable[Callable],
        derivatives: Iterable[Callable],
    ):
        self._basis = basis

        self._derivs = []
        for derivative in derivatives:
            if derivative == np.poly1d([0]):
                derivative = None
            self._derivs.append(derivative)

    def response_basis_set(self, refbasis, element, scale):
        uncontracted = pyscf.gto.uncontract(pyscf.gto.load(refbasis, element))
        basis = []
        for basisfunction in uncontracted:
            basisfunction[1][0] *= scale
            basis.append(basisfunction)
        molresp = pyscf.gto.M(
            atom=f"H 0 0 0",
            basis={"H": basis},
            spin=1,
            verbose=0,
        )
        self._molresp = molresp

    def fit_ao_basis(self, points):
        derivs = pyscf.dft.numint.eval_ao(
            self._molresp, points, deriv=1
        )  # [0, x, y, z]:[pts]:[nao]

        ncoords = len(points)
        nao = self._molresp.nao
        A = np.zeros((ncoords**2, 9 * nao**2))
        ys = np.zeros(ncoords**2)
        for i, j in it.product(range(3), range(3)):
            label = f"A_{i},{j}"
            row = 0
            for r, rprime in tqdm.tqdm(
                it.product(range(ncoords), range(ncoords)),
                total=ncoords**2,
                desc=label,
                leave=False,
            ):
                if i == 0 and j == 0:
                    ys[row] = exact.nonlocal_susceptibility(
                        points[r], points[rprime], 1
                    )

                left = derivs[i + 1, r, :]
                right = derivs[j + 1, rprime, :]
                A[row, (i * 3 + j) * nao**2 : (i * 3 + j + 1) * nao**2] = np.outer(
                    left, right
                ).reshape(-1)

                row += 1

        # filter
        mask = np.isfinite(ys)
        ys = ys[mask]
        A = A[mask, :]
        solution, res, rank, s = np.linalg.lstsq(A, ys, rcond=None)
        residuals = A @ solution - ys
        print(np.linalg.norm(residuals), np.linalg.norm(ys))
        return solution

    def fit(self, points: Iterable[Iterable[float]], susceptibilities: Iterable[float]):
        xs = np.array(points)
        ys = np.array(susceptibilities)
        assert xs.shape[1] == 6
        assert len(xs) == len(ys)

        # scale
        xs -= np.mean(xs)
        xs /= np.amax(np.abs(xs))

        n_basis = len(self._basis)
        n_points = len(ys)
        # limit for development
        # n_points = 7000
        # xs = xs[:n_points, :]
        # ys = ys[:n_points]
        # end limit
        basis_idx = list(range(n_basis))
        A = np.zeros((3 * n_points, 9 * n_basis**6))

        col = 0
        for i, j, b_1, b_2, b_3, b_4, b_5, b_6 in it.product(
            *([range(3)] * 2), *([basis_idx] * 6)
        ):
            bs = b_1, b_2, b_3, b_4, b_5, b_6

            # force symmetry alpha(r, rp) == alpha(rp, r), zero-pad ys
            pleft = 1
            pright = 1
            for k, basisindex in enumerate(bs):
                pleft *= self._basis[basisindex](xs[:, k])
            for k, basisindex in enumerate(list(bs[3:]) + list(bs[:3])):
                pright *= self._basis[basisindex](xs[:, k])
            A[n_points : 2 * n_points, col] = pleft - pright

            # force symmetry a_ij = a_ji, zero-pad ys
            if i > j:
                prefactor = 1
            if j > i:
                prefactor = -1
            if i == j:
                prefactor = 0
            A[2 * n_points :, col] = prefactor * pleft

            # actual susceptibility fit
            if self._derivs[bs[i]] is not None and self._derivs[bs[j]] is not None:
                d1 = self._derivs[bs[i]](xs[:, i])
                d2 = self._derivs[bs[j]](xs[:, j])
                p = d1 * d2
                for k in range(3):
                    if k != i:
                        p *= self._basis[bs[k]](xs[:, k])
                    if k != j:
                        p *= self._basis[bs[3 + k]](xs[:, 3 + k])

                A[:n_points, col] = p

            # move counter
            col += 1

        # solve
        ys = np.concatenate((ys, ys * 0, ys * 0))
        solution, res, rank, s = np.linalg.lstsq(A, ys, rcond=None)
        residuals = A @ solution - ys
        print(np.linalg.norm(residuals), np.linalg.norm(ys))


# %%
def test():
    # scheme = quadpy.u3.schemes["lebedev_011"]()
    # radial = (0.2, 1.2, 10)
    # coords = np.concatenate([scheme.points.T * _ for _ in radial])
    N = 33
    coords = np.zeros((N * 3, 3))
    xs = np.linspace(0.01, 5, N)
    coords[:N, 0] = xs
    coords[N : 2 * N, 1] = xs
    coords[2 * N :, 2] = xs

    chi = np.zeros(len(coords) ** 2)
    points = np.zeros((len(coords) ** 2, 6))

    print(f"Calculating {len(chi)} points")
    last = 0
    for ra, rb in it.product(coords, coords):
        points[last, :3] = ra
        points[last, 3:] = rb
        last += 1
    points += 0.1 * np.random.normal(size=points.shape)
    for last in range(len(chi)):
        chi[last] = exact.nonlocal_susceptibility(points[last, :3], points[last, 3:], 1)

    mask = np.isnan(chi)
    chi = chi[~mask]
    points = points[~mask]

    max_order = 3
    legs = [
        np.poly1d(np.polynomial.legendre.leg2poly([0] * (order) + [1])[::-1])
        for order in range(max_order)
    ]
    print(legs)
    derivs = [_.deriv(1) for _ in legs]
    print(derivs)
    print("Fitting")
    P = PolarizabilityBasis(legs, derivs)
    # P.fit(points, chi)
    P.response_basis_set("def2-TZVP", "C", 2)
    return P.fit_ao_basis(coords)


def to_points(phi: float, magr: float, magrp: float):
    return np.array((magr, 0, 0)), np.array(
        (np.cos(phi) * magrp, np.sin(phi) * magrp, 0)
    )


def to_value(mol, rs, rps, i, j, solution):
    nao = mol.nao
    ijsolution = solution[(i * 3 + j) * nao**2 : (i * 3 + j + 1) * nao**2]
    xaovalue = pyscf.dft.numint.eval_ao(mol, rs, deriv=0)
    yaovalue = pyscf.dft.numint.eval_ao(mol, rps, deriv=0)
    print(ijsolution.shape, xaovalue.shape, yaovalue.shape)
    values = []
    for idx in range(len(rs)):
        values.append(
            np.sum(ijsolution * np.outer(xaovalue[idx], yaovalue[idx]).reshape(-1))
        )
    return np.array(values)


def do_panel(ax, i, j, angle, solution):
    N = 100
    mags = np.linspace(0, 0.5, N)
    xs = np.repeat(mags, N)
    ys = np.tile(mags, N)
    rs, rps = np.zeros((N * N, 3)), np.zeros((N * N, 3))
    rs[:, 0] = xs
    rps[:, 0] = np.cos(angle) * ys
    rps[:, 1] = np.sin(angle) * ys
    P = PolarizabilityBasis([], [])
    P.response_basis_set("def2-TZVP", "C", 2)
    values = to_value(P._molresp, rs, rps, i, j, solution)
    print(min(values), max(values))
    # plt.imshow(values.reshape(N, N), cmap="Blues", vmin=-5)
    # levels = np.percentile(values, np.arange(0, 110, 10))
    levels = np.linspace(np.percentile(values, 0.1), max(values), 8)

    print(levels)
    ax.contourf(values.reshape(N, N), levels, cmap="Blues_r")
    # plt.colorbar()


def do_panels(solution):
    f, axs = plt.subplots(5, 9, sharex=True, sharey=True)
    ivals = np.repeat((0, 1, 2), 3)
    jvals = np.tile((0, 1, 2), 3)
    angles = (0, 45, 90, 135, 180)
    for row in range(5):
        for col in range(9):
            do_panel(axs[row, col], ivals[col], jvals[col], angles[row], solution)
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("panels.svg")


#%%
if __name__ == "__main__":
    # from pyinstrument import Profiler

    # profiler = Profiler()
    # profiler.start()
    solution = test()
    do_panels(solution)
    # profiler.stop()

    # print(profiler.output_text(unicode=True, color=True))
