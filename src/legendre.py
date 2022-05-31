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
        return A, ys
        solution, res, rank, s = np.linalg.lstsq(A, ys)
        residuals = A @ solution - ys
        return np.linalg.norm(residuals), np.linalg.norm(ys)

        #     col += 1
        # print (col, A.shape)
        # # solve
        # return A
        # # return A, ys
        # ys = np.concatenate((ys, ys * 0))
        # print("solving")
        # solution, res, rank, s = np.linalg.lstsq(A, ys, rcond=None)
        # residuals = A @ solution - ys
        # print(np.linalg.norm(residuals), np.linalg.norm(ys))

        # return res


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
    return P.fit(points, chi)


#%%
if __name__ == "__main__":
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()
    test()
    profiler.stop()

    print(profiler.output_text(unicode=True, color=True))
