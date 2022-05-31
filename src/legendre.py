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
        self._derivs = derivatives

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
        basis_idx = list(range(n_basis))
        A = np.zeros((n_points, 9 * n_basis**6))
        for idx in range(n_points):
            col = 0
            for i, j, b_1, b_2, b_3, b_4, b_5, b_6 in it.product(
                *([range(3)] * 2), *([basis_idx] * 6)
            ):
                bs = b_1, b_2, b_3, b_4, b_5, b_6
                p = 1
                for k in range(3):
                    if k == i:
                        p *= self._derivs[bs[k]](xs[idx][k])
                    else:
                        p *= self._basis[bs[k]](xs[idx][k])
                    if k == j:
                        p *= self._derivs[bs[3 + k]](xs[idx][3 + k])
                    else:
                        p *= self._basis[bs[3 + k]](xs[idx][3 + k])

                A[idx, col] = p
                col += 1
            if idx == 100:
                break

        return A  # plt.imshow(A[:576, :])


# %%
def test():
    scheme = quadpy.u3.schemes["lebedev_011"]()
    radial = (0.2, 1.2, 10)
    coords = np.concatenate([scheme.points.T * _ for _ in radial])

    chi = np.zeros(len(coords) ** 2)
    points = np.zeros((len(coords) ** 2, 6))

    print(f"Calculating {len(chi)} points")
    last = 0
    for ra, rb in it.product(coords, coords):
        chi[last] = exact.nonlocal_susceptibility(ra, rb, 1)
        points[last, :3] = ra
        points[last, 3:] = rb
        last += 1

    max_order = 3
    legs = [
        np.poly1d(np.polynomial.legendre.leg2poly([0] * (order) + [1])[::-1])
        for order in range(max_order)
    ]
    print(legs)
    derivs = [_.deriv(1) for _ in legs]
    print("Fitting")
    P = PolarizabilityBasis(legs, derivs)
    return points, chi, P.fit(points, chi)


A = test()

# %%
# %%
p, c, A = A
plt.imshow(A[:100, :500])

# %%
np.allclose(A[3], A[4]), p[3], p[4], c[3], c[4]
# %%
plt.scatter(A[3], A[4])
# %%
plt.hist(A[:100, :].reshape(-1), bins=50)
# %%
