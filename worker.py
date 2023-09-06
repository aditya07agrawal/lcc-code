"""
LCC Code
"""
# pylint: disable=C0103, missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Callable

import galois
import numpy as np


if TYPE_CHECKING:
    Array = np.ndarray

GF = galois.GF(31)


@dataclass
class Worker:
    """Worker"""

    computation: Callable

    def compute(self, data: Array):
        return self.computation(data)


@dataclass
class Master:
    """Master"""

    resiliency: int
    security: int
    data: list[Array]
    workers: list[Worker]
    computation: Callable

    def run(self):
        encoded_data = self.encode()
        print("Encoded data:")
        print(encoded_data := list(encoded_data))

        result = [
            worker.compute(data) for worker, data in zip(self.workers, encoded_data)
        ]
        print("Result: ")
        print(result)

        print("Decoded result: ")
        return list(self.decode(result))

    def encode(self) -> map[Array]:
        lag_poly = compute_lagrange(self.data)
        return map(lag_poly, range(len(self.data), len(self.workers) + len(self.data)))

    def decode(self, computed_data: list[Array]) -> map[Array]:
        lag_poly = compute_lagrange(computed_data, encode_by=[2, 3])
        return map(lag_poly, range(0, len(self.data)))


def compute_lagrange(
    matrices: list[Array], encode_by: list[int] | None = None
) -> Callable[[int], Array]:
    def _factor(x: int, i: int, j: int) -> int:
        return (x - j) / (i - j)

    m = len(matrices)
    if encode_by is None:
        encode_by = list(range(m))

    def _coefficient(x: int, i: int) -> int:
        x = GF(x)
        i = GF(i)
        res = GF(1)
        for j in [GF(j) for j in encode_by]:
            if j != i:
                res *= (x - j) / (i - j)

        return res

    def polynomial(z: int) -> Array:
        return reduce(
            lambda x, y: x + y,
            [matrix * _coefficient(z, i) for i, matrix in zip(encode_by, matrices)],
        )

    return polynomial
