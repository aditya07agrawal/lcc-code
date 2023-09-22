"""
LCC Code
"""
# pylint: disable=C0103, missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, cast

import galois
import numpy as np

from galois import FieldArray


if TYPE_CHECKING:
    from typing import Callable, Iterable

    Array = np.ndarray

GF = galois.GF(88772077)


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

    @property
    def K(self):
        return len(self.data)

    @property
    def N(self):
        return len(self.workers)

    def run(self) -> list[FieldArray]:
        result = [
            worker.compute(data) for worker, data in zip(self.workers, self.encode())
        ]
        return list(self.decode(result))

    def encode(self) -> map[Array]:
        lag_poly = compute_lagrange(self.data, encode_by=range(self.K))
        return map(lag_poly, range(self.K, self.K + self.N))

    def decode(self, data: list[Array]) -> map[FieldArray]:
        lag_poly = compute_lagrange(data, encode_by=range(self.K, self.K + self.N))
        return map(lag_poly, range(self.K))


def compute_lagrange(
    matrices: list[Array], encode_by: Iterable[int]
) -> Callable[[int], FieldArray]:
    def _coefficient(x: FieldArray, i: FieldArray) -> FieldArray:
        res = GF(1)
        for j in [GF(j) for j in encode_by]:
            if j != i:
                res *= (x - j) / (i - j)

        return cast(FieldArray, res)

    def polynomial(z: int) -> FieldArray:
        return reduce(
            lambda x, y: x + y,
            [
                matrix * _coefficient(GF(z), GF(i))
                for i, matrix in zip(encode_by, matrices)
            ],
        )

    return polynomial
