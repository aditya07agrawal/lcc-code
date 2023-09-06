"""Script"""
# pylint: disable=invalid-name, missing-function-docstring
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import galois

from worker import Worker, Master

if TYPE_CHECKING:
    Array = np.ndarray


GF = galois.GF(31)


def func(x: Array) -> Array:
    return x.T @ x


X1 = GF([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
X2 = GF([[4, 5, 6], [4, 5, 6], [4, 5, 6]])

Workers = [Worker(computation=func) for i in range(2)]

mst = Master(0, 0, [X1, X2], Workers, func)

print(mst.run())
