from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
import galois

from worker import Worker, Master

if TYPE_CHECKING:
    Array = np.ndarray

GF = galois.GF(88772077)        #O(2^26)

g = [0.1, 0.2, 0.3]
print(np.square(g))
print(np.square(g).mean())
print(np.sqrt(np.square(g).mean()))