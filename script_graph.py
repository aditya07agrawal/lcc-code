# pylint: disable=invalid-name, missing-function-docstring
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
import galois

from worker import Worker, Master

if TYPE_CHECKING:
    Array = np.ndarray

p = 88772077
GF = galois.GF(p)        #O(2^26)
sp = int(np.sqrt(p))

rng = np.random.default_rng()

def func(x: Array) -> Array:
    return x.T @ x

n = 10      #100
t = 0
s = 0
k = 5
N = 10
mu = 0
sigma = 1

Workers = [Worker(computation=func) for i in range(N)]

e = [[],[],[]]
for m in [10,20]:           #[100,300,500,1000,2000,3000,5000]:
    X_orig = [rng.normal(mu, sigma, size=(m,n)) for _ in range(k)]
    X = [GF(((sp*Xi).astype(int))%p) for Xi in X_orig]

    mst = Master(0, 0, X, Workers, func)
    lcc_op = mst.run()
    
    quantize_then_op = []
    for matrix in X:
        quantize_then_op.append(matrix.T @ matrix)
    
    op = []
    for matrix in X_orig:
        op.append(matrix.T @ matrix)
    op_then_quantize = [GF(((p*op_i).astype(int))%p) for op_i in op]

    e_ab = []
    e_ac = []
    e_bc = []
    for (a,b,c) in zip(lcc_op, quantize_then_op, op_then_quantize):
        e_ab.append( np.linalg.norm((a-b), 2)/np.linalg.norm(b, 2) )
        e_ac.append( np.linalg.norm((a-c), 2)/np.linalg.norm(c, 2) )
        e_bc.append( np.linalg.norm((b-c), 2)/np.linalg.norm(c, 2) )

    e_ab_total = np.sqrt(np.square(e_ab).mean())
    e_ac_total = np.sqrt(np.square(e_ac).mean())
    e_bc_total = np.sqrt(np.square(e_bc).mean())
    
    e[0].append(e_ab_total)
    e[1].append(e_ac_total)
    e[2].append(e_bc_total)

    print(f"Iteration with m={m} completed")

plt.plot(e[0])
plt.show()
plt.plot(e[1])
plt.show()
plt.plot(e[2])
plt.show()
