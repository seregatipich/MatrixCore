# MatrixCore

**MatrixCore** solves dense linear systems `Ax = b` using **30 solvers written
from scratch in C**. The numerical core depends only on the C standard library;
NumPy and SciPy are used solely for the Python interface and file I/O.

MatrixCore is an **educational / research** library — a readable implementation
of classic algorithms. For production numerical work, prefer the LAPACK-backed
routines in `scipy.linalg` and `numpy.linalg`.

```python
import numpy as np
from matrixcore import solve_system

A = np.array([[4.0, 1.0], [1.0, 3.0]])
b = np.array([5.0, 4.0])
x = solve_system(A, b, method="cholesky")
```

```{toctree}
:maxdepth: 2
:caption: Contents

tutorial
api
```

## Indices

- {ref}`genindex`
- {ref}`search`
