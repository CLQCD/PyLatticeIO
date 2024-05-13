# PyLatticeIO

Read and write lattice QCD data in Python with [NumPy](https://github.com/numpy/numpy).

This project was originally the I/O part of [PyQUDA](https://github.com/CLQCD/PyQUDA), and I simplified the code to create this repository. This might be helpful to analysis lattice QCD data in Python.

This repository contains Python implementations to read and write different lattice QCD data such as gauge configurations and propagators generated by [Chroma](https://github.com/JeffersonLab/chroma) software.

Just loading data into NumPy format is useless, and you should use these I/O samples along with your analysis script.

The propagator will be rotated to DeGrand-Rossi basis.

## Returned NDArray
| Type                 | shape                              | dtype        | Comment           |
| -------------------- | ---------------------------------- | ------------ | ----------------- |
| Gauge                | `(Nd, Lt, Lz, Ly, Lx, Nc, Nc)`     | `complex128` | row-colum order   |
| Propagator           | `(Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc)` | `complex128` | sink-source order |
| Staggered propagator | `(Lt, Lz, Ly, Lx, Nc, Nc)`         | `complex128` | sink-source order |
| Fermion              | `(Lt, Lz, Ly, Lx, Ns, Nc)`         | `complex128` |                   |
| Staggered fermion    | `(Lt, Lz, Ly, Lx, Nc)`             | `complex128` |                   |

## Supported format

| Software   | Gauge      | Propagator | Other |
| ---------- | ---------- | ---------- | ----- |
| Chroma     | Read       | Read       |       |
| MILC       | Read       | Read       |       |
| KYU        | Read/Write | Read/Write |       |
| IO General |            |            | Read  |

## MPI I/O with mpi4py
The repository requires [mpi4py](https://github.com/mpi4py/mpi4py) to read lattice QCD data in parallel. You could setup the grid by calling `setGrid(grid_size)`. If you do not call this function, each process will read the entire field.

test.mpi.py:
```Python
import pylatticeio as io

io.setGrid([1, 1, 2, 2])
gauge = io.readChromaQIOGauge("./data/weak_field.lime")
print(gauge.shape)
```

And then run the script by
```Bash
mpiexec -n 4 python test.mpi.py
```

You should get the shape of `gauge` to be `(Nd, Lt/2, Lz/2, Ly, Lx, Nc, Nc)`, since the gauge is divided in `z` and `t` directions.

The rank of a process varies fastest in the `t` direction. For example, with `grid_size=[1, 1, 2, 2]`, the grid coordinate and the corresponding rank of a process is
| item            |           |           |           |           |
| --------------- | --------- | --------- | --------- | --------- |
| rank            | `0`       | `1`       | `2`       | `3`       |
| grid coordinate | `0,0,0,0` | `0,0,0,1` | `0,0,1,0` | `0,0,1,1` |

You can also check this by calling `io.getCoordFromRank` and `io.getRankFromCoord`.
