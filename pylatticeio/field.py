from typing import List

import numpy


class LatticeInfo:
    Ns: int = 4
    Nc: int = 3
    Nd: int = 4

    def __init__(self, latt_size: List[int]) -> None:
        self._checkLattice(latt_size)
        self._setLattice(latt_size)

    def _checkLattice(self, latt_size: List[int]):
        from . import getGridSize

        if getGridSize() is None:
            raise RuntimeError("pyquda.init() must be called before contructing LatticeInfo")
        Gx, Gy, Gz, Gt = getGridSize()
        Lx, Ly, Lz, Lt = latt_size
        assert Lx % (2 * Gx) == 0 and Ly % (2 * Gy) == 0 and Lz % (2 * Gz) == 0 and Lt % (2 * Gt) == 0, (
            "lattice size must be divisible by gird size, "
            "and sublattice size must be even in every direction for consistant even-odd preconditioning"
        )

    def _setLattice(self, latt_size: List[int]):
        from . import getMPIComm, getMPISize, getMPIRank, getGridSize, getGridCoord

        self.mpi_comm = getMPIComm()
        self.mpi_size = getMPISize()
        self.mpi_rank = getMPIRank()
        self.grid_size = getGridSize()
        self.grid_coord = getGridCoord()

        Gx, Gy, Gz, Gt = self.grid_size
        gx, gy, gz, gt = self.grid_coord
        Lx, Ly, Lz, Lt = latt_size

        self.Gx, self.Gy, self.Gz, self.Gt = Gx, Gy, Gz, Gt
        self.gx, self.gy, self.gz, self.gt = gx, gy, gz, gt
        self.global_size = [Lx, Ly, Lz, Lt]
        self.global_volume = Lx * Ly * Lz * Lt
        Lx, Ly, Lz, Lt = Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt
        self.Lx, self.Ly, self.Lz, self.Lt = Lx, Ly, Lz, Lt
        self.size = [Lx, Ly, Lz, Lt]
        self.volume = Lx * Ly * Lz * Lt
        self.size_cb2 = [Lx // 2, Ly, Lz, Lt]
        self.volume_cb2 = Lx * Ly * Lz * Lt // 2
        self.ga_pad = Lx * Ly * Lz * Lt // min(Lx, Ly, Lz, Lt) // 2


Ns, Nc, Nd = LatticeInfo.Ns, LatticeInfo.Nc, LatticeInfo.Nd


def lexico(data: numpy.ndarray, axes: List[int], dtype=None):
    shape = data.shape
    Np, Lt, Lz, Ly, Lx = [shape[axis] for axis in axes]
    assert Np == 2, "There must be 2 parities."
    Lx *= 2
    Npre = int(numpy.prod(shape[: axes[0]]))
    Nsuf = int(numpy.prod(shape[axes[-1] + 1 :]))
    dtype = data.dtype if dtype is None else dtype
    data_cb2 = data.reshape(Npre, 2, Lt, Lz, Ly, Lx // 2, Nsuf)
    data_lexico = numpy.zeros((Npre, Lt, Lz, Ly, Lx, Nsuf), dtype)
    for t in range(Lt):
        for z in range(Lz):
            for y in range(Ly):
                eo = (t + z + y) % 2
                if eo == 0:
                    data_lexico[:, t, z, y, 0::2] = data_cb2[:, 0, t, z, y, :]
                    data_lexico[:, t, z, y, 1::2] = data_cb2[:, 1, t, z, y, :]
                else:
                    data_lexico[:, t, z, y, 1::2] = data_cb2[:, 0, t, z, y, :]
                    data_lexico[:, t, z, y, 0::2] = data_cb2[:, 1, t, z, y, :]
    return data_lexico.reshape(*shape[: axes[0]], Lt, Lz, Ly, Lx, *shape[axes[-1] + 1 :])


def cb2(data: numpy.ndarray, axes: List[int], dtype=None):
    shape = data.shape
    Lt, Lz, Ly, Lx = [shape[axis] for axis in axes]
    Npre = int(numpy.prod(shape[: axes[0]]))
    Nsuf = int(numpy.prod(shape[axes[-1] + 1 :]))
    dtype = data.dtype if dtype is None else dtype
    data_lexico = data.reshape(Npre, Lt, Lz, Ly, Lx, Nsuf)
    data_cb2 = numpy.zeros((Npre, 2, Lt, Lz, Ly, Lx // 2, Nsuf), dtype)
    for t in range(Lt):
        for z in range(Lz):
            for y in range(Ly):
                eo = (t + z + y) % 2
                if eo == 0:
                    data_cb2[:, 0, t, z, y, :] = data_lexico[:, t, z, y, 0::2]
                    data_cb2[:, 1, t, z, y, :] = data_lexico[:, t, z, y, 1::2]
                else:
                    data_cb2[:, 0, t, z, y, :] = data_lexico[:, t, z, y, 1::2]
                    data_cb2[:, 1, t, z, y, :] = data_lexico[:, t, z, y, 0::2]
    return data_cb2.reshape(*shape[: axes[0]], 2, Lt, Lz, Ly, Lx // 2, *shape[axes[-1] + 1 :])
