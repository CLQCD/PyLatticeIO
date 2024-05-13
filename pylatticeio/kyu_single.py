from os import path
from typing import List

import numpy

from .field import Ns, Nc, LatticeInfo


# matrices to convert gamma basis bewteen DeGrand-Rossi and Dirac-Pauli
# \psi(DP) = _DR_TO_DP \psi(DR)
# \psi(DR) = _DP_TO_DR \psi(DP)
_DP_TO_DR = [
    [0, 1, 0, -1],
    [-1, 0, 1, 0],
    [0, 1, 0, 1],
    [-1, 0, -1, 0],
]
_DR_TO_DP = [
    [0, -1, 0, -1],
    [1, 0, 1, 0],
    [0, 1, 0, -1],
    [-1, 0, 1, 0],
]


def rotateToDiracPauli(propagator: numpy.ndarray):
    A = numpy.asarray(_DP_TO_DR)
    Ainv = numpy.asarray(_DR_TO_DP) / 2

    return numpy.ascontiguousarray(numpy.einsum("ij,tzyxjkab,kl->tzyxilab", Ainv, propagator, A, optimize=True))


def rotateToDeGrandRossi(propagator: numpy.ndarray):
    A = numpy.asarray(_DR_TO_DP)
    Ainv = numpy.asarray(_DP_TO_DR) / 2

    return numpy.ascontiguousarray(numpy.einsum("ij,tzyxjkab,kl->tzyxilab", Ainv, propagator, A, optimize=True))


def fromPropagatorBuffer(filename: str, offset: int, dtype: str, latt_info: LatticeInfo):
    from . import openMPIFileRead, getMPIDatatype

    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size
    native_dtype = dtype if not dtype.startswith(">") else dtype.replace(">", "<")

    fh = openMPIFileRead(filename)
    propagator_raw = numpy.empty((Ns, Nc, Lt, Lz, Ly, Lx, Ns, Nc), native_dtype)
    filetype = getMPIDatatype(native_dtype).Create_subarray(
        (Ns, Nc, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Ns, Nc),
        (Ns, Nc, Lt, Lz, Ly, Lx, Ns, Nc),
        (0, 0, gt * Lt, gz * Lz, gy * Ly, gx * Lx, 0, 0),
    )
    filetype.Commit()
    fh.Set_view(offset, filetype=filetype)
    fh.Read_all(propagator_raw)
    filetype.Free()
    fh.Close()

    return propagator_raw.transpose(2, 3, 4, 5, 6, 0, 7, 1).view(dtype).astype("<c16")


def toPropagatorBuffer(filename: str, offset: int, propagator_raw: numpy.ndarray, dtype: str, latt_info: LatticeInfo):
    from . import openMPIFileWrite, getMPIDatatype

    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size
    native_dtype = dtype if not dtype.startswith(">") else dtype.replace(">", "<")

    fh = openMPIFileWrite(filename)
    propagator_raw = propagator_raw.astype(dtype).view(native_dtype).transpose(5, 7, 0, 1, 2, 3, 4, 6).copy()
    filetype = getMPIDatatype(native_dtype).Create_subarray(
        (Ns, Nc, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Ns, Nc),
        (Ns, Nc, Lt, Lz, Ly, Lx, Ns, Nc),
        (0, 0, gt * Lt, gz * Lz, gy * Ly, gx * Lx, 0, 0),
    )
    filetype.Commit()
    fh.Set_view(offset, filetype=filetype)
    fh.Write_all(propagator_raw)
    filetype.Free()
    fh.Close()


def readPropagator(filename: str, latt_size: List[int]):
    filename = path.expanduser(path.expandvars(filename))
    latt_info = LatticeInfo(latt_size)

    return rotateToDeGrandRossi(fromPropagatorBuffer(filename, 0, "<c8", latt_info))


def writePropagator(filename: str, propagator: numpy.ndarray):
    from . import getGridSize

    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = propagator.shape[0:4][::-1]
    Gx, Gy, Gz, Gt = getGridSize()
    latt_info = LatticeInfo([Gx * Lx, Gy * Ly, Gz * Lz, Gt * Lt])

    toPropagatorBuffer(filename, 0, rotateToDiracPauli(propagator), "<c8", latt_info)


class StopWatch:
    def __init__(self) -> None:
        self.value = 0
        self._time = 0

    def reset(self):
        self.value = 0
        self._time = 0

    def start(self):
        from . import getMPIComm
        from time import perf_counter

        getMPIComm().Barrier()
        self._time = perf_counter()

    def stop(self):
        from . import getMPIComm
        from time import perf_counter

        getMPIComm().Barrier()
        self.value += perf_counter() - self._time
        self._time = 0

    def __str__(self):
        return f"{self.value:.3f}"


def writePropagatorFast(filename: str, propagator: numpy.ndarray):
    from . import getGridSize

    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = propagator.shape[0:4][::-1]
    Gx, Gy, Gz, Gt = getGridSize()
    latt_info = LatticeInfo([Gx * Lx, Gy * Ly, Gz * Lz, Gt * Lt])

    stopWatch = StopWatch()
    stopWatch.reset()
    stopWatch.start()
    propagator = rotateToDiracPauli(propagator)
    stopWatch.stop()
    print(f"Rotate: {stopWatch} secs")
    stopWatch.reset()
    stopWatch.start()
    propagator.data = propagator.data.astype("<c8").transpose(5, 7, 0, 1, 2, 3, 4, 6)
    stopWatch.stop()
    print(f"Transpose: {stopWatch} secs")
    stopWatch.reset()
    stopWatch.start()
    from . import openMPIFileWrite, getMPIDatatype

    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size

    fh = openMPIFileWrite(filename)
    filetype = getMPIDatatype("<c8").Create_subarray(
        (Ns, Nc, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Ns, Nc),
        (Ns, Nc, Lt, Lz, Ly, Lx, Ns, Nc),
        (0, 0, gt * Lt, gz * Lz, gy * Ly, gx * Lx, 0, 0),
    )
    filetype.Commit()
    fh.Set_view(filetype=filetype)
    fh.Write_all(propagator)
    filetype.Free()
    fh.Close()
    stopWatch.stop()
    print(f"Write: {stopWatch} secs")
