from typing import List, Sequence

from mpi4py import MPI
from mpi4py.util import dtlib
import numpy


_MPI_COMM: MPI.Comm = MPI.COMM_WORLD
_MPI_SIZE: int = _MPI_COMM.Get_size()
_MPI_RANK: int = _MPI_COMM.Get_rank()
_GRID_SIZE: List[int] = [1, 1, 1, 1]
_GRID_COORD: List[int] = [0, 0, 0, 0]


def getRankFromCoord(coord: List[int], grid: List[int]) -> int:
    x, y, z, t = grid
    return ((coord[0] * y + coord[1]) * z + coord[2]) * t + coord[3]


def getCoordFromRank(rank: int, grid: List[int]) -> List[int]:
    x, y, z, t = grid
    return [rank // t // z // y, rank // t // z % y, rank // t % z, rank % t]


def getMPIComm():
    return _MPI_COMM


def getMPISize():
    return _MPI_SIZE


def getMPIRank():
    return _MPI_RANK


def setGrid(
    grid_size: List[int] = None,
):
    global _GRID_SIZE, _GRID_COORD
    Gx, Gy, Gz, Gt = grid_size if grid_size is not None else [1, 1, 1, 1]
    assert Gx * Gy * Gz * Gt == _MPI_SIZE
    _GRID_SIZE = [Gx, Gy, Gz, Gt]
    _GRID_COORD = getCoordFromRank(_MPI_RANK, _GRID_SIZE)
    if _MPI_RANK == 0:
        print(f"INFO: Using gird {_GRID_SIZE}")


def getGridSize():
    Gx, Gy, Gz, Gt = _GRID_SIZE
    assert Gx * Gy * Gz * Gt == _MPI_SIZE
    return _GRID_SIZE


def getGridCoord():
    Gx, Gy, Gz, Gt = _GRID_SIZE
    assert Gx * Gy * Gz * Gt == _MPI_SIZE
    return _GRID_COORD


def getSublatticeSize(latt_size: List[int]):
    Lx, Ly, Lz, Lt = latt_size
    Gx, Gy, Gz, Gt = _GRID_SIZE
    assert Lx % Gx == 0 and Ly % Gy == 0 and Lz % Gz == 0 and Lt % Gt == 0
    return [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]


def _getSubarray(shape: Sequence[int], axes: Sequence[int]):
    sizes = [d for d in shape]
    subsizes = [d for d in shape]
    starts = [d if i in axes else 0 for i, d in enumerate(shape)]
    for j, i in enumerate(axes):
        sizes[i] *= _GRID_SIZE[j]
        starts[i] *= _GRID_COORD[j]
    return sizes, subsizes, starts


def readMPIFile(
    filename: str,
    dtype: str,
    offset: int,
    shape: Sequence[int],
    axes: Sequence[int],
):
    sizes, subsizes, starts = _getSubarray(shape, axes)
    native_dtype = dtype if not dtype.startswith(">") else dtype.replace(">", "<")
    buf = numpy.empty(subsizes, native_dtype)

    fh = MPI.File.Open(_MPI_COMM, filename, MPI.MODE_RDONLY)
    filetype = dtlib.from_numpy_dtype(native_dtype).Create_subarray(sizes, subsizes, starts)
    filetype.Commit()
    fh.Set_view(disp=offset, filetype=filetype)
    fh.Read_all(buf)
    filetype.Free()
    fh.Close()

    return buf.view(dtype)


def writeMPIFile(
    filename: str,
    dtype: str,
    offset: int,
    shape: Sequence[int],
    axes: Sequence[int],
    buf: numpy.ndarray,
):
    sizes, subsizes, starts = _getSubarray(shape, axes)
    native_dtype = dtype if not dtype.startswith(">") else dtype.replace(">", "<")
    buf = buf.view(native_dtype)

    fh = MPI.File.Open(_MPI_COMM, filename, MPI.MODE_WRONLY | MPI.MODE_CREATE)
    filetype = dtlib.from_numpy_dtype(native_dtype).Create_subarray(sizes, subsizes, starts)
    filetype.Commit()
    fh.Set_view(disp=offset, filetype=filetype)
    fh.Write_all(buf)
    filetype.Free()
    fh.Close()


# matrices to convert gamma basis bewteen DeGrand-Rossi and Dirac-Pauli
# \psi(DP) = _DR_TO_DP \psi(DR)
# \psi(DR) = _DP_TO_DR \psi(DP)
_DP_TO_DR = numpy.array(
    [
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 1, 0, 1],
        [-1, 0, -1, 0],
    ]
)
_DR_TO_DP = numpy.array(
    [
        [0, -1, 0, -1],
        [1, 0, 1, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
    ]
)


def rotateToDiracPauli(propagator: numpy.ndarray):
    A = numpy.asarray(_DP_TO_DR)
    Ainv = numpy.asarray(_DR_TO_DP) / 2

    return numpy.ascontiguousarray(numpy.einsum("ij,tzyxjkab,kl->tzyxilab", Ainv, propagator, A, optimize=True))


def rotateToDeGrandRossi(propagator: numpy.ndarray):
    A = numpy.asarray(_DR_TO_DP)
    Ainv = numpy.asarray(_DP_TO_DR) / 2

    return numpy.ascontiguousarray(numpy.einsum("ij,tzyxjkab,kl->tzyxilab", Ainv, propagator, A, optimize=True))


def readChromaQIOGauge(filename: str):
    from .chroma import readQIOGauge as read

    latt_size, gauge_raw = read(filename)
    return gauge_raw


def readQIOGauge(filename: str):
    return readChromaQIOGauge(filename)


def readILDGBinGauge(filename: str, dtype: str, latt_size: List[int]):
    from .chroma import readILDGBinGauge as read

    gauge_raw = read(filename, dtype, latt_size)
    return gauge_raw


def readChromaQIOPropagator(filename: str):
    from .chroma import readQIOPropagator as read

    latt_size, staggered, propagator_raw = read(filename)
    return propagator_raw


def readQIOPropagator(filename: str):
    return readChromaQIOPropagator(filename)


def readMILCGauge(filename: str):
    from .milc import readGauge as read

    latt_size, gauge_raw = read(filename)
    return gauge_raw


def readMILCQIOPropagator(filename: str):
    from .milc import readQIOPropagator as read

    latt_size, staggered, propagator_raw = read(filename)
    return propagator_raw


def readKYUGauge(filename: str, latt_size: List[int]):
    from .kyu import readGauge as read

    gauge_raw = read(filename, latt_size)
    return gauge_raw


def writeKYUGauge(filename: str, gauge: numpy.ndarray, latt_size: List[int]):
    from .kyu import writeGauge as write

    write(filename, gauge, latt_size)


def readKYUPropagator(filename: str, latt_size: List[int]):
    from .kyu import readPropagator as read

    propagator_raw = read(filename, latt_size)
    return rotateToDeGrandRossi(propagator_raw)


def writeKYUPropagator(filename: str, propagator: numpy.ndarray, latt_size: List[int]):
    from .kyu import writePropagator as write

    write(filename, rotateToDiracPauli(propagator), latt_size)


def readXQCDPropagator(filename: str, latt_size: List[int], staggered: bool):
    from .xqcd import readPropagator as read

    propagator_raw = read(filename, latt_size, staggered)
    if not staggered:
        return rotateToDeGrandRossi(propagator_raw)
    else:
        return propagator_raw


def writeXQCDPropagator(filename: str, propagator: numpy.ndarray, latt_size: List[int], staggered: bool):
    from .xqcd import writePropagator as write

    if not staggered:
        write(filename, rotateToDiracPauli(propagator), latt_size, staggered)
    else:
        write(filename, propagator, latt_size, staggered)


def readKYUPropagatorF(filename: str, latt_size: List[int]):
    return readXQCDPropagator(filename, latt_size, False)


def writeKYUPropagatorF(filename: str, propagator: numpy.ndarray, latt_size: List[int]):
    writeXQCDPropagator(filename, propagator, latt_size, False)


from .io_general import IOGeneral


def readIOGeneral(filename: str):
    from .io_general import read

    return read(filename)


def writeIOGeneral(filename: str, head, data):
    from .io_general import write

    write(filename, head, data)
