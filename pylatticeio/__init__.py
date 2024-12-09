from typing import List

from mpi4py import MPI
import numpy

from .mpi_file import getGridCoord

_GRID_SIZE: List[int] = [1, 1, 1, 1]
_GRID_COORD: List[int] = [0, 0, 0, 0]


def setGrid(
    grid_size: List[int] = None,
):
    global _GRID_SIZE, _GRID_COORD
    Gx, Gy, Gz, Gt = grid_size if grid_size is not None else [1, 1, 1, 1]
    assert MPI.COMM_WORLD.Get_size() == Gx * Gy * Gz * Gt
    _GRID_SIZE = [Gx, Gy, Gz, Gt]
    _GRID_COORD = getGridCoord(_GRID_SIZE)
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"INFO: Using gird {_GRID_SIZE}")


def getGridSize():
    Gx, Gy, Gz, Gt = _GRID_SIZE
    assert MPI.COMM_WORLD.Get_size() == Gx * Gy * Gz * Gt
    return _GRID_SIZE


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


def readChromaQIOGauge(filename: str, checksum: bool = True):
    from .chroma import readQIOGauge as read

    latt_size, gauge_raw = read(filename, getGridSize(), checksum)
    return gauge_raw


def readILDGBinGauge(filename: str, dtype: str, latt_size: List[int]):
    from .chroma import readILDGBinGauge as read

    gauge_raw = read(filename, dtype, latt_size, getGridSize())
    return gauge_raw


def readChromaQIOPropagator(filename: str, checksum: bool = True):
    from .chroma import readQIOPropagator as read

    latt_size, staggered, propagator_raw = read(filename, getGridSize(), checksum)
    return propagator_raw


def readMILCGauge(filename: str, checksum: bool = True):
    from .milc import readGauge as read

    latt_size, gauge_raw = read(filename, getGridSize(), checksum)
    return gauge_raw


def writeMILCGauge(filename: str, latt_size: List[int], gauge: numpy.ndarray):
    from .milc import writeGauge as write

    write(filename, latt_size, getGridSize(), gauge)


def readMILCQIOPropagator(filename: str):
    from .milc import readQIOPropagator as read

    latt_size, staggered, propagator_raw = read(filename, getGridSize())
    return propagator_raw


def readKYUGauge(filename: str, latt_size: List[int]):
    from .kyu import readGauge as read

    gauge_raw = read(filename, latt_size, getGridSize())
    return gauge_raw


def writeKYUGauge(filename: str, latt_size: List[int], gauge: numpy.ndarray):
    from .kyu import writeGauge as write

    write(filename, latt_size, getGridSize(), gauge)


def readKYUPropagator(filename: str, latt_size: List[int]):
    from .kyu import readPropagator as read

    propagator_raw = read(filename, latt_size, getGridSize())
    return rotateToDeGrandRossi(propagator_raw)


def writeKYUPropagator(filename: str, latt_size: List[int], propagator: numpy.ndarray):
    from .kyu import writePropagator as write

    write(filename, latt_size, getGridSize(), rotateToDiracPauli(propagator))


def readXQCDPropagator(filename: str, latt_size: List[int], staggered: bool):
    from .xqcd import readPropagator as read

    propagator_raw = read(filename, latt_size, getGridSize(), staggered)
    if not staggered:
        return rotateToDeGrandRossi(propagator_raw)
    else:
        return propagator_raw


def writeXQCDPropagator(filename: str, latt_size: List[int], propagator: numpy.ndarray, staggered: bool):
    from .xqcd import writePropagator as write

    if not staggered:
        write(filename, latt_size, getGridSize(), rotateToDiracPauli(propagator), staggered)
    else:
        write(filename, latt_size, getGridSize(), propagator, staggered)


def readNERSCGauge(filename: str, link_trace: bool = True, checksum: bool = True):
    from .nersc import readGauge as read

    latt_size, plaquette, gauge_raw = read(filename, getGridSize(), link_trace, checksum)
    return gauge_raw


def readOpenQCDGauge(filename: str):
    from .openqcd import readGauge as read

    latt_size, plaquette, gauge_raw = read(filename, getGridSize())
    return gauge_raw


def writeOpenQCDGauge(filename: str, latt_size: List[int], gauge: numpy.ndarray, plaquette: float):
    from .openqcd import writeGauge as write

    write(filename, latt_size, getGridSize(), plaquette, gauge)


def readIOGeneral(filename: str):
    from .io_general import read

    return read(filename)


def writeIOGeneral(filename: str, head, data):
    from .io_general import write

    write(filename, head, data)


def readQIOGauge(filename: str):
    return readChromaQIOGauge(filename, getGridSize())


def readQIOPropagator(filename: str):
    return readChromaQIOPropagator(filename)


def readKYUPropagatorF(filename: str, latt_size: List[int]):
    return readXQCDPropagator(filename, latt_size, False)


def writeKYUPropagatorF(filename: str, latt_size: List[int], propagator: numpy.ndarray):
    writeXQCDPropagator(filename, latt_size, propagator, False)
