from __future__ import annotations  # TYPE_CHECKING
from typing import TYPE_CHECKING, Callable, List, Literal, NamedTuple, Tuple
from warnings import warn, filterwarnings

if TYPE_CHECKING:
    from typing import Protocol, TypeVar
    from _typeshed import SupportsFlush, SupportsWrite

    _T_contra = TypeVar("_T_contra", contravariant=True)

    class SupportsWriteAndFlush(SupportsWrite[_T_contra], SupportsFlush, Protocol[_T_contra]):
        pass


from mpi4py import MPI
from mpi4py.util import dtlib

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


def printRoot(
    *values: object,
    sep: str | None = " ",
    end: str | None = "\n",
    file: SupportsWriteAndFlush[str] | None = None,
    flush: bool = False,
):
    if _MPI_RANK == 0:
        print(*values, sep=sep, end=end, file=file, flush=flush)


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
    assert _MPI_SIZE == Gx * Gy * Gz * Gt
    _GRID_SIZE = [Gx, Gy, Gz, Gt]
    _GRID_COORD = getCoordFromRank(_MPI_RANK, _GRID_SIZE)
    printRoot(f"INFO: Using gird {_GRID_SIZE}")


def getGridSize():
    Gx, Gy, Gz, Gt = _GRID_SIZE
    assert _MPI_SIZE == Gx * Gy * Gz * Gt
    return _GRID_SIZE


def getGridCoord():
    Gx, Gy, Gz, Gt = _GRID_SIZE
    assert _MPI_SIZE == Gx * Gy * Gz * Gt
    return _GRID_COORD


def openMPIFileRead(filename: str):
    return MPI.File.Open(_MPI_COMM, filename, MPI.MODE_RDONLY)


def openMPIFileWrite(filename: str):
    return MPI.File.Open(_MPI_COMM, filename, MPI.MODE_WRONLY | MPI.MODE_CREATE)


def getMPIDatatype(dtype: str):
    return dtlib.from_numpy_dtype(dtype)


from .chroma import (
    readQIOGauge as readChromaQIOGauge,
    readQIOPropagator as readChromaQIOPropagator,
)
from .milc import (
    readGauge as readMILCGauge,
    readQIOPropagator as readMILCQIOPropagator,
)
from .kyu import (
    readGauge as readKYUGauge,
    writeGauge as writeKYUGauge,
    readPropagator as readKYUPropagator,
    writePropagator as writeKYUPropagator,
)
from .kyu_single import (
    readPropagator as readKYUPropagatorF,
    writePropagator as writeKYUPropagatorF,
)
from .io_general import (
    read as readIOGeneral,
    write as writeIOGeneral,
)
