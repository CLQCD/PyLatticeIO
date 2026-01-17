# from pyquda_comm import (  # noqa: F401
#     initGrid,
#     initDevice,
#     getMPIComm,
#     getMPIRank,
#     getGridCoord,
#     getNeighbourRank,
#     getSublatticeSize,
#     openReadHeader,
#     openWriteHeader,
#     readMPIFile,
#     writeMPIFile,
# )

from contextlib import contextmanager
import logging
import sys
from typing import IO, List, Optional, Sequence, Tuple, Type

import numpy
from numpy.typing import DTypeLike, NDArray
from mpi4py import MPI
from mpi4py.util import dtlib


class _MPILogger:
    def __init__(self, root: int = 0) -> None:
        self.root = root
        formatter = logging.Formatter(fmt="{name} {levelname}: {message}", style="{")
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)
        stderr_handler = logging.StreamHandler()
        stderr_handler.setFormatter(formatter)
        stderr_handler.setLevel(logging.WARNING)
        self.logger = logging.getLogger("PyQUDA")
        self.logger.level = logging.INFO
        self.logger.handlers = [stdout_handler, stderr_handler]

    def debug(self, msg: str):
        if _MPI_RANK == self.root:
            self.logger.debug(msg)

    def info(self, msg: str):
        if _MPI_RANK == self.root:
            self.logger.info(msg)

    def warning(self, msg: str, category: Type[Warning]):
        if _MPI_RANK == self.root:
            self.logger.warning(msg, exc_info=category(msg))

    def error(self, msg: str, category: Type[Exception]):
        if _MPI_RANK == self.root:
            self.logger.error(msg, exc_info=category(msg), stack_info=True)

    def critical(self, msg: str, category: Type[Exception]):
        if _MPI_RANK == self.root:
            self.logger.critical(msg, exc_info=category(msg), stack_info=True)
        raise category(msg)


_MPI_COMM = MPI.COMM_WORLD
_MPI_SIZE = _MPI_COMM.Get_size()
_MPI_RANK = _MPI_COMM.Get_rank()
_MPI_IO_MAX_COUNT: int = 2**30
_MPI_LOGGER = _MPILogger()
_GRID_SIZE: Optional[Tuple[int, ...]] = None
_GRID_COORD: Optional[Tuple[int, ...]] = None


def _defaultRankFromCoord(coords: Sequence[int], dims: Sequence[int]) -> int:
    rank = 0
    for coord, dim in zip(coords, dims):
        rank = rank * dim + coord
    return rank


def _defaultCoordFromRank(rank: int, dims: Sequence[int]) -> List[int]:
    coords = []
    for dim in dims[::-1]:
        coords.append(rank % dim)
        rank = rank // dim
    return coords[::-1]


def getRankFromCoord(grid_coord: List[int]) -> int:
    grid_size = getGridSize()
    if len(grid_coord) != len(grid_size):
        _MPI_LOGGER.critical(
            f"Grid coordinate {grid_coord} and grid size {grid_size} must have the same dimension",
            ValueError,
        )

    return _defaultRankFromCoord(grid_coord, grid_size)


def getCoordFromRank(mpi_rank: int) -> List[int]:
    grid_size = getGridSize()

    return _defaultCoordFromRank(mpi_rank, grid_size)


def getNeighbourRank():
    grid_size = getGridSize()
    grid_coord = getGridCoord()

    neighbour_forward = []
    neighbour_backward = []
    for d in range(len(grid_size)):
        g, G = grid_coord[d], grid_size[d]
        grid_coord[d] = (g + 1) % G
        neighbour_forward.append(getRankFromCoord(grid_coord))
        grid_coord[d] = (g - 1) % G
        neighbour_backward.append(getRankFromCoord(grid_coord))
        grid_coord[d] = g
    return neighbour_forward + neighbour_backward


def getSublatticeSize(latt_size: Sequence[int], force_even: bool = True):
    grid_size = getGridSize()
    if len(latt_size) != len(grid_size):
        _MPI_LOGGER.critical(
            f"Lattice size {latt_size} and grid size {grid_size} must have the same dimension",
            ValueError,
        )
    if force_even:
        if not all([(GL % (2 * G) == 0 or GL * G == 1) for GL, G in zip(latt_size, grid_size)]):
            _MPI_LOGGER.critical(
                f"lattice size {latt_size} must be divisible by gird size {grid_size}, "
                "and sublattice size must be even in all directions for consistant even-odd preconditioning, "
                "otherwise lattice size and grid size for this direction must be 1",
                ValueError,
            )
    else:
        if not all([(GL % G == 0) for GL, G in zip(latt_size, grid_size)]):
            _MPI_LOGGER.critical(
                f"lattice size {latt_size} must be divisible by gird size {grid_size}",
                ValueError,
            )
    return [GL // G for GL, G in zip(latt_size, grid_size)]


def initGrid(grid_size: Optional[Sequence[int]] = None):
    global _GRID_SIZE, _GRID_COORD
    if _GRID_SIZE is None:
        if grid_size is None:
            grid_size = [1, 1, 1, 1]

        _GRID_SIZE = tuple(grid_size)
        _GRID_COORD = tuple(getCoordFromRank(_MPI_RANK))
        _MPI_LOGGER.info(f"Using grid size {_GRID_SIZE}")
    else:
        _MPI_LOGGER.warning("Grid is already initialized", RuntimeWarning)


def getMPIComm():
    return _MPI_COMM


def getMPISize():
    return _MPI_SIZE


def getMPIRank():
    return _MPI_RANK


def getGridSize():
    if _GRID_SIZE is None:
        initGrid()
    assert _GRID_SIZE is not None
    return list(_GRID_SIZE)


def getGridCoord():
    if _GRID_COORD is None:
        initGrid()
    assert _GRID_COORD is not None
    return list(_GRID_COORD)


class _FileWithOffset:
    def __init__(self, fp: Optional[IO]):
        self.fp = fp
        self.offset: int = -1


@contextmanager
def openReadHeader(filename: str):
    fp = None
    try:
        fp = open(filename, "rb")
    except Exception as e:
        _MPI_LOGGER.critical(str(e), type(e))
    try:
        f = _FileWithOffset(fp)
        yield f
    except Exception as e:
        _MPI_LOGGER.critical(str(e), type(e))
    finally:
        f.offset = fp.tell()
        fp.close()


@contextmanager
def openWriteHeader(filename: str, root: int = 0):
    fp, e_ = None, None
    if _MPI_RANK == root:
        try:
            fp = open(filename, "wb")
        except Exception as e:
            e_ = e
    e_ = _MPI_COMM.bcast(e_, root)
    if e_ is not None:
        _MPI_LOGGER.critical(str(e_), type(e_))
    try:
        f = _FileWithOffset(fp)
        yield f
    except Exception as e:
        e_ = e
    finally:
        e_ = _MPI_COMM.bcast(e_, root)
        if e_ is not None:
            _MPI_LOGGER.critical(str(e_), type(e_))
        if fp is not None:
            f.offset = fp.tell()
            fp.close()
        f.offset = _MPI_COMM.bcast(f.offset, root)


def _getSubarray(dtype: DTypeLike, shape: Sequence[int], axes: Sequence[int]):
    sizes = [d for d in shape]
    subsizes = [d for d in shape]
    starts = [d if i in axes else 0 for i, d in enumerate(shape)]
    grid = getGridSize()
    coord = getGridCoord()
    for j, i in enumerate(axes):
        sizes[i] *= grid[j]
        starts[i] *= coord[j]

    dtype_str = numpy.dtype(dtype).str
    native_dtype_str = dtype_str if not dtype_str.startswith(">") else dtype_str.replace(">", "<")
    return native_dtype_str, dtlib.from_numpy_dtype(native_dtype_str).Create_subarray(sizes, subsizes, starts)


def readMPIFile(filename: str, dtype: DTypeLike, offset: int, shape: Sequence[int], axes: Sequence[int]) -> NDArray:
    native_dtype_str, filetype = _getSubarray(dtype, shape, axes)
    buf = numpy.empty(shape, native_dtype_str)
    buf_flat = buf.reshape(-1)

    fh = MPI.File.Open(_MPI_COMM, filename, MPI.MODE_RDONLY)
    filetype.Commit()
    fh.Set_view(disp=offset, filetype=filetype)
    for start in range(0, buf.size, _MPI_IO_MAX_COUNT):
        fh.Read_all(buf_flat[start : start + _MPI_IO_MAX_COUNT])
    filetype.Free()
    fh.Close()

    return buf.view(dtype)


def writeMPIFile(filename: str, dtype: DTypeLike, offset: int, shape: Sequence[int], axes: Sequence[int], buf: NDArray):
    native_dtype_str, filetype = _getSubarray(dtype, shape, axes)
    buf = buf.view(native_dtype_str)
    buf_flat = buf.reshape(-1)

    fh = MPI.File.Open(_MPI_COMM, filename, MPI.MODE_WRONLY | MPI.MODE_CREATE)
    filetype.Commit()
    fh.Set_view(disp=offset, filetype=filetype)
    for start in range(0, buf.size, _MPI_IO_MAX_COUNT):
        fh.Write_all(buf_flat[start : start + _MPI_IO_MAX_COUNT])
    filetype.Free()
    fh.Close()
