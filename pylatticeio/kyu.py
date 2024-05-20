from os import path
from typing import List, Union

import numpy

from .field import Ns, Nc, Nd, LatticeInfo


def fromGaugeBuffer(filename: str, offset: int, dtype: str, latt_info: LatticeInfo):
    from . import readMPIFile

    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size

    gauge_raw = readMPIFile(
        filename,
        dtype,
        offset,
        (Nd, Nc, Nc, 2, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx),
        (Nd, Nc, Nc, 2, Lt, Lz, Ly, Lx),
        (0, 0, 0, 0, gt * Lt, gz * Lz, gy * Ly, gx * Lx),
    )
    gauge_raw = (
        gauge_raw.transpose(0, 4, 5, 6, 7, 2, 1, 3)
        .astype("<f8")
        .copy()
        .reshape(Nd, Lt, Lz, Ly, Lx, Nc, Nc * 2)
        .view("<c16")
    )

    return gauge_raw


def toGaugeBuffer(filename: str, offset: int, gauge_raw: numpy.ndarray, dtype: str, latt_info: LatticeInfo):
    from . import writeMPIFile

    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size

    gauge_raw = (
        gauge_raw.view("<f8")
        .reshape(Nd, Lt, Lz, Ly, Lx, Nc, Nc, 2)
        .astype(dtype)
        .transpose(0, 6, 5, 7, 1, 2, 3, 4)
        .copy()
    )
    writeMPIFile(
        filename,
        dtype,
        offset,
        (Nd, Nc, Nc, 2, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx),
        (Nd, Nc, Nc, 2, Lt, Lz, Ly, Lx),
        (0, 0, 0, 0, gt * Lt, gz * Lz, gy * Ly, gx * Lx),
        gauge_raw,
    )


def readGauge(filename: str, latt_info: Union[LatticeInfo, List[int]]):
    filename = path.expanduser(path.expandvars(filename))
    latt_info = LatticeInfo(latt_info) if not isinstance(latt_info, LatticeInfo) else latt_info
    gauge_raw = fromGaugeBuffer(filename, 0, ">f8", latt_info)

    return gauge_raw


def writeGauge(filename: str, gauge: numpy.ndarray, latt_info: Union[LatticeInfo, List[int]]):
    filename = path.expanduser(path.expandvars(filename))
    latt_info = LatticeInfo(latt_info) if not isinstance(latt_info, LatticeInfo) else latt_info

    toGaugeBuffer(filename, 0, gauge, ">f8", latt_info)


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


def fromPropagatorBuffer(filename: str, offset: int, dtype: str, latt_info: LatticeInfo):
    from . import readMPIFile

    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size

    propagator_raw = readMPIFile(
        filename,
        dtype,
        offset,
        (Ns, Nc, 2, Ns, Nc, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx),
        (Ns, Nc, 2, Ns, Nc, Lt, Lz, Ly, Lx),
        (0, 0, 0, 0, 0, gt * Lt, gz * Lz, gy * Ly, gx * Lx),
    )
    propagator_raw = (
        propagator_raw.transpose(5, 6, 7, 8, 3, 0, 4, 1, 2)
        .astype("<f8")
        .copy()
        .reshape(Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc * 2)
        .view("<c16")
    )

    return propagator_raw


def toPropagatorBuffer(filename: str, offset: int, propagator_raw: numpy.ndarray, dtype: str, latt_info: LatticeInfo):
    from . import writeMPIFile

    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size

    propagator_raw = (
        propagator_raw.view("<f8")
        .reshape(Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc, 2)
        .astype(dtype)
        .transpose(5, 7, 8, 4, 6, 0, 1, 2, 3)
        .copy()
    )
    writeMPIFile(
        filename,
        dtype,
        offset,
        (Ns, Nc, 2, Ns, Nc, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx),
        (Ns, Nc, 2, Ns, Nc, Lt, Lz, Ly, Lx),
        (0, 0, 0, 0, 0, gt * Lt, gz * Lz, gy * Ly, gx * Lx),
        propagator_raw,
    )


def readPropagator(filename: str, latt_info: Union[LatticeInfo, List[int]]):
    filename = path.expanduser(path.expandvars(filename))
    latt_info = LatticeInfo(latt_info) if not isinstance(latt_info, LatticeInfo) else latt_info
    propagator_raw = fromPropagatorBuffer(filename, 0, ">f8", latt_info)

    return rotateToDeGrandRossi(propagator_raw)


def writePropagator(filename: str, propagator: numpy.ndarray, latt_info: Union[LatticeInfo, List[int]]):
    filename = path.expanduser(path.expandvars(filename))
    latt_info = LatticeInfo(latt_info) if not isinstance(latt_info, LatticeInfo) else latt_info

    toPropagatorBuffer(filename, 0, rotateToDiracPauli(propagator), ">f8", latt_info)
