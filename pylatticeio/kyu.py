from os import path
from typing import List

import numpy

from .field import Ns, Nc, Nd, LatticeInfo


def fromGaugeBuffer(filename: str, offset: int, dtype: str, latt_info: LatticeInfo):
    from . import openMPIFileRead, getMPIDatatype

    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size
    native_dtype = dtype if not dtype.startswith(">") else dtype.replace(">", "<")

    fh = openMPIFileRead(filename)
    gauge_raw = numpy.empty((Nd, Nc, Nc, 2, Lt, Lz, Ly, Lx), native_dtype)
    filetype = getMPIDatatype(native_dtype).Create_subarray(
        (Nd, Nc, Nc, 2, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx),
        (Nd, Nc, Nc, 2, Lt, Lz, Ly, Lx),
        (0, 0, 0, 0, gt * Lt, gz * Lz, gy * Ly, gx * Lx),
    )
    filetype.Commit()
    fh.Set_view(offset, filetype=filetype)
    fh.Read_all(gauge_raw)
    filetype.Free()
    fh.Close()

    gauge_raw = (
        gauge_raw.transpose(0, 4, 5, 6, 7, 2, 1, 3)
        .view(dtype)
        .astype("<f8")
        .copy()
        .reshape(Nd, Lt, Lz, Ly, Lx, Nc, Nc * 2)
        .view("<c16")
    )

    return gauge_raw


def toGaugeBuffer(filename: str, offset: int, gauge_raw: numpy.ndarray, dtype: str, latt_info: LatticeInfo):
    from . import openMPIFileWrite, getMPIDatatype

    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size
    native_dtype = dtype if not dtype.startswith(">") else dtype.replace(">", "<")

    fh = openMPIFileWrite(filename)
    gauge_raw = (
        gauge_raw.view("<f8")
        .reshape(Nd, Lt, Lz, Ly, Lx, Nc, Nc, 2)
        .astype(dtype)
        .view(native_dtype)
        .transpose(0, 6, 5, 7, 1, 2, 3, 4)
        .copy()
    )
    filetype = getMPIDatatype(native_dtype).Create_subarray(
        (Nd, Nc, Nc, 2, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx),
        (Nd, Nc, Nc, 2, Lt, Lz, Ly, Lx),
        (0, 0, 0, 0, gt * Lt, gz * Lz, gy * Ly, gx * Lx),
    )
    filetype.Commit()
    fh.Set_view(offset, filetype=filetype)
    fh.Write_all(gauge_raw)
    filetype.Free()
    fh.Close()


def readGauge(filename: str, latt_size: List[int]):
    filename = path.expanduser(path.expandvars(filename))
    latt_info = LatticeInfo(latt_size)

    return fromGaugeBuffer(filename, 0, ">f8", latt_info)


def writeGauge(filename: str, gauge: numpy.ndarray):
    from . import getGridSize

    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = gauge.shape[1:5][::-1]
    Gx, Gy, Gz, Gt = getGridSize()
    latt_info = LatticeInfo([Gx * Lx, Gy * Ly, Gz * Lz, Gt * Lt])

    toGaugeBuffer(filename, 0, gauge, ">f8", latt_info)


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
    propagator_raw = numpy.empty((Ns, Nc, 2, Ns, Nc, Lt, Lz, Ly, Lx), native_dtype)
    filetype = getMPIDatatype(native_dtype).Create_subarray(
        (Ns, Nc, 2, Ns, Nc, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx),
        (Ns, Nc, 2, Ns, Nc, Lt, Lz, Ly, Lx),
        (0, 0, 0, 0, 0, gt * Lt, gz * Lz, gy * Ly, gx * Lx),
    )
    filetype.Commit()
    fh.Set_view(offset, filetype=filetype)
    fh.Read_all(propagator_raw)
    filetype.Free()
    fh.Close()

    return (
        propagator_raw.transpose(5, 6, 7, 8, 3, 0, 4, 1, 2)
        .view(dtype)
        .astype("<f8")
        .copy()
        .reshape(Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc * 2)
        .view("<c16")
    )


def toPropagatorBuffer(filename: str, offset: int, propagator_raw: numpy.ndarray, dtype: str, latt_info: LatticeInfo):
    from . import openMPIFileWrite, getMPIDatatype

    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size
    native_dtype = dtype if not dtype.startswith(">") else dtype.replace(">", "<")

    fh = openMPIFileWrite(filename)
    propagator_raw = (
        propagator_raw.view("<f8")
        .reshape(Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc, 2)
        .astype(dtype)
        .view(native_dtype)
        .transpose(5, 7, 8, 4, 6, 0, 1, 2, 3)
        .copy()
    )
    filetype = getMPIDatatype(native_dtype).Create_subarray(
        (Ns, Nc, 2, Ns, Nc, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx),
        (Ns, Nc, 2, Ns, Nc, Lt, Lz, Ly, Lx),
        (0, 0, 0, 0, 0, gt * Lt, gz * Lz, gy * Ly, gx * Lx),
    )
    filetype.Commit()
    fh.Set_view(offset, filetype=filetype)
    fh.Write_all(propagator_raw)
    filetype.Free()
    fh.Close()


def readPropagator(filename: str, latt_size: List[int]):
    filename = path.expanduser(path.expandvars(filename))
    latt_info = LatticeInfo(latt_size)

    return rotateToDeGrandRossi(fromPropagatorBuffer(filename, 0, ">f8", latt_info))


def writePropagator(filename: str, propagator: numpy.ndarray):
    from . import getGridSize

    filename = path.expanduser(path.expandvars(filename))
    Lx, Ly, Lz, Lt = propagator.shape[0:4][::-1]
    Gx, Gy, Gz, Gt = getGridSize()
    latt_info = LatticeInfo([Gx * Lx, Gy * Ly, Gz * Lz, Gt * Lt])

    toPropagatorBuffer(filename, 0, rotateToDiracPauli(propagator), ">f8", latt_info)
