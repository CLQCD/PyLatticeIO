from os import path

import numpy

from .field import Ns, Nc, LatticeInfo

from .kyu import rotateToDiracPauli, rotateToDeGrandRossi


def fromPropagatorBuffer(filename: str, offset: int, dtype: str, latt_info: LatticeInfo):
    from . import readMPIFile

    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size

    propagator_raw = readMPIFile(
        filename,
        offset,
        dtype,
        (Ns, Nc, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Ns, Nc),
        (Ns, Nc, Lt, Lz, Ly, Lx, Ns, Nc),
        (0, 0, gt * Lt, gz * Lz, gy * Ly, gx * Lx, 0, 0),
    )
    propagator_raw = propagator_raw.transpose(2, 3, 4, 5, 6, 0, 7, 1).astype("<c16")

    return propagator_raw


def toPropagatorBuffer(filename: str, offset: int, propagator_raw: numpy.ndarray, dtype: str, latt_info: LatticeInfo):
    from . import writeMPIFile

    Gx, Gy, Gz, Gt = latt_info.grid_size
    gx, gy, gz, gt = latt_info.grid_coord
    Lx, Ly, Lz, Lt = latt_info.size

    propagator_raw = propagator_raw.astype(dtype).transpose(5, 7, 0, 1, 2, 3, 4, 6).copy()
    writeMPIFile(
        filename,
        offset,
        propagator_raw,
        dtype,
        (Ns, Nc, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Ns, Nc),
        (Ns, Nc, Lt, Lz, Ly, Lx, Ns, Nc),
        (0, 0, gt * Lt, gz * Lz, gy * Ly, gx * Lx, 0, 0),
    )


def readPropagator(filename: str, latt_info: LatticeInfo):
    filename = path.expanduser(path.expandvars(filename))
    propagator_raw = fromPropagatorBuffer(filename, 0, "<c8", latt_info)

    return rotateToDeGrandRossi(propagator_raw)


def writePropagator(filename: str, propagator: numpy.ndarray, latt_info: LatticeInfo):
    filename = path.expanduser(path.expandvars(filename))

    toPropagatorBuffer(filename, 0, rotateToDiracPauli(propagator), "<c8", latt_info)
