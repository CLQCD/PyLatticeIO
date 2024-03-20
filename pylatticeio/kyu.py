from os import path
from typing import List

import numpy
from numpy.typing import NDArray

from ._qcd import Ns, Nc, Nd


# [Nd, Nc, Nc, 2, Lt, Lz, Ly, Lx] >f8
def readGauge(filename: str, latt_size: List[int]):
    filename = path.expanduser(path.expandvars(filename))
    with open(filename, "rb") as f:
        kyu_binary_data = f.read()

    Lx, Ly, Lz, Lt = latt_size
    return (
        numpy.frombuffer(kyu_binary_data, ">f8")
        .reshape(Nd, Nc, Nc, 2, Lt, Lz, Ly, Lx)
        .astype("<f8")
        .transpose(0, 4, 5, 6, 7, 2, 1, 3)
        .reshape(Nd, Lt, Lz, Ly, Lx, Nc, Nc * 2)
        .view("<c16")
    )


def writeGauge(filename: str, gauge_ndarray: NDArray[numpy.complex128]):
    filename = path.expanduser(path.expandvars(filename))
    latt_size = gauge_ndarray.shape[4:0:-1]

    Lx, Ly, Lz, Lt = latt_size
    kyu_binary_data = (
        gauge_ndarray.view("<f8")
        .reshape(Nd, Lt, Lz, Ly, Lx, Nc, Nc, 2)
        .transpose(0, 6, 5, 7, 1, 2, 3, 4)
        .astype(">f8")
        .tobytes()
    )
    with open(filename, "wb") as f:
        f.write(kyu_binary_data)


# [2, Ns, Nc, Lt, Lz, Ly, Lx] ">f8"
def readFermion(filename: str, latt_size: List[int]):
    filename = path.expanduser(path.expandvars(filename))
    with open(filename, "rb") as f:
        kyu_binary_data = f.read()

    Lx, Ly, Lz, Lt = latt_size
    return (
        numpy.frombuffer(kyu_binary_data, ">f8")
        .reshape(2, Ns, Nc, Lt, Lz, Ly, Lx)
        .astype("<f8")
        .transpose(3, 4, 5, 6, 1, 2, 0)
        .reshape(Lt, Lz, Ly, Lx, Ns, Nc * 2)
        .view("<c16")
    )


def writeFermion(filename: str, fermion_ndarray: NDArray):
    filename = path.expanduser(path.expandvars(filename))
    latt_size = fermion_ndarray.shape[3:-1:-1]

    Lx, Ly, Lz, Lt = latt_size
    kyu_binary_data = (
        fermion_ndarray.view("<f8")
        .reshape(Lt, Lz, Ly, Lx, Ns, Nc, 2)
        .transpose(6, 4, 5, 0, 1, 2, 3)
        .astype(">f8")
        .tobytes()
    )
    with open(filename, "wb") as f:
        f.write(kyu_binary_data)
