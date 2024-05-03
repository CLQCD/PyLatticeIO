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


def readPropagator(filename: str, latt_size: List[int]):
    filename = path.expanduser(path.expandvars(filename))
    with open(filename, "rb") as f:
        kyu_binary_data = f.read()

    Lx, Ly, Lz, Lt = latt_size
    kyu_data = (
        numpy.frombuffer(kyu_binary_data, ">f8")
        .reshape(Ns, Nc, 2, Ns, Nc, Lt, Lz, Ly, Lx)
        .astype("<f8")
        .transpose(5, 6, 7, 8, 3, 0, 4, 1, 2)
        .reshape(Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc * 2)
        .view("<c16")
    )
    data = numpy.zeros_like(kyu_data)
    data[:, :, :, :, 0] = -(2**-0.5) * kyu_data[:, :, :, :, 1] - 2**-0.5 * kyu_data[:, :, :, :, 3]
    data[:, :, :, :, 1] = +(2**-0.5) * kyu_data[:, :, :, :, 2] + 2**-0.5 * kyu_data[:, :, :, :, 0]
    data[:, :, :, :, 2] = +(2**-0.5) * kyu_data[:, :, :, :, 3] - 2**-0.5 * kyu_data[:, :, :, :, 1]
    data[:, :, :, :, 3] = +(2**-0.5) * kyu_data[:, :, :, :, 0] - 2**-0.5 * kyu_data[:, :, :, :, 2]
    return data


def writePropagator(filename: str, data: NDArray[numpy.complex128]):
    filename = path.expanduser(path.expandvars(filename))
    latt_size = data.shape[3:-1:-1]

    Lx, Ly, Lz, Lt = latt_size
    kyu_data = numpy.zeros_like(data)
    kyu_data[:, :, :, :, 0] = +(2**-0.5) * data[:, :, :, :, 1] + 2**-0.5 * data[:, :, :, :, 3]
    kyu_data[:, :, :, :, 1] = -(2**-0.5) * data[:, :, :, :, 2] - 2**-0.5 * data[:, :, :, :, 0]
    kyu_data[:, :, :, :, 2] = -(2**-0.5) * data[:, :, :, :, 3] + 2**-0.5 * data[:, :, :, :, 1]
    kyu_data[:, :, :, :, 3] = -(2**-0.5) * data[:, :, :, :, 0] + 2**-0.5 * data[:, :, :, :, 2]
    kyu_binary_data = (
        kyu_data.view("<f8")
        .reshape(Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc, 2)
        .transpose(5, 7, 8, 4, 6, 0, 1, 2, 3)
        .astype(">f8")
        .tobytes()
    )
    with open(filename, "wb") as f:
        f.write(kyu_binary_data)
