from typing import List, Optional

from numpy.typing import NDArray

from ._mpi_file import initGrid


def init(grid_size: Optional[List[int]] = None):
    initGrid(grid_size)


def readChromaQIOGauge(filename: str, checksum: bool = True):
    from .chroma import readQIOGauge as read

    latt_size, gauge_raw = read(filename, checksum)
    return gauge_raw


def readILDGBinGauge(filename: str, dtype: str, latt_size: List[int]):
    from .ildg import readBinGauge as read

    gauge_raw = read(filename, dtype, latt_size)
    return gauge_raw


def readChromaQIOPropagator(filename: str, checksum: bool = True):
    from .chroma import readQIOPropagator as read

    latt_size, propagator_raw = read(filename, checksum)
    return propagator_raw


def readChromaQIOStaggeredPropagator(filename: str, checksum: bool = True):
    from .chroma import readQIOStaggeredPropagator as read

    latt_size, propagator_raw = read(filename, checksum)
    return propagator_raw


def readMILCGauge(filename: str, checksum: bool = True):
    from .milc import readGauge as read

    latt_size, gauge_raw = read(filename, checksum)
    return gauge_raw


def writeMILCGauge(filename: str, latt_size: List[int], gauge: NDArray):
    from .milc import writeGauge as write

    write(filename, latt_size, gauge)


def readMILCQIOPropagator(filename: str):
    from .milc import readQIOPropagator as read

    latt_size, propagator_raw = read(filename)
    return propagator_raw


def readMILCQIOStaggeredPropagator(filename: str):
    from .milc import readQIOStaggeredPropagator as read

    latt_size, propagator_raw = read(filename)
    return propagator_raw


def readKYUGauge(filename: str, latt_size: List[int]):
    from .kyu import readGauge as read

    gauge_raw = read(filename, latt_size)
    return gauge_raw


def writeKYUGauge(filename: str, latt_size: List[int], gauge: NDArray):
    from .kyu import writeGauge as write

    write(filename, latt_size, gauge)


def readKYUPropagator(filename: str, latt_size: List[int]):
    from .kyu import readPropagator as read

    propagator_raw = read(filename, latt_size)
    return propagator_raw


def writeKYUPropagator(filename: str, latt_size: List[int], propagator: NDArray):
    from .kyu import writePropagator as write

    write(filename, latt_size, propagator)


def readXQCDPropagator(filename: str, latt_size: List[int]):
    from .xqcd import readPropagator as read

    propagator_raw = read(filename, latt_size)
    return propagator_raw


def writeXQCDPropagator(filename: str, latt_size: List[int], propagator: NDArray):
    from .xqcd import writePropagator as write

    write(filename, latt_size, propagator)


def readXQCDStaggeredPropagator(filename: str, latt_size: List[int]):
    from .xqcd import readStaggeredPropagator as read

    propagator_raw = read(filename, latt_size)
    return propagator_raw


def writeXQCDStaggeredPropagator(filename: str, latt_size: List[int], propagator: NDArray):
    from .xqcd import writeStaggeredPropagator as write

    write(filename, latt_size, propagator)


def readNERSCGauge(filename: str, link_trace: bool = True, checksum: bool = True):
    from .nersc import readGauge as read

    latt_size, gauge_raw = read(filename, link_trace, checksum)
    return gauge_raw


def readOpenQCDGauge(filename: str):
    from .openqcd import readGauge as read

    latt_size, gauge_raw = read(filename)
    return gauge_raw


def writeOpenQCDGauge(filename: str, latt_size: List[int], gauge: NDArray):
    from .openqcd import writeGauge as write

    write(filename, latt_size, gauge)


def readIOGeneral(filename: str):
    from .io_general import read

    return read(filename)


def writeIOGeneral(filename: str, head, data):
    from .io_general import write

    write(filename, head, data)


def readQIOGauge(filename: str):
    return readChromaQIOGauge(filename)


def readQIOPropagator(filename: str):
    return readChromaQIOPropagator(filename)


def readQIOStaggeredPropagator(filename: str):
    return readChromaQIOStaggeredPropagator(filename)


def readKYUPropagatorF(filename: str, latt_size: List[int]):
    return readXQCDPropagator(filename, latt_size)


def writeKYUPropagatorF(filename: str, latt_size: List[int], propagator: NDArray):
    writeXQCDPropagator(filename, latt_size, propagator)
