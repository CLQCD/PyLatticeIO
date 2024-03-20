from os import path
from enum import IntEnum
import struct
from typing import NamedTuple

import numpy


class _DimensionType(IntEnum):
    dim_other = 0
    dim_x = 1
    dim_y = 2
    dim_z = 3
    dim_t = 4
    dim_d = 5
    dim_c = 6
    dim_d2 = 7
    dim_c2 = 8
    dim_complex = 9
    dim_mass = 10
    dim_smear = 11
    dim_displacement = 12

    dim_s_01 = 13
    dim_s_02 = 14
    dim_s_03 = 15
    dim_s_11 = 16
    dim_s_12 = 17
    dim_s_13 = 18
    dim_d_01 = 19
    dim_d_02 = 20
    dim_d_03 = 21
    dim_d_11 = 22
    dim_d_12 = 23
    dim_d_13 = 24

    dim_conf = 25
    dim_operator = 26
    dim_momentum = 27
    dim_direction = 28
    dim_t2 = 29
    dim_mass2 = 30

    dim_column = 31
    dim_row = 32
    dim_temporary = 33
    dim_temporary2 = 34
    dim_temporary3 = 35
    dim_temporary4 = 36

    dim_errorbar = 37
    """0 means average, 1 means errorbar, ..."""

    dim_operator2 = 38

    dim_param = 39
    dim_fitleft = 40
    dim_fitright = 41

    dim_jackknife = 42
    dim_jackknife2 = 43
    dim_jackknife3 = 44
    dim_jackknife4 = 45

    dim_summary = 46
    """
    0 means average, 1 means standard deviation, 2 means minimal value, 3 means maximum value, 4 means standard error,
    5 means median, ...
    """

    dim_channel = 47
    dim_channel2 = 48

    dim_eigen = 49

    dim_d_row = 50
    """on matrix multiplication, row is contracted with the left operand, col is contracted with the right operand."""
    dim_d_col = 51
    dim_c_row = 52
    dim_c_col = 53

    dim_parity = 54
    """dimension for different parities. we use 1/-1 for +/- parities for baryons."""

    dim_noise = 55
    dim_evenodd = 56

    dim_disp_x = 57
    dim_disp_y = 58
    dim_disp_z = 59
    dim_disp_t = 60

    dim_t3 = 61
    dim_t4 = 62
    dim_t_source = 63
    dim_t_current = 64
    dim_t_sink = 65

    dim_nothing = 66
    """do not use this unless for unused data."""

    dim_bootstrap = 67

    # add new dimensions here and add a string name in xqcd_type_dim_desc[] in io_general.c
    # ...

    dim_last = 68


class _OneDim(NamedTuple):
    type: _DimensionType
    n_indices: int
    indices: list[int]


class _FileType(NamedTuple):
    n_dimensions: int
    dimensions: list[_OneDim]

    @property
    def dimensions_type(self):
        return tuple([self.dimensions[i].type._name_ for i in range(self.n_dimensions)])

    @property
    def dimensions_n_indices(self):
        return tuple([self.dimensions[i].n_indices for i in range(self.n_dimensions)])


_header_format = "<" + "i" + ("ii" + "i" * 1024) * 16
_header_size = 4 + (8 + 4 * 1024) * 16


def read(filename: str):
    filename = path.expanduser(path.expandvars(filename))
    with open(filename, "rb") as f:
        head_raw = struct.unpack(_header_format, f.read(102400)[:_header_size])
        head = _FileType(
            head_raw[0],
            [
                _OneDim(
                    _DimensionType(head_raw[1 + dimension * 1026 + 0]),
                    head_raw[1 + dimension * 1026 + 1],
                    head_raw[1 + dimension * 1026 + 2 : 1 + dimension * 1026 + 1026],
                )
                for dimension in range(16)
            ],
        )
        binary_data = f.read()

    return numpy.frombuffer(binary_data, "<f8").reshape(head.dimensions_n_indices)
