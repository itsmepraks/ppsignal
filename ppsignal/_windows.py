from postpyc import guvectorize
from postpyc.math import PI, cos
from postyp import Array, Float64, Int64


@guvectorize([], "(n)->(n)")
def boxcar(shape: Array[Float64], out: Array[Float64]) -> None:
    length: Int64 = len(shape)
    for i in range(length):
        out[i] = 1.0


@guvectorize([], "(n)->(n)")
def hann(shape: Array[Float64], out: Array[Float64]) -> None:
    length: Int64 = len(shape)
    if length == 1:
        out[0] = 1.0
    else:
        for i in range(length):
            out[i] = 0.5 - 0.5 * cos(2 * PI * i / (length - 1))
