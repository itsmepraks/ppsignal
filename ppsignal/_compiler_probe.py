"""Internal kernel that proves interpreted and compiled execution."""

from postpyc import guvectorize
from postyp import Array, Float64, Int64


@guvectorize([], "(n)->(n)")
def identity(values: Array[Float64], out: Array[Float64]) -> None:
    length: Int64 = len(values)
    for i in range(length):
        out[i] = values[i]
