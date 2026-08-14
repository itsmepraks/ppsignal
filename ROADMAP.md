# ppsignal roadmap

## Package targets

Each public function must work in interpreted Python and in compiled native
code. Each function must have fixed reference values and boundary tests.
SciPy is a reference. SciPy is not a runtime dependency.

## Accuracy target

Window tests use fixed `float64` reference values. Compiled comparisons
use a maximum relative tolerance of `1e-14` and an absolute tolerance of
`1e-15`. Interpreted fixed-value tests use a relative tolerance of `1e-15`
and an absolute tolerance of `1e-15`.

## Completed

- Hann supports symmetric and periodic windows.
- Hann handles lengths zero, one, and two, for both symmetric and periodic
  windows.
- Hann rejects negative lengths and non-integer lengths.
- Hann has a C ABI and a NumPy extension, each with test coverage.
- Hann prefers the native kernel. When the native kernel is not
  available, Hann uses the interpreted kernel.

## Next windows

1. `boxcar`
2. `hamming`
3. `blackman`
4. `bartlett`
5. `triang`

## Deferred work

- Defer `kaiser` until the ppspecial cross-package link is ready.
- Defer convolution with a computed output length until the compiler
  supports computed output dimensions.

## Compiler gaps

If you find a compiler gap, file a postpython issue with a minimal
reproducer. Do not add a production workaround for a compiler gap.
