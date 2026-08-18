# ppsignal

Signal processing kernels, in POST Python.

`ppsignal` reimplements `scipy.signal` in
[POST Python](https://github.com/openteams-ai/postpython). Each kernel runs
under the standard CPython interpreter. Each kernel can also compile ahead
of time into two native formats: a plain C shared library and a NumPy
ufunc extension.

Status: **Alpha**. The package has two complete window functions, Hann and
Boxcar, with an interpreted path and a native path. It is part of the
[PostSciPy effort](https://github.com/openteams-ai/postpython/blob/main/postscipy-roadmap.md)
to rebuild SciPy one subpackage at a time, as the compiler's proving ground.

## Install

```bash
python -m pip install .
```

Installation and import do not compile native code. Unless
`ppsignal_native` is available on Python's import path, `ppsignal` uses
the interpreted window kernels.

## Use Hann

```python
from ppsignal.windows import hann

symmetric = hann(5)
periodic = hann(5, sym=False)
```

`hann(M)` returns a symmetric window, for filter design. `hann(M,
sym=False)` returns a periodic window, for spectral analysis.

## Use Boxcar

```python
from ppsignal.windows import boxcar

window = boxcar(5)
```

`boxcar(M)` returns a window of all ones.

The sym argument does not change Boxcar values.

## Develop

```bash
pixi install -e dev
pixi run -e dev test
pixi run -e dev build-dist
pixi run build-native
pixi run build-ext
pixi run build-prefix
```

`pixi run build-native` creates one plain shared library that contains
Hann and Boxcar, a C header, and an ABI manifest. `pixi run build-ext`
builds one NumPy extension that contains Hann and Boxcar and verifies
that each is a ufunc. Neither command installs the extension into the
package.

## Working rules (summary)

- Pure POST Python: every kernel runs the same source, interpreted and
  compiled. Do not add code for one compiler only.
- SciPy is the reference. SciPy is never a runtime dependency. Tests can use
  it as an optional check. Prefer fixed reference values.
- If you find a compiler limitation, report it as a postpython issue with a
  reproducer. Do not add a production workaround for a compiler limitation.
- Verify against a postpython checkout on `main`.
- Document the accuracy target and the reference source for each function.

Read the [POST Python spec](https://github.com/openteams-ai/postpython/blob/main/docs/spec.md)
and the [PostSciPy roadmap](https://github.com/openteams-ai/postpython/blob/main/postscipy-roadmap.md)
for the full rules and the definition of done.
