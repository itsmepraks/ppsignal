# ppsignal

Signal processing kernels, in POST Python.

`ppsignal` reimplements `scipy.signal` in
[POST Python](https://github.com/openteams-ai/postpython) — every kernel is
fully-typed Python that runs under the standard CPython interpreter **and**
compiles ahead-of-time to native code (a plain C shared library and a NumPy
ufunc extension module) with the POST Python reference compiler.

Status: **Pre-alpha foundation** — no public `scipy.signal` functions yet.
The repository provides packaging, a pixi developer environment, and an
internal compiler probe with tests for its interpreted and compiled build
paths. Runtime dependencies are `numpy`,
`postpyc>=0.3.0`, and `postyp>=0.3.0`; `scipy` is not a dependency. See
[ROADMAP.md](ROADMAP.md) for targets and status. It is part of the
[PostSciPy effort](https://github.com/openteams-ai/postpython/blob/main/postscipy-roadmap.md)
to rebuild SciPy one subpackage at a time as the compiler's proving ground.

Primary compiler pressure this package generates: recurrence gufuncs, window kernels, cross-package dependency on ppspecial.

## Start here

1. Read the [POST Python spec](https://github.com/openteams-ai/postpython/blob/main/docs/spec.md)
   and the [PostSciPy roadmap](https://github.com/openteams-ai/postpython/blob/main/postscipy-roadmap.md)
   (package map, working rules, capability matrix).
2. Set up the environment with `pixi install`, then use `pixi run test`,
   `pixi run build-native`, `pixi run build-ext`, and
   `pixi run build-prefix`. The layout follows
   [ppspecial](https://github.com/openteams-ai/ppspecial), the exemplar
   package. Published sdists and wheels are pure Python source; native
   builds happen only when you run these tasks.
3. Start with a slice from "Compiles today" below; land it as a small PR with
   tests in both execution modes.

## First slices

### Compiles today

- Window functions as gufuncs `(n)->(n)` or generators: boxcar, triang, bartlett, hann, hamming, blackman, cosine; `kaiser` via ppspecial's `i0` (cross-package dependency)
- `lfilter` as a direct-form II transposed recurrence gufunc `(b),(a),(n)->(n)`
- `sosfilt` with per-section state carried in an output workspace parameter
- `unit_impulse`, simple `detrend`

### Blocked on compiler capabilities

File these as [postpython issues](https://github.com/openteams-ai/postpython/issues)
with minimal reproducers when you start on them — the filing is part of the
work and drives the compiler roadmap.

- General `convolve`/`correlate` — output length n+m-1 is a computed core dimension, which the gufunc signature algebra cannot express (NumPy shares this limit); needs a design decision or the C-ABI path
- Filter design (`butter`, `cheby1`, ...) — needs complex polynomial root finding (pplinalg/ppfft maturity)

## Working rules (summary)

- Pure POST Python: no compiler-specific escape hatches; every kernel runs
  interpreted and compiled.
- `scipy` is the reference, never a runtime dependency. Tests may use it
  optionally; prefer deterministic hardcoded reference values.
- Compiler gaps go upstream as postpython issues with reproducers, not
  silent workarounds.
- Verify against a postpython checkout on `main`.
- Document accuracy targets and reference sources per function.

The full rules and the definition of done live in the
[PostSciPy roadmap](https://github.com/openteams-ai/postpython/blob/main/postscipy-roadmap.md).
