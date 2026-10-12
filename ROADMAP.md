# ppsignal roadmap

`ppsignal` rebuilds `scipy.signal` in [POST Python](https://github.com/openteams-ai/postpython):
every kernel is fully-typed Python that runs on the standard CPython
interpreter and also compiles ahead-of-time to native code. It is part of the
[PostSciPy effort](https://github.com/openteams-ai/postpython/blob/main/postscipy-roadmap.md),
which rebuilds SciPy one subpackage at a time as a proving ground for the
POST Python compiler.

## Current status

The package currently provides developer foundation only:

- a reproducible pixi developer environment and `pyproject.toml` packaging;
- a compiler probe (`ppsignal/_compiler_probe.py`) that proves one internal
  gufunc kernel, `identity`, running interpreted and building through all
  three compiled paths: a plain C shared library (`scripts/build_native.py`),
  a NumPy ufunc extension module (`scripts/build_ext.py`), and a POST Python
  prefix (`pixi run build-prefix`);
- CI provenance and diagnostics: CI records compiler, linker, SDK, package,
  and run context (`scripts/compiler_provenance.py`) and validates the
  resulting JSON against its schema before upload. Validation checks
  structure and types; fields the runner cannot determine are recorded as
  `null`, so a valid document is not proof that every field is populated.

There are **no** public `scipy.signal` functions yet. `ppsignal/__init__.py`
is empty — there is no `ppsignal.windows`, no `lfilter`, no `sosfilt`. Any
target function named below is a plan, not a claim that it exists.

## Target families

| Family | Targets | Status |
| --- | --- | --- |
| Windows | `boxcar`, `triang`, `bartlett`, `hann`, `hamming`, `blackman`, `cosine` | not started |
| Windows (cross-package) | `kaiser` (needs `i0` from `ppspecial`) | not started |
| Filters | `lfilter` (direct-form II transposed recurrence, `(b),(a),(n)->(n)`) | not started |
| Filters | `sosfilt` (per-section state in an output workspace parameter) | not started |
| Utilities | `unit_impulse`, simple `detrend` | not started |
| Blocked | `convolve` / `correlate` — output length `n+m-1` is a computed core dimension the gufunc signature algebra can't express today | blocked on compiler capability |
| Blocked | Filter design (`butter`, `cheby1`, ...) — needs complex polynomial root finding (`pplinalg` / `ppfft` maturity) | blocked on dependency |

Windows are the recommended starting point; filters and utilities follow.

## Accuracy and reference policy

- `scipy` is the reference for correctness, never a runtime dependency of
  `ppsignal`. Tests may compare against `scipy` optionally, but prefer
  deterministic hardcoded reference values so the test suite does not
  require `scipy` to be installed.
- Each function's accuracy target and reference source must be documented
  next to its implementation and tests.
- Implementations are verified against a `postpython` checkout on `main`,
  not against a pinned release.

## Definition of done

A target function is done only when all of the following hold:

1. **Interpreted**: runs correctly under the standard CPython interpreter,
   with no compiler-specific escape hatches.
2. **Plain C**: compiles to a plain C shared library via the postpyc
   toolchain (see `scripts/build_native.py` for the pattern proven by the
   compiler probe).
3. **NumPy extension**: compiles to a NumPy ufunc CPython extension module
   (see `scripts/build_ext.py` for the pattern), and the compiled ufunc
   produces results that match the interpreted path.
4. **Source-only distribution**: the sdist and wheel stay pure Python
   source — no `.so` / `.pyd` / `.dll` / `.dylib` binaries are bundled, and
   building them does not compile anything. Developers and consumers who
   want native code run the pixi `build-native` / `build-ext` /
   `build-prefix` tasks explicitly.
5. Tests cover both the interpreted and compiled execution modes, and pass
   in CI.

## Compiler and upstream requests

Compiler gaps go upstream as `postpython` issues with minimal reproducers —
filing the issue is part of the work and drives the compiler roadmap. Do not
work around a gap silently.

Relevant upstream requests for this package:

- [postpython #12](https://github.com/openteams-ai/postpython/issues/12):
  stable package C ABI (export manifest, headers, symbol naming) — closed
  as completed
- [postpython #13](https://github.com/openteams-ai/postpython/issues/13):
  cross-module optimization and inlining — open
- [postpython #14](https://github.com/openteams-ai/postpython/issues/14):
  distribution story for compiled outputs (no binary wheels) — open
- [postpython #15](https://github.com/openteams-ai/postpython/issues/15):
  floating-point semantics (NaN/inf/signed zero, strict vs fast-math) — open

When a new gap is found while implementing a target above, file a new
`postpython` issue with a minimal reproducer before working around it.

## References

- [POST Python spec](https://github.com/openteams-ai/postpython/blob/main/docs/spec.md)
- [PostSciPy roadmap](https://github.com/openteams-ai/postpython/blob/main/postscipy-roadmap.md)
- [ppspecial](https://github.com/openteams-ai/ppspecial) — exemplar package
  for layout and working rules
