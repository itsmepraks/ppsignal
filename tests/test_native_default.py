import subprocess
import sys
import textwrap


def test_public_wrapper_prefers_available_native_hann():
    script = textwrap.dedent(
        """
        import sys
        import types

        native = types.ModuleType("ppsignal_native")
        native.hann = lambda values: "native-hann"
        sys.modules["ppsignal_native"] = native

        import ppsignal.windows as windows

        assert windows._hann is native.hann
        assert windows.hann(3) == "native-hann"
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_public_wrapper_falls_back_to_interpreted_hann_when_native_module_is_absent():
    script = textwrap.dedent(
        """
        import builtins

        real_import = builtins.__import__

        def without_native(name, *args, **kwargs):
            if name == "ppsignal_native":
                raise ModuleNotFoundError(
                    "No module named 'ppsignal_native'", name="ppsignal_native"
                )
            return real_import(name, *args, **kwargs)

        builtins.__import__ = without_native

        import ppsignal._windows as interpreted
        import ppsignal.windows as windows

        assert windows._hann is interpreted.hann
        assert windows.hann(3).tolist() == [0.0, 1.0, 0.0]
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_public_wrapper_reraises_missing_dependency_from_native_module():
    script = textwrap.dedent(
        """
        import builtins

        import pytest

        real_import = builtins.__import__

        def missing_native_dependency(name, *args, **kwargs):
            if name == "ppsignal_native":
                raise ModuleNotFoundError(
                    "No module named 'native_dependency'", name="native_dependency"
                )
            return real_import(name, *args, **kwargs)

        builtins.__import__ = missing_native_dependency

        with pytest.raises(ModuleNotFoundError) as excinfo:
            import ppsignal.windows

        assert excinfo.value.name == "native_dependency"
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)
