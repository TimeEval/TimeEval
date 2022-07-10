from importlib.metadata import PackageNotFoundError


def _get_version() -> str:
    """Get TimeEval version number from package metadata or (if not installed) from the pyproject.toml-file."""
    package_name = __package__
    try:
        import sys

        assert package_name is not None
        if sys.version_info >= (3, 8):
            from importlib import metadata
        else:
            import importlib_metadata as metadata
        return metadata.version(package_name)
    except (AssertionError, PackageNotFoundError):
        try:
            import tomlkit
        except ModuleNotFoundError:
            return "unknown"

        from pathlib import Path

        toml_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        with toml_path.open("r") as fh:
            pyproject = tomlkit.parse(fh.read())
        return str(pyproject["tool"]["poetry"]["version"])


__version__: str = _get_version()
