"""Runtime package for the refactored real-time audio pipeline."""


def main():
    """Import the heavy runtime entrypoint only when it is actually needed."""
    from .main import main as runtime_main

    return runtime_main()


__all__ = ["main"]
