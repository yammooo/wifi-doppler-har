"""Backward-compatible entry point for PI SHARP preprocessing."""

try:
    from .preprocess_sharp import main
except ImportError:
    from preprocess_sharp import main


if __name__ == "__main__":
    main()
