import logging
import os
import sys


def setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, (level or "INFO").upper(), logging.INFO)

    # Windows consoles often default to a non-UTF8 code page (e.g., GBK) which can crash logging
    # when questions contain non-ASCII characters. Prefer a safe UTF-8 stream.
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

    # Avoid duplicate handlers in interactive runs.
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(lvl)
        return

    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(lvl)

    # Reduce noisy libs
    logging.getLogger("httpx").setLevel(os.getenv("HTTPX_LOG_LEVEL", "WARNING"))
    logging.getLogger("openai").setLevel(os.getenv("OPENAI_LOG_LEVEL", "WARNING"))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
