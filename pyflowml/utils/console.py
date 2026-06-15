"""
Console helpers — keep PyFlowML's decorated output (box-drawing characters,
progress bars, emoji) from crashing on legacy terminals.
"""

import sys

_CONFIGURED = False


def ensure_utf8_console() -> None:
    """
    Best-effort: make ``stdout``/``stderr`` tolerate the Unicode glyphs that
    PyFlowML prints (``⏱``, ``█``, ``→``, emoji, box-drawing).

    On legacy Windows consoles (cp1252) — and whenever output is piped/redirected
    with a non-UTF-8 locale — writing those glyphs raises ``UnicodeEncodeError``
    and aborts the run. This switches the streams to UTF-8 (falling back to
    ``errors="replace"`` if that is not possible). It is idempotent and never
    raises, so it is safe to call from any entry point.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue  # e.g. captured by a test harness — nothing to do
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            try:
                reconfigure(errors="replace")
            except Exception:
                pass
