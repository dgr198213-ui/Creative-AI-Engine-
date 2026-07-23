"""Medición y liberación de memoria del proceso entre runs.

Incidente de producción (22-jul-2026): un run pequeño (población 8, 3
generaciones) hacía subir la RAM de ~94 MB a ~870 MB y se quedaba ahí
90 minutos después de terminar. La causa no es una fuga de objetos Python
—esos se recolectan solos por conteo de referencias— sino que los
asignadores nativos de C que usa `sentence-transformers`/torch (y en menor
medida glibc para el resto de la basura del run) no devuelven la memoria
liberada al sistema operativo: la retienen en arenas por si se vuelve a
pedir. `gc.collect()` limpia el lado Python; `malloc_trim(0)` es lo que de
verdad le pide a glibc que devuelva páginas libres al SO.
"""

from __future__ import annotations

import contextlib
import ctypes
import gc


def current_rss_mb() -> float | None:
    """RSS actual del proceso en MB, o None si no se puede leer (no-Linux)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


def release_memory() -> None:
    """Recolecta basura Python y pide al allocator nativo que suelte páginas libres.

    Best-effort: `malloc_trim` es una extensión de glibc. En plataformas sin
    ella (macOS, musl) simplemente no hace nada — no es crítico, solo deja
    de recuperarse esa memoria hasta que el SO decida paginarla.
    """
    gc.collect()
    with contextlib.suppress(OSError, AttributeError):
        ctypes.CDLL("libc.so.6").malloc_trim(0)
