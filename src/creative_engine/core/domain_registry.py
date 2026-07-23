"""Registro dinámico de domain packs (Fase 6, decisión D1 opción B).

Un dominio ya no es un enum en código: es un directorio autocontenido
bajo `configs/domains/<nombre>/`:

    configs/domains/<nombre>/
    ├── domain.yaml       # obligatorio: identidad, pesos, mutaciones, dimensiones
    ├── prompts/
    │   ├── generator.md  # opcional; si falta, hereda del pack "base"
    │   ├── evaluator.md  # opcional; ídem
    │   └── analyst.md    # opcional; ídem
    ├── profile.yaml      # opcional: campos extra de ChallengeProfile.dominio
    ├── examples.yaml     # opcional: retos de ejemplo (panel)
    └── bench.yaml        # opcional: set de retos para `creative-engine bench`

Todo salvo `domain.yaml` es opcional y hereda del pack `base`. Añadir un
dominio es copiar un directorio — cero líneas en `src/`.

Arranque ruidoso: un pack cuyo `domain.yaml`/`profile.yaml`/`examples.yaml`
sea inválido hace FALLAR la carga (`DomainPackError`), nunca se salta en
silencio — un dominio mal configurado que "funciona" con los defaults de
otro es peor que no arrancar. Un `configs/domains/` ausente no es un
error: permite arrancar sin config (ver `config.default_generic_domain`).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .models import DomainConfig

BASE_PACK_NAME = "base"
PROMPT_ROLES = ("generator", "evaluator", "analyst")

# Únicos placeholders que un prompt de pack puede usar — ver
# `format_domain_prompt`. Cualquier otro `{nombre}` en un .md es casi
# siempre un typo, no una llave literal (por eso `domain validate` los
# detecta como error).
ALLOWED_PLACEHOLDERS = {"reto", "perfil", "inspiraciones"}
_PLACEHOLDER_RE = re.compile(r"\{(\w*)\}")


class DomainPackError(Exception):
    """Un pack de dominio está mal formado: esquema, prompts o ejemplos."""


class DomainPack:
    """Pack de dominio cargado: config resuelta + datos auxiliares del panel/Analista."""

    def __init__(
        self,
        directory: Path,
        config: DomainConfig,
        examples: list[str],
    ) -> None:
        self.directory = directory
        self.config = config
        self.examples = examples

    @property
    def pack_name(self) -> str:
        return self.directory.name

    def to_summary_dict(self) -> dict[str, Any]:
        """Forma ligera para `GET /api/v1/domains` (sin prompts completos)."""
        return {
            "name": self.config.name,
            "pack_name": self.pack_name,
            "display_name": self.config.display_name,
            "description": self.config.description,
            "examples": self.examples,
        }


def find_placeholders(text: str) -> set[str]:
    """Nombres de placeholder `{xxx}` presentes en un texto de prompt."""
    return {m for m in _PLACEHOLDER_RE.findall(text) if m}


def format_domain_prompt(
    template: str, *, reto: str = "", perfil: str = "", inspiraciones: str = ""
) -> str:
    """Resuelve los placeholders {reto}/{perfil}/{inspiraciones} de un prompt.

    Best-effort: si el template tiene una llave que no es ninguno de los
    tres placeholders conocidos (typo, o una llave literal que debería
    haberse escapado `{{...}}`), `.format()` lanzaría KeyError — en vez de
    tumbar el run por un pack mal escrito, se devuelve el template TAL
    CUAL sin resolver (degradación con elegancia; `domain validate`
    detecta esto antes de que llegue a producción).
    """
    if "{" not in template:
        return template
    try:
        return template.format(reto=reto, perfil=perfil, inspiraciones=inspiraciones)
    except (KeyError, IndexError, ValueError):
        return template


def _read_prompt(directory: Path, role: str) -> str | None:
    path = directory / "prompts" / f"{role}.md"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def _resolve_prompts(directory: Path, base_dir: Path | None) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for role in PROMPT_ROLES:
        text = _read_prompt(directory, role)
        if text is None and base_dir is not None and base_dir.resolve() != directory.resolve():
            text = _read_prompt(base_dir, role)
        resolved[role] = text or ""
    return resolved


def _load_yaml(path: Path, label: str) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise DomainPackError(f"{label}: YAML inválido: {e}") from e
    except OSError as e:
        raise DomainPackError(f"{label}: no se pudo leer: {e}") from e


def _load_examples(directory: Path) -> list[str]:
    path = directory / "examples.yaml"
    if not path.exists():
        return []
    data = _load_yaml(path, f"Pack '{directory.name}': examples.yaml") or []
    if not isinstance(data, list):
        raise DomainPackError(
            f"Pack '{directory.name}': examples.yaml debe ser una lista de retos"
        )
    return [str(x) for x in data]


def _load_profile_fields(directory: Path) -> list[dict[str, str]]:
    path = directory / "profile.yaml"
    if not path.exists():
        return []
    data = _load_yaml(path, f"Pack '{directory.name}': profile.yaml") or {}
    if not isinstance(data, dict) or not isinstance(data.get("campos", []), list):
        raise DomainPackError(
            f"Pack '{directory.name}': profile.yaml debe tener una clave "
            "'campos' con una lista de {nombre, descripcion}"
        )
    fields: list[dict[str, str]] = []
    for i, item in enumerate(data.get("campos", [])):
        if not isinstance(item, dict) or "nombre" not in item:
            raise DomainPackError(
                f"Pack '{directory.name}': profile.yaml campos[{i}] necesita 'nombre'"
            )
        fields.append(
            {"nombre": str(item["nombre"]), "descripcion": str(item.get("descripcion", ""))}
        )
    return fields


def load_pack(directory: Path, base_dir: Path | None = None) -> DomainPack:
    """Carga un único pack desde su directorio. Falla ruidosamente si está mal formado."""
    domain_yaml = directory / "domain.yaml"
    if not domain_yaml.exists():
        raise DomainPackError(f"Pack '{directory.name}': falta domain.yaml")

    raw = _load_yaml(domain_yaml, f"Pack '{directory.name}': domain.yaml")
    if not isinstance(raw, dict):
        raise DomainPackError(f"Pack '{directory.name}': domain.yaml debe ser un mapeo")

    prompts = _resolve_prompts(directory, base_dir)
    merged = {
        **raw,
        "generator_prompt": raw.get("generator_prompt") or prompts["generator"],
        "evaluator_prompt": raw.get("evaluator_prompt") or prompts["evaluator"],
        "analyst_prompt": raw.get("analyst_prompt") or prompts["analyst"],
        "profile_fields": raw.get("profile_fields") or _load_profile_fields(directory),
    }

    try:
        config = DomainConfig.model_validate(merged)
    except ValidationError as e:
        raise DomainPackError(f"Pack '{directory.name}': esquema inválido: {e}") from e

    examples = _load_examples(directory)
    return DomainPack(directory=directory, config=config, examples=examples)


def load_domain_packs(configs_dir: Path) -> dict[str, DomainPack]:
    """Escanea `configs_dir/domains/*/` y devuelve {nombre_declarado: DomainPack}.

    Un `configs/domains/` ausente devuelve `{}` sin error (arranque sin
    config). Si existe pero un pack dentro está mal formado, o dos packs
    declaran el mismo `name`, se propaga `DomainPackError` — el arranque
    debe fallar, no degradar en silencio.
    """
    domains_dir = configs_dir / "domains"
    if not domains_dir.exists():
        return {}

    base_dir = domains_dir / BASE_PACK_NAME
    base_dir = base_dir if base_dir.is_dir() else None

    packs: dict[str, DomainPack] = {}
    for entry in sorted(domains_dir.iterdir()):
        if not entry.is_dir():
            continue
        pack = load_pack(entry, base_dir)
        if pack.config.name in packs:
            raise DomainPackError(
                f"Nombre de dominio duplicado '{pack.config.name}': "
                f"packs '{packs[pack.config.name].pack_name}' y '{pack.pack_name}'"
            )
        packs[pack.config.name] = pack
    return packs


def validate_pack(directory: Path, base_dir: Path | None = None) -> list[str]:
    """Valida un pack y devuelve la lista de problemas encontrados (vacía = OK).

    A diferencia de `load_pack` (que lanza en el primer error, para el
    arranque real), esto acumula TODOS los problemas detectables para que
    `creative-engine domain validate` los reporte de una vez: esquema
    inválido, placeholders desconocidos en los prompts, y prompts que no
    se pueden formatear con los tres placeholders soportados.
    """
    problems: list[str] = []

    try:
        pack = load_pack(directory, base_dir)
    except DomainPackError as e:
        return [str(e)]

    for role in PROMPT_ROLES:
        text = _read_prompt(directory, role)
        if text is None:
            continue
        unknown = find_placeholders(text) - ALLOWED_PLACEHOLDERS
        if unknown:
            problems.append(
                f"prompts/{role}.md: placeholder(s) desconocido(s) {sorted(unknown)} "
                f"— solo se admiten {sorted(ALLOWED_PLACEHOLDERS)}"
            )
        try:
            text.format(reto="x", perfil="x", inspiraciones="x")
        except (KeyError, IndexError, ValueError) as e:
            problems.append(f"prompts/{role}.md: no se puede formatear ({e})")

    for field in pack.config.profile_fields:
        if not field.get("nombre", "").strip():
            problems.append("profile.yaml: un campo tiene 'nombre' vacío")

    return problems
