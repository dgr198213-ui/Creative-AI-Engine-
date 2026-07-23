"""Registro dinámico de domain packs (Fase 6, D1 opción B).

Sin red ni BD: todo se prueba con directorios temporales de packs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from creative_engine.core.domain_registry import (
    DomainPackError,
    format_domain_prompt,
    load_domain_packs,
    load_pack,
    validate_pack,
)


def _write_domain_yaml(directory: Path, name: str = "test", **extra: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    lines = [
        f"name: {name}",
        'display_name: "Test"',
        "behavior_dimensions:",
        "  - name: a",
        "    bins: 5",
        "  - name: b",
        "    bins: 5",
    ]
    for key, value in extra.items():
        lines.append(f'{key}: "{value}"')
    (directory / "domain.yaml").write_text("\n".join(lines), encoding="utf-8")


class TestFormatDomainPrompt:
    def test_resolves_known_placeholders(self) -> None:
        result = format_domain_prompt(
            "Reto: {reto}. Perfil: {perfil}. Pistas: {inspiraciones}.",
            reto="vender más",
            perfil="tienda online",
            inspiraciones="checkout",
        )
        assert result == "Reto: vender más. Perfil: tienda online. Pistas: checkout."

    def test_template_without_placeholders_returned_as_is(self) -> None:
        assert format_domain_prompt("Texto fijo sin llaves.") == "Texto fijo sin llaves."

    def test_unknown_placeholder_degrades_to_raw_template(self) -> None:
        template = "Hola {typo_desconocido}"
        # No lanza KeyError: devuelve el template sin resolver.
        assert format_domain_prompt(template, reto="x") == template


class TestLoadPack:
    def test_loads_minimal_pack(self, tmp_path: Path) -> None:
        pack_dir = tmp_path / "minimal"
        _write_domain_yaml(pack_dir, name="minimal")

        pack = load_pack(pack_dir)

        assert pack.config.name == "minimal"
        assert pack.config.display_name == "Test"
        assert pack.config.grid_shape == (5, 5)
        assert pack.examples == []
        assert pack.config.generator_prompt == ""

    def test_missing_domain_yaml_raises(self, tmp_path: Path) -> None:
        pack_dir = tmp_path / "empty"
        pack_dir.mkdir()
        with pytest.raises(DomainPackError, match=r"falta domain\.yaml"):
            load_pack(pack_dir)

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        pack_dir = tmp_path / "broken"
        pack_dir.mkdir()
        (pack_dir / "domain.yaml").write_text("name: [unclosed", encoding="utf-8")
        with pytest.raises(DomainPackError, match="YAML inválido"):
            load_pack(pack_dir)

    def test_invalid_schema_raises(self, tmp_path: Path) -> None:
        pack_dir = tmp_path / "bad_schema"
        pack_dir.mkdir()
        # Sin behavior_dimensions (obligatorio) → ValidationError de Pydantic.
        (pack_dir / "domain.yaml").write_text(
            'name: bad\ndisplay_name: "Bad"\n', encoding="utf-8"
        )
        with pytest.raises(DomainPackError, match="esquema inválido"):
            load_pack(pack_dir)

    def test_loads_own_prompts(self, tmp_path: Path) -> None:
        pack_dir = tmp_path / "withprompts"
        _write_domain_yaml(pack_dir, name="withprompts")
        prompts_dir = pack_dir / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "generator.md").write_text("Persona del generador", encoding="utf-8")
        (prompts_dir / "evaluator.md").write_text("Rúbrica del evaluador", encoding="utf-8")

        pack = load_pack(pack_dir)

        assert pack.config.generator_prompt == "Persona del generador"
        assert pack.config.evaluator_prompt == "Rúbrica del evaluador"
        assert pack.config.analyst_prompt == ""  # no lo declaró, sin base al que caer

    def test_falls_back_to_base_prompts(self, tmp_path: Path) -> None:
        base_dir = tmp_path / "base"
        _write_domain_yaml(base_dir, name="generic")
        base_prompts = base_dir / "prompts"
        base_prompts.mkdir()
        (base_prompts / "evaluator.md").write_text("Rúbrica base", encoding="utf-8")

        pack_dir = tmp_path / "child"
        _write_domain_yaml(pack_dir, name="child")
        # child no trae prompts/ en absoluto → todo hereda de base.

        pack = load_pack(pack_dir, base_dir=base_dir)

        assert pack.config.evaluator_prompt == "Rúbrica base"

    def test_own_prompt_overrides_base(self, tmp_path: Path) -> None:
        base_dir = tmp_path / "base"
        _write_domain_yaml(base_dir, name="generic")
        (base_dir / "prompts").mkdir()
        (base_dir / "prompts" / "evaluator.md").write_text("Rúbrica base", encoding="utf-8")

        pack_dir = tmp_path / "child"
        _write_domain_yaml(pack_dir, name="child")
        (pack_dir / "prompts").mkdir()
        (pack_dir / "prompts" / "evaluator.md").write_text("Rúbrica propia", encoding="utf-8")

        pack = load_pack(pack_dir, base_dir=base_dir)

        assert pack.config.evaluator_prompt == "Rúbrica propia"

    def test_loads_examples(self, tmp_path: Path) -> None:
        pack_dir = tmp_path / "withexamples"
        _write_domain_yaml(pack_dir, name="withexamples")
        (pack_dir / "examples.yaml").write_text(
            "- Primer reto de ejemplo\n- Segundo reto de ejemplo\n", encoding="utf-8"
        )

        pack = load_pack(pack_dir)

        assert pack.examples == ["Primer reto de ejemplo", "Segundo reto de ejemplo"]

    def test_invalid_examples_yaml_raises(self, tmp_path: Path) -> None:
        pack_dir = tmp_path / "badexamples"
        _write_domain_yaml(pack_dir, name="badexamples")
        (pack_dir / "examples.yaml").write_text("no_es_una_lista: true\n", encoding="utf-8")

        with pytest.raises(DomainPackError, match=r"examples\.yaml debe ser una lista"):
            load_pack(pack_dir)

    def test_loads_profile_fields(self, tmp_path: Path) -> None:
        pack_dir = tmp_path / "withprofile"
        _write_domain_yaml(pack_dir, name="withprofile")
        (pack_dir / "profile.yaml").write_text(
            "campos:\n"
            "  - nombre: tipo_artista\n"
            "    descripcion: Género o disciplina\n"
            "  - nombre: aforo_tipico\n",
            encoding="utf-8",
        )

        pack = load_pack(pack_dir)

        assert pack.config.profile_fields == [
            {"nombre": "tipo_artista", "descripcion": "Género o disciplina"},
            {"nombre": "aforo_tipico", "descripcion": ""},
        ]

    def test_invalid_profile_yaml_raises(self, tmp_path: Path) -> None:
        pack_dir = tmp_path / "badprofile"
        _write_domain_yaml(pack_dir, name="badprofile")
        (pack_dir / "profile.yaml").write_text(
            "campos:\n  - descripcion: sin nombre\n", encoding="utf-8"
        )

        with pytest.raises(DomainPackError, match="necesita 'nombre'"):
            load_pack(pack_dir)


class TestLoadDomainPacks:
    def test_missing_domains_dir_returns_empty(self, tmp_path: Path) -> None:
        assert load_domain_packs(tmp_path) == {}

    def test_loads_all_packs_in_directory(self, tmp_path: Path) -> None:
        domains_dir = tmp_path / "domains"
        _write_domain_yaml(domains_dir / "generic", name="generic")
        _write_domain_yaml(domains_dir / "marketing", name="marketing")

        packs = load_domain_packs(tmp_path)

        assert set(packs) == {"generic", "marketing"}
        assert packs["generic"].pack_name == "generic"

    def test_duplicate_declared_name_raises(self, tmp_path: Path) -> None:
        domains_dir = tmp_path / "domains"
        _write_domain_yaml(domains_dir / "pack_a", name="mismo_nombre")
        _write_domain_yaml(domains_dir / "pack_b", name="mismo_nombre")

        with pytest.raises(DomainPackError, match="duplicado"):
            load_domain_packs(tmp_path)

    def test_malformed_pack_fails_loudly_not_silently(self, tmp_path: Path) -> None:
        """Requisito explícito del diseño: nunca degradar en silencio."""
        domains_dir = tmp_path / "domains"
        _write_domain_yaml(domains_dir / "good", name="good")
        (domains_dir / "bad").mkdir(parents=True)
        (domains_dir / "bad" / "domain.yaml").write_text("name: [", encoding="utf-8")

        with pytest.raises(DomainPackError):
            load_domain_packs(tmp_path)

    def test_non_directory_entries_ignored(self, tmp_path: Path) -> None:
        domains_dir = tmp_path / "domains"
        _write_domain_yaml(domains_dir / "generic", name="generic")
        (domains_dir / "README.md").write_text("no es un pack", encoding="utf-8")

        packs = load_domain_packs(tmp_path)

        assert set(packs) == {"generic"}

    def test_base_pack_serves_as_fallback_for_others(self, tmp_path: Path) -> None:
        domains_dir = tmp_path / "domains"
        _write_domain_yaml(domains_dir / "base", name="generic")
        (domains_dir / "base" / "prompts").mkdir()
        (domains_dir / "base" / "prompts" / "analyst.md").write_text(
            "Persona base del analista", encoding="utf-8"
        )
        _write_domain_yaml(domains_dir / "tuesdi", name="tuesdi")

        packs = load_domain_packs(tmp_path)

        assert packs["tuesdi"].config.analyst_prompt == "Persona base del analista"


class TestValidatePack:
    def test_valid_pack_has_no_problems(self, tmp_path: Path) -> None:
        pack_dir = tmp_path / "ok"
        _write_domain_yaml(pack_dir, name="ok")
        assert validate_pack(pack_dir) == []

    def test_reports_schema_error(self, tmp_path: Path) -> None:
        pack_dir = tmp_path / "bad"
        pack_dir.mkdir()
        (pack_dir / "domain.yaml").write_text('name: bad\ndisplay_name: "x"\n', encoding="utf-8")

        problems = validate_pack(pack_dir)

        assert len(problems) == 1
        assert "esquema inválido" in problems[0]

    def test_reports_unknown_placeholder(self, tmp_path: Path) -> None:
        pack_dir = tmp_path / "typo"
        _write_domain_yaml(pack_dir, name="typo")
        (pack_dir / "prompts").mkdir()
        (pack_dir / "prompts" / "evaluator.md").write_text(
            "Evalúa {reto} con {criterio_secreto}", encoding="utf-8"
        )

        problems = validate_pack(pack_dir)

        assert any("desconocido" in p and "criterio_secreto" in p for p in problems)

    def test_reports_empty_profile_field_name(self, tmp_path: Path) -> None:
        pack_dir = tmp_path / "emptyfield"
        _write_domain_yaml(pack_dir, name="emptyfield")
        (pack_dir / "profile.yaml").write_text(
            'campos:\n  - nombre: ""\n', encoding="utf-8"
        )

        problems = validate_pack(pack_dir)

        assert any("nombre" in p and "vacío" in p for p in problems)
