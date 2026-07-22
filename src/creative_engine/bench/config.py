"""Configuración de un set de retos para el benchmark de 3 brazos."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class BenchChallenge(BaseModel):
    """Un reto del set: vago (empresario no técnico) o control (bien formulado)."""

    texto: str = Field(..., min_length=10)
    tipo: Literal["vago", "control"]


class BenchSetConfig(BaseModel):
    """Set completo de retos + parámetros de ejecución del benchmark."""

    name: str
    domain: str = "generic"
    retos: list[BenchChallenge] = Field(min_length=1)
    repeticiones: int = Field(default=3, ge=1, le=20)
    poblacion_motor: int = Field(default=8, ge=4)
    generaciones_motor: int = Field(default=3, ge=1)
    ideas_por_brazo: int = Field(default=8, ge=3)

    @classmethod
    def from_yaml(cls, path: str | Path) -> BenchSetConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)
