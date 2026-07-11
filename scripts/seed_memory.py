#!/usr/bin/env python3
"""Pobla la base de datos con datos de ejemplo para desarrollo."""

import asyncio
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from creative_engine.core.models import (
    DomainName,
    EvaluationScores,
    Idea,
    IdeaFeatures,
    IdeaStatus,
    ValueHypothesis,
)
from creative_engine.evolution.encoders import project_to_descriptor
from creative_engine.memory.repository import IdeaRepository

SAMPLE_IDEAS = [
    {
        "title": "Bicicleta Plegable con Paneles Solares Integrados",
        "description": (
            "Bicicleta eléctrica urbana plegable que integra paneles solares "
            "flexibles en el cuadro y las ruedas, permitiendo recarga pasiva "
            "durante el estacionamiento."
        ),
        "advantages": ["Recarga solar pasiva", "Plegable", "Cero emisiones"],
        "limitations": ["Autonomía solar limitada en días nublados", "Coste elevado"],
        "features": {
            "technologies": ["Solar Fotovoltaica Flexible", "Batería Litio-Ion"],
            "materials": ["Fibra de carbono", "Silicio amorfo"],
        },
    },
    {
        "title": "Sistema de Bicicletas Modulares Compartidas",
        "description": (
            "Plataforma de micro-movilidad donde los usuarios configuran "
            "bicicletas modulares combinando componentes estandarizados: "
            "cuadro, ruedas, motor y accesorios."
        ),
        "advantages": ["Máxima personalización", "Mantenimiento simple", "Economía circular"],
        "limitations": ["Complejidad logística", "Requiere infraestructura de intercambio"],
        "features": {
            "technologies": ["IoT"],
            "materials": ["Aluminio reciclado"],
        },
    },
    {
        "title": "Bicicleta de Carga Ultraligera con Dirección Asistida",
        "description": (
            "Bicicleta cargo eléctrica con chasis de composite ultraligero y "
            "dirección asistida eléctricamente para maniobrar con cargas "
            "pesadas en entornos urbanos."
        ),
        "advantages": ["Manejo fácil con carga", "Bajo peso", "Logística de última milla"],
        "limitations": ["Precio premium", "Mantenimiento especializado"],
        "features": {
            "technologies": ["Dirección asistida eléctrica"],
            "materials": ["Fibra de carbono", "Titanio"],
        },
    },
]


async def seed() -> None:
    repo = IdeaRepository()
    await repo.initialize()

    run_id = "seed_run_001"

    for data in SAMPLE_IDEAS:
        feat_data = data.pop("features", {})
        genome = [random.uniform(-1, 1) for _ in range(384)]
        idea = Idea(
            **data,
            value_hypothesis=ValueHypothesis(
                target_user="Commuters urbanos",
                problem_solved="Dependencia del coche para desplazamientos cortos",
                value_proposition="Movilidad eléctrica adaptable y sostenible",
            ),
            features=IdeaFeatures(**feat_data, complexity_level=random.uniform(0.4, 0.8)),
            evaluation=EvaluationScores(
                novelty=random.uniform(0.5, 0.95),
                utility=random.uniform(0.7, 0.95),
                feasibility=random.uniform(0.4, 0.8),
                impact=random.uniform(0.6, 0.9),
                market_fit=random.uniform(0.5, 0.85),
                sustainability=random.uniform(0.7, 0.95),
                scalability=random.uniform(0.4, 0.8),
            ),
            status=IdeaStatus.ELITE,
            generation=0,
            run_id=run_id,
            domain=DomainName.INDUSTRIAL_DESIGN,
        )
        idea.genome_vector = genome
        idea.behavior_descriptor = project_to_descriptor(genome, 3)

        await repo.store_idea(idea)
        print(f"  ✓ {idea.title}")

    stats = await repo.get_stats(run_id=run_id)
    print(f"\nResumen: {stats}")

    await repo.close()
    print("\n✅ Seed completado.")


if __name__ == "__main__":
    print("🌱 Poblando base de datos con ideas de ejemplo...")
    asyncio.run(seed())
