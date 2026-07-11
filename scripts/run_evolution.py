#!/usr/bin/env python3
"""Script de conveniencia para ejecutar evoluciones rápidamente."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from creative_engine.main import cli

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "evolve",
                "--challenge",
                "Diseña una bicicleta eléctrica urbana innovadora que se "
                "diferencie de las existentes en el mercado.",
                "--domain",
                "industrial_design",
                "--population",
                "12",
                "--generations",
                "3",
            ]
        )
    cli()
