"""Punto de entrada principal del Creative AI Engine (CLI)."""

from __future__ import annotations

import asyncio
import logging

import click
import structlog
from rich.console import Console
from rich.table import Table

from .core.config import get_settings
from .core.models import DomainName, EvolutionRequest, EvolutionState

console = Console()


def _setup_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
    )


@click.group()
@click.option("--debug", is_flag=True, help="Activar modo debug")
def cli(debug: bool) -> None:
    """Creative AI Engine — Motor de creatividad computacional."""
    _setup_logging("DEBUG" if debug else get_settings().log_level)


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host de escucha")
@click.option("--port", default=8000, type=int, help="Puerto de escucha")
def serve(host: str, port: int) -> None:
    """Inicia el servidor API."""
    import uvicorn

    console.print("[bold green]🚀 Creative AI Engine — Servidor API[/bold green]")
    console.print(f"   Host: {host}:{port}")
    console.print(f"   Docs: http://{host}:{port}/docs")

    uvicorn.run(
        "creative_engine.api.app:app",
        host=host,
        port=port,
        reload=get_settings().debug,
    )


@cli.command()
@click.option("--challenge", required=True, help="Reto creativo")
@click.option(
    "--domain",
    default="generic",
    type=click.Choice([d.value for d in DomainName]),
)
@click.option("--population", default=20, type=int, help="Tamaño de población")
@click.option("--generations", default=5, type=int, help="Número de generaciones")
@click.option("--no-db", is_flag=True, help="No persistir en PostgreSQL (solo memoria)")
def evolve(challenge: str, domain: str, population: int, generations: int, no_db: bool) -> None:
    """Ejecuta una evolución desde CLI."""

    async def _run() -> None:
        from .agents.evaluator_orchestrator import EvaluatorOrchestrator
        from .agents.feasibility import FeasibilityAgent
        from .agents.generator import IdeaGeneratorAgent
        from .agents.innovation import InnovationAgent
        from .agents.market import MarketAgent
        from .evolution.crossover import CrossoverEngine
        from .evolution.encoders import IdeaEncoder
        from .evolution.mutation import MutationEngine
        from .evolution.qd_engine import QDEngine
        from .llm.factory import build_router, role_llms
        from .memory.repository import IdeaRepository

        settings = get_settings()

        if not settings.llm:
            console.print("[red]Error: no hay proveedores LLM configurados.[/red]")
            console.print("Define variables CREATIVE_LLM__* en tu .env (ver .env.example)")
            return

        first_config = next(iter(settings.llm.values()))
        router = build_router(settings)
        roles = role_llms(router)
        gen_llm = roles["generator"]
        eval_llm = roles["evaluator"]

        console.print("\n[bold cyan]🧬 Creative AI Engine — Evolución[/bold cyan]")
        console.print(f"   Reto: {challenge[:80]}")
        console.print(f"   Dominio: {domain}")
        console.print(f"   Población: {population} x {generations} generaciones\n")

        repo: IdeaRepository | None = None
        if not no_db:
            repo = IdeaRepository()
            try:
                await repo.initialize()
            except Exception as e:
                console.print(f"[yellow]PostgreSQL no disponible ({e}); modo sin persistencia.[/yellow]")
                repo = None

        try:
            evaluator = EvaluatorOrchestrator(
                agents={
                    "innovation": InnovationAgent(eval_llm),
                    "feasibility": FeasibilityAgent(eval_llm),
                    "market": MarketAgent(eval_llm),
                }
            )

            engine = QDEngine(
                generator=IdeaGeneratorAgent(gen_llm),
                evaluator=evaluator,
                mutation=MutationEngine(gen_llm, max_concurrent=first_config.max_concurrent),
                crossover=CrossoverEngine(gen_llm, max_concurrent=first_config.max_concurrent),
                encoder=IdeaEncoder(),
                repository=repo,
            )

            request = EvolutionRequest(
                challenge=challenge,
                domain=DomainName(domain),
                population_size=population,
                generations=generations,
            )

            state = await engine.run_evolution(request)
            _display_results(state)

        finally:
            if repo is not None:
                await repo.close()
            await router.close_all()

    asyncio.run(_run())


def _display_results(state: EvolutionState) -> None:
    """Muestra los resultados de la evolución en tablas Rich."""
    console.print("\n[bold green]✅ Evolución completada[/bold green]\n")

    summary = Table(title="Resumen", show_header=False)
    summary.add_column("Métrica", style="cyan")
    summary.add_column("Valor", style="green")
    summary.add_row("Generaciones", str(state.generation))
    summary.add_row("Ideas generadas", str(len(state.all_ideas)))
    summary.add_row("Ideas élite", str(len(state.archive)))
    summary.add_row("Cobertura", f"{state.coverage:.1%}")
    summary.add_row("QD-Score", f"{state.qd_score:.2f}")
    summary.add_row("Mejor fitness", f"{state.best_fitness:.4f}")
    console.print(summary)

    top = sorted(state.archive, key=lambda c: c.fitness, reverse=True)[:10]
    if not top:
        return

    table = Table(title="\nTop 10 Ideas Élite (diversas entre sí)")
    table.add_column("#", style="dim", width=4)
    table.add_column("Título", style="bold", max_width=40)
    table.add_column("Fitness", justify="right", style="green")
    table.add_column("Novedad", justify="right")
    table.add_column("Viabilidad", justify="right")
    table.add_column("Utilidad", justify="right")
    table.add_column("Gen", justify="right", style="dim")

    for i, cell in enumerate(top, 1):
        e = cell.elite.evaluation
        table.add_row(
            str(i),
            cell.elite.title[:40],
            f"{cell.fitness:.3f}",
            f"{e.novelty:.2f}" if e else "-",
            f"{e.feasibility:.2f}" if e else "-",
            f"{e.utility:.2f}" if e else "-",
            str(cell.elite.generation),
        )

    console.print(table)

    best = top[0].elite
    console.print(f"\n[bold yellow]⭐ Mejor idea:[/bold yellow] {best.title}")
    console.print(f"   {best.description[:300]}")


if __name__ == "__main__":
    cli()
