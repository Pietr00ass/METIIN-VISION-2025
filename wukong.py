"""Automation scaffold for the WuKong expedition dungeon.

This module mirrors the command line ergonomics of :mod:`dung_polana` but
provides a dedicated stage catalogue for the thirteen-step WuKong flow.  The
actual combat/interaction logic is intentionally left as thin placeholders so
that future development can focus on wiring new computer-vision assets and
controller routines without rewriting the CLI plumbing again.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, sleep
from typing import Callable, Dict, Sequence, Tuple

import click
from loguru import logger

from game_controller import GameController
from settings import GameBind, UserBind
from utils import setup_logger
from vision_detector import VisionDetector


@dataclass(frozen=True)
class StageDefinition:
    """Describe a single WuKong expedition stage."""

    key: str
    title: str
    objective: str
    hint: str | None = None


STAGES: Tuple[StageDefinition, ...] = (
    StageDefinition(
        key="slay_first_wave",
        title="Pokonaj potwory",
        objective="Usuń początkowych przeciwników broniących wejścia do wyprawy.",
        hint="Skup się na mobach pojawiających się wokół startowej polany.",
    ),
    StageDefinition(
        key="destroy_first_metins",
        title="Zniszcz kamienie Metin",
        objective="Rozbij cztery kamienie, aby odblokować dalsze zadania.",
        hint="Po zniszczeniu każdego kamienia sprawdź pasek postępu (Pozostało: 4).",
    ),
    StageDefinition(
        key="clear_second_wave",
        title="Zniszcz wszystkie potwory",
        objective="Wyeliminuj potwory przywołane po rozbiciu metinów.",
    ),
    StageDefinition(
        key="defeat_cloud_guardian",
        title="Pokonaj Obrońcę Chmur",
        objective="Zlikwiduj strażnika blokującego dalszą część wyprawy.",
        hint="Boss posiada krótkie odnowienia – zachowaj czujność na jego uniki.",
    ),
    StageDefinition(
        key="place_phoenix_eggs",
        title="Umieść Jaja Feniksa",
        objective="Zbieraj jaja z pokonanych przeciwników i ustaw je na piedestałach.",
        hint=(
            "Po podniesieniu jaj otwórz ekwipunek (klawisz 'i') i kliknij prawym "
            "przyciskiem myszy na przedmiocie, aby je aktywować."
        ),
    ),
    StageDefinition(
        key="destroy_phoenix_eggs",
        title="Zniszcz Jaja Feniksa",
        objective="Rozbij wszystkie aktywowane jaja, aby zatrzymać wykluwanie wrogów.",
        hint="Wymaga walki w zwarciu – używaj umiejętności obszarowych.",
    ),
    StageDefinition(
        key="repel_three_waves",
        title="Pokonaj trzy fale przeciwników",
        objective="Odeprzyj kolejne fale napastników.",
        hint="Użyj peleryny pod przyciskiem 'F4', aby przyspieszyć spawn fal.",
    ),
    StageDefinition(
        key="destroy_second_metin",
        title="Zniszcz kamień Metin",
        objective="Zlikwiduj kolejny kamień blokujący drogę do bossa.",
    ),
    StageDefinition(
        key="defeat_flaming_phoenix",
        title="Pokonaj Płomiennego Feniksa",
        objective="Zwycięż w starciu z ognistym feniksem strzegącym świątyni.",
        hint="Boss często odskakuje – miej przygotowane umiejętności dystansowe.",
    ),
    StageDefinition(
        key="clear_final_wave",
        title="Pokonaj wszystkich przeciwników",
        objective="Posprzątaj pozostałe grupy potworów po pokonaniu feniksa.",
    ),
    StageDefinition(
        key="destroy_crimson_gourds",
        title="Zniszcz Karmazynowe Gurdy",
        objective="Rozbij cztery gurdy i wyeliminuj obrońców.",
        hint="Każda gurdę najlepiej niszczyć z dystansu, aby uniknąć obrażeń obszarowych.",
    ),
    StageDefinition(
        key="defeat_wukong",
        title="Pokonaj Małpiego Króla WuKonga",
        objective="Pokonaj finałowego bossa wyprawy.",
        hint="Kontroluj aggro i unikaj ogłuszeń, szczególnie w końcowej fazie walki.",
    ),
    StageDefinition(
        key="restart_expedition",
        title="Zacznij wyprawę ponownie",
        objective="Użyj przycisku restartu, aby przygotować się na kolejne przejście.",
        hint="Po wyjściu z areny podejdź do piedestału i kliknij przycisk 'Zacznij ponownie'.",
    ),
)

DEFAULT_STAGE_TIMEOUTS: Tuple[float, ...] = (
    90,
    180,
    120,
    150,
    210,
    150,
    180,
    150,
    210,
    150,
    210,
    240,
    90,
)

StageHandler = Callable[[GameController, VisionDetector, StageDefinition, float], None]


def _parse_stage_timeouts(stage_timeouts: Sequence[float] | str) -> Tuple[float, ...]:
    """Validate and normalize stage timeout configuration.

    Parameters
    ----------
    stage_timeouts:
        Sequence of per-stage timeout values or a comma separated string.
    """

    if isinstance(stage_timeouts, str):
        raw = stage_timeouts.replace(";", ",").replace("\n", ",")
        items = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        items = [str(item) for item in stage_timeouts]

    expected = len(STAGES)
    if len(items) != expected:
        raise ValueError(
            "stage_timeouts must contain exactly "
            f"{expected} wartości rozdzielonych przecinkiem (po jednej na każdy etap)."
        )

    try:
        parsed = tuple(float(item) for item in items)
    except ValueError as exc:  # pragma: no cover - defensive programming
        raise ValueError(
            "Nie udało się sparsować stage_timeouts – upewnij się, że podajesz tylko liczby."
        ) from exc

    return parsed


def _validate_start_stage(stage_index: int) -> None:
    if not 0 <= stage_index < len(STAGES):
        raise ValueError(
            f"Stage index {stage_index} is out of bounds for WuKong expedition (0-{len(STAGES) - 1})."
        )


def _log_stage_banner(stage_index: int, stage: StageDefinition, timeout: float) -> None:
    logger.info("=" * 72)
    logger.info(
        "WuKong etap %s/%s — %s (limit: %ss)",
        stage_index + 1,
        len(STAGES),
        stage.title,
        timeout,
    )
    logger.info(stage.objective)
    if stage.hint:
        logger.debug(stage.hint)
    logger.info("=" * 72)


def _handle_generic_stage(
    game: GameController,
    vision: VisionDetector,
    stage: StageDefinition,
    timeout: float,
) -> None:
    """Fallback handler that simply logs the stage objective.

    The method intentionally avoids injecting any automation until the
    computer-vision templates for WuKong are prepared.  For now it introduces a
    short sleep to mimic the pacing of a live dungeon run and gives room for
    manual interactions.
    """

    del vision  # Unused for now; kept for interface compatibility.
    logger.info("Rozpoczynam etap: %s", stage.title)
    if stage.key == "place_phoenix_eggs":
        logger.info("Otwieram ekwipunek, aby umożliwić rozmieszczenie jaj.")
        game.tap_key(GameBind.EQ_MENU)
        sleep(1.0)
    elif stage.key == "repel_three_waves":
        logger.info("Aktywuję pelerynę (F4), aby wywołać fale przeciwników.")
        game.tap_key(UserBind.MARMUREK)
        sleep(0.5)
    elif stage.key == "restart_expedition":
        logger.info("Zbliż się do piedestału i potwierdź restart wyprawy.")

    sleep(min(timeout, 5.0))
    logger.info("Etap '%s' uznany za zakończony (placeholder).", stage.title)


STAGE_HANDLERS: Dict[str, StageHandler] = {
    stage.key: _handle_generic_stage for stage in STAGES
}


def run(
    stage: int,
    log_level: str,
    saved_credentials_idx: int,
    stage_timeouts: Sequence[float] | str = DEFAULT_STAGE_TIMEOUTS,
) -> None:
    """Run WuKong expedition automation starting from the requested stage."""

    del log_level  # Log level is configured globally via :func:`setup_logger`.

    stage_timeouts_tuple = _parse_stage_timeouts(stage_timeouts)
    _validate_start_stage(stage)

    vision = VisionDetector()
    game = GameController(vision_detector=vision, start_delay=2, saved_credentials_idx=saved_credentials_idx)

    start_time = perf_counter()
    for idx in range(stage, len(STAGES)):
        stage_def = STAGES[idx]
        timeout = stage_timeouts_tuple[idx]
        _log_stage_banner(idx, stage_def, timeout)
        handler = STAGE_HANDLERS.get(stage_def.key, _handle_generic_stage)
        handler(game, vision, stage_def, timeout)

    elapsed = perf_counter() - start_time
    logger.success("WuKong expedition flow finished in %.2fs", elapsed)
    game.reset_game_state()


@click.command()
@click.option("--stage", default=0, type=int, show_default=True, help="Stage to start from (0-indexed).")
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["TRACE", "DEBUG", "INFO"], case_sensitive=False),
    help="Set the logging level.",
)
@click.option("--saved_credentials_idx", default=1, type=int, show_default=True, help="Saved credentials index to use.")
@click.option(
    "--stage-timeouts",
    default=",".join(str(value) for value in DEFAULT_STAGE_TIMEOUTS),
    show_default=True,
    help="Comma separated per-stage timeout configuration.",
)
def main(stage: int, log_level: str, saved_credentials_idx: int, stage_timeouts: str) -> None:
    """CLI entrypoint mirroring :mod:`dung_polana`."""

    log_level = log_level.upper()
    setup_logger(script_name=Path(__file__).name, level=log_level)
    logger.warning("Starting the WuKong expedition bot...")
    run(stage, log_level, saved_credentials_idx, stage_timeouts)


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("WuKong expedition bot interrupted by user.")
