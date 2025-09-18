"""Automation scaffold for the WuKong expedition dungeon.

This module mirrors the command line ergonomics of :mod:`dung_polana` but
provides a dedicated stage catalogue for the thirteen-step WuKong flow.  The
actual combat/interaction logic is intentionally left as thin placeholders so
that future development can focus on wiring new computer-vision assets and
controller routines without rewriting the CLI plumbing again.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, sleep
from typing import Callable, Dict, Optional, Sequence, Tuple

import click
from loguru import logger

from game_controller import GameController
from settings import GameBind, ResourceName, UserBind
from utils import setup_logger
from vision_detector import VisionDetector


@dataclass(frozen=True)
class StageDefinition:
    """Describe a single WuKong expedition stage."""

    key: str
    title: str
    objective: str
    hint: str | None = None
    prompt_keywords: Tuple[str, ...] = ()
    completion_keywords: Tuple[str, ...] = ()


@dataclass(frozen=True)
class WuKongAutomationConfig:
    """Runtime configuration for interactive WuKong expedition actions."""

    egg_slots: Tuple[Tuple[int, int], ...]
    restart_button: Optional[Tuple[int, int]] = None
    restart_confirm_button: Optional[Tuple[int, int]] = None


CONFIG_PATH = Path("data/wukong_config.json")

DEFAULT_EGG_SLOT_CANDIDATES: Tuple[Tuple[int, int], ...] = (
    (648, 244),
    (680, 244),
    (710, 244),
    (742, 244),
    (776, 224),
    (648, 275),
    (680, 275),
    (710, 275),
    (742, 275),
    (776, 275),
)

DEFAULT_RESTART_BUTTON = (733, 65)
DEFAULT_RESTART_CONFIRM_BUTTON = (359, 318)

DEFAULT_AUTOMATION_CONFIG = WuKongAutomationConfig(
    egg_slots=DEFAULT_EGG_SLOT_CANDIDATES,
    restart_button=DEFAULT_RESTART_BUTTON,
    restart_confirm_button=DEFAULT_RESTART_CONFIRM_BUTTON,
)


class StageTimeoutError(RuntimeError):
    """Raised when a WuKong stage exceeds its allotted timeout."""


def _load_wukong_config(path: Path = CONFIG_PATH) -> WuKongAutomationConfig:
    """Load automation coordinates from JSON configuration if available."""

    if not path.exists():
        logger.warning(
            "Nie znaleziono pliku %s – używam domyślnej konfiguracji WuKonga.",
            path,
        )
        return DEFAULT_AUTOMATION_CONFIG

    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("Nie można sparsować %s: %s", path, exc)
        return DEFAULT_AUTOMATION_CONFIG

    def _ensure_points(key: str) -> Tuple[Tuple[int, int], ...]:
        value = raw_data.get(key)
        if not value:
            return getattr(DEFAULT_AUTOMATION_CONFIG, key)
        try:
            return tuple((int(x), int(y)) for x, y in value)
        except (TypeError, ValueError):
            logger.error(
                "Nieprawidłowe współrzędne w polu '%s' w %s – korzystam z domyślnych.",
                key,
                path,
            )
            return getattr(DEFAULT_AUTOMATION_CONFIG, key)

    egg_slots = _ensure_points("egg_slots")

    def _ensure_point(key: str) -> Optional[Tuple[int, int]]:
        value = raw_data.get(key)
        if not value:
            return getattr(DEFAULT_AUTOMATION_CONFIG, key)
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError, IndexError):
            logger.error(
                "Nieprawidłowe współrzędne w polu '%s' w %s – korzystam z domyślnych.",
                key,
                path,
            )
            return getattr(DEFAULT_AUTOMATION_CONFIG, key)

    restart_button = _ensure_point("restart_button")
    restart_confirm_button = _ensure_point("restart_confirm_button")

    return WuKongAutomationConfig(
        egg_slots=egg_slots,
        restart_button=restart_button,
        restart_confirm_button=restart_confirm_button,
    )


STAGES: Tuple[StageDefinition, ...] = (
    StageDefinition(
        key="slay_first_wave",
        title="Pokonaj potwory",
        objective="Usuń początkowych przeciwników broniących wejścia do wyprawy.",
        hint="Skup się na mobach pojawiających się wokół startowej polany.",
        prompt_keywords=("pokonaj", "potwory"),
        completion_keywords=("zniszcz", "kamienie"),
    ),
    StageDefinition(
        key="destroy_first_metins",
        title="Zniszcz kamienie Metin",
        objective="Rozbij cztery kamienie, aby odblokować dalsze zadania.",
        hint="Po zniszczeniu każdego kamienia sprawdź pasek postępu (Pozostało: 4).",
        prompt_keywords=("zniszcz", "kamienie", "metin"),
        completion_keywords=("zniszcz", "wszystkie", "potwory"),
    ),
    StageDefinition(
        key="clear_second_wave",
        title="Zniszcz wszystkie potwory",
        objective="Wyeliminuj potwory przywołane po rozbiciu metinów.",
        prompt_keywords=("zniszcz", "wszystkie", "potwory"),
        completion_keywords=("pokonaj", "obrońcę"),
    ),
    StageDefinition(
        key="defeat_cloud_guardian",
        title="Pokonaj Obrońcę Chmur",
        objective="Zlikwiduj strażnika blokującego dalszą część wyprawy.",
        hint="Boss posiada krótkie odnowienia – zachowaj czujność na jego uniki.",
        prompt_keywords=("pokonaj", "obrońcę", "chmur"),
        completion_keywords=("umieść", "jaja", "feniksa"),
    ),
    StageDefinition(
        key="place_phoenix_eggs",
        title="Umieść Jaja Feniksa",
        objective="Zbieraj jaja z pokonanych przeciwników i ustaw je na piedestałach.",
        hint=(
            "Po podniesieniu jaj otwórz ekwipunek (klawisz 'i') i kliknij prawym "
            "przyciskiem myszy na przedmiocie, aby je aktywować."
        ),
        prompt_keywords=("umieść", "jaja", "feniksa"),
        completion_keywords=("zniszcz", "jaja", "feniksa"),
    ),
    StageDefinition(
        key="destroy_phoenix_eggs",
        title="Zniszcz Jaja Feniksa",
        objective="Rozbij wszystkie aktywowane jaja, aby zatrzymać wykluwanie wrogów.",
        hint="Wymaga walki w zwarciu – używaj umiejętności obszarowych.",
        prompt_keywords=("zniszcz", "jaja", "feniksa"),
        completion_keywords=("pokonaj", "trzy", "fale"),
    ),
    StageDefinition(
        key="repel_three_waves",
        title="Pokonaj trzy fale przeciwników",
        objective="Odeprzyj kolejne fale napastników.",
        hint="Użyj peleryny pod przyciskiem 'F4', aby przyspieszyć spawn fal.",
        prompt_keywords=("pokonaj", "trzy", "fale"),
        completion_keywords=("zniszcz", "kamień", "metin"),
    ),
    StageDefinition(
        key="destroy_second_metin",
        title="Zniszcz kamień Metin",
        objective="Zlikwiduj kolejny kamień blokujący drogę do bossa.",
        prompt_keywords=("zniszcz", "kamień", "metin"),
        completion_keywords=("pokonaj", "płomiennego", "feniksa"),
    ),
    StageDefinition(
        key="defeat_flaming_phoenix",
        title="Pokonaj Płomiennego Feniksa",
        objective="Zwycięż w starciu z ognistym feniksem strzegącym świątyni.",
        hint="Boss często odskakuje – miej przygotowane umiejętności dystansowe.",
        prompt_keywords=("pokonaj", "płomiennego", "feniksa"),
        completion_keywords=("pokonaj", "wszystkich", "przeciwników"),
    ),
    StageDefinition(
        key="clear_final_wave",
        title="Pokonaj wszystkich przeciwników",
        objective="Posprzątaj pozostałe grupy potworów po pokonaniu feniksa.",
        prompt_keywords=("pokonaj", "wszystkich", "przeciwników"),
        completion_keywords=("karmazynowe", "gurdy"),
    ),
    StageDefinition(
        key="destroy_crimson_gourds",
        title="Zniszcz Karmazynowe Gurdy",
        objective="Rozbij cztery gurdy i wyeliminuj obrońców.",
        hint="Każda gurdę najlepiej niszczyć z dystansu, aby uniknąć obrażeń obszarowych.",
        prompt_keywords=("karmazynowe", "gurdy"),
        completion_keywords=("małpiego", "króla", "wukonga"),
    ),
    StageDefinition(
        key="defeat_wukong",
        title="Pokonaj Małpiego Króla WuKonga",
        objective="Pokonaj finałowego bossa wyprawy.",
        hint="Kontroluj aggro i unikaj ogłuszeń, szczególnie w końcowej fazie walki.",
        prompt_keywords=("małpiego", "króla", "wukonga"),
        completion_keywords=("zacznij", "ponownie"),
    ),
    StageDefinition(
        key="restart_expedition",
        title="Zacznij wyprawę ponownie",
        objective="Użyj przycisku restartu, aby przygotować się na kolejne przejście.",
        hint="Po wyjściu z areny podejdź do piedestału i kliknij przycisk 'Zacznij ponownie'.",
        prompt_keywords=("zacznij", "ponownie"),
        completion_keywords=("pokonaj", "potwory"),
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


AUTOMATION_CONFIG = DEFAULT_AUTOMATION_CONFIG
_BUFFS_INITIALIZED = False
_REMAINING_PATTERN = re.compile(r"pozostał[oa]?\s*:?\s*(\d+)")
_PROMPT_WAIT_LIMIT = 15.0

DEFAULT_SKILL_COOLDOWNS: Dict[UserBind, float] = {
    UserBind.WIREK: 9.0,
    UserBind.SZARZA: 12.0,
}


def _normalize_text(text: str) -> str:
    return text.strip().lower()


def _message_contains_keywords(message: str, keywords: Sequence[str]) -> bool:
    if not keywords:
        return False
    normalized = _normalize_text(message)
    return all(keyword in normalized for keyword in keywords)


def _extract_remaining_count(message: str) -> Optional[int]:
    match = _REMAINING_PATTERN.search(_normalize_text(message))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:  # pragma: no cover - defensive
        return None


def _capture_dungeon_message(vision: VisionDetector) -> str:
    frame = vision.capture_frame()
    if frame is None:
        logger.warning("Brak klatki z gry – ponawiam próbę po krótkiej pauzie.")
        sleep(1.0)
        return ""
    return vision.get_dungeon_message(frame)


def _wait_for_stage_prompt(vision: VisionDetector, stage: StageDefinition, timeout: float) -> None:
    if not stage.prompt_keywords:
        return

    deadline = perf_counter() + min(timeout, _PROMPT_WAIT_LIMIT)
    while perf_counter() < deadline:
        message = _capture_dungeon_message(vision)
        if message and _message_contains_keywords(message, stage.prompt_keywords):
            logger.info("Wykryto komunikat etapu: %s", message)
            return
        sleep(0.5)

    logger.warning(
        "Nie udało się potwierdzić komunikatu etapu '%s' w ciągu %.1fs.",
        stage.title,
        min(timeout, _PROMPT_WAIT_LIMIT),
    )


def _prepare_for_combat(game: GameController) -> None:
    global _BUFFS_INITIALIZED
    if _BUFFS_INITIALIZED:
        return

    logger.debug("Aktywuję premie bojowe przed pierwszym starciem.")
    game.use_boosters()
    game.toggle_passive_skills(reset_animation=False)
    _BUFFS_INITIALIZED = True


def _make_basic_combat_action(
    game: GameController,
    *,
    attack_interval: float = 1.0,
    skill_cooldowns: Optional[Dict[UserBind, float]] = None,
    lure_interval: Optional[float] = None,
    lure_key: UserBind = UserBind.MARMUREK,
    extra_callbacks: Optional[Sequence[Callable[[float], None]]] = None,
) -> Callable[[float], None]:
    last_attack = 0.0
    next_skill_use: Dict[UserBind, float] = {}
    if skill_cooldowns:
        next_skill_use = {skill: 0.0 for skill in skill_cooldowns}
    next_lure = 0.0 if lure_interval is not None else None

    def _action(now: float) -> None:
        nonlocal last_attack, next_lure

        if now - last_attack >= attack_interval:
            game.tap_key(GameBind.ATTACK)
            last_attack = now

        for skill, cooldown in (skill_cooldowns or {}).items():
            if now >= next_skill_use[skill]:
                game.tap_key(skill)
                next_skill_use[skill] = now + cooldown

        if lure_interval is not None and next_lure is not None and now >= next_lure:
            game.tap_key(lure_key)
            next_lure = now + lure_interval

        if extra_callbacks:
            for callback in extra_callbacks:
                callback(now)

    return _action


def _monitor_stage(
    game: GameController,
    vision: VisionDetector,
    stage: StageDefinition,
    timeout: float,
    *,
    action_callback: Optional[Callable[[float], None]] = None,
    progress_parser: Optional[Callable[[str], Optional[int]]] = None,
    completion_keywords: Optional[Sequence[str]] = None,
    prompt_already_confirmed: bool = False,
) -> None:
    stage_start = perf_counter()
    if not prompt_already_confirmed:
        _wait_for_stage_prompt(vision, stage, timeout)

    prompt_seen = prompt_already_confirmed
    last_progress: Optional[int] = None
    last_message = ""

    while True:
        now = perf_counter()
        if now - stage_start > timeout:
            raise StageTimeoutError(
                f"Etap '{stage.title}' przekroczył limit {timeout:.0f}s."
            )

        if action_callback is not None:
            action_callback(now)

        message = _capture_dungeon_message(vision)
        if message:
            if message != last_message:
                logger.debug("Komunikat lochów: %s", message)
                last_message = message

            if _message_contains_keywords(message, stage.prompt_keywords):
                prompt_seen = True
                if progress_parser is not None:
                    remaining = progress_parser(message)
                    if remaining is not None and remaining != last_progress:
                        logger.info("Pozostało: %s", remaining)
                        last_progress = remaining
            elif prompt_seen:
                if completion_keywords and _message_contains_keywords(message, completion_keywords):
                    logger.info("Wykryto komunikat kolejnego etapu: %s", message)
                else:
                    logger.info("Komunikat etapu uległ zmianie: %s", message)
                elapsed = now - stage_start
                logger.success("Etap '%s' ukończony w %.1fs.", stage.title, elapsed)
                return

        sleep(0.5)


def _use_phoenix_eggs(game: GameController) -> None:
    logger.info("Otwieram ekwipunek w poszukiwaniu jaj feniksa.")
    game.tap_key(GameBind.EQ_MENU)
    sleep(0.4)

    vision = game.vision_detector
    frame = vision.capture_frame()
    detected_positions: Tuple[Tuple[int, int], ...] = tuple()

    if frame is None:
        logger.warning("Nie udało się pobrać klatki ekranu – używam pozycji zapasowych.")
    else:
        detected_positions = vision.locate_all_templates(
            ResourceName.WUKONG_PHOENIX_EGG,
            frame=frame,
            threshold=0.82,
        )
        if detected_positions:
            logger.info("Wykryto %d jaj feniksa w ekwipunku.", len(detected_positions))

    if not detected_positions:
        if not AUTOMATION_CONFIG.egg_slots:
            logger.warning(
                "Brak skonfigurowanych slotów jaj feniksa – oczekuję na ręczne użycie przedmiotu."
            )
            fallback_positions: Tuple[Tuple[int, int], ...] = tuple()
        else:
            fallback_positions = tuple(
                vision.get_global_pos(slot) for slot in AUTOMATION_CONFIG.egg_slots
            )
            logger.info(
                "Nie znaleziono jaj poprzez wzorzec – klikam %d zaprogramowanych slotów.",
                len(fallback_positions),
            )
        detected_positions = fallback_positions

    unique_positions = list(dict.fromkeys(detected_positions))
    for pos in unique_positions:
        game.click_at(pos, right=True)
        sleep(0.2)

    game.tap_key(GameBind.EQ_MENU)
    sleep(0.3)

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
    """Fallback handler that only monitors dungeon messages and respects timeout."""

    logger.warning(
        "Brak dedykowanego handlera dla etapu '%s' – działanie ograniczone do monitoringu.",
        stage.title,
    )
    _monitor_stage(
        game,
        vision,
        stage,
        timeout,
        action_callback=None,
        completion_keywords=stage.completion_keywords,
        progress_parser=_extract_remaining_count,
    )


def _run_combat_stage(
    game: GameController,
    vision: VisionDetector,
    stage: StageDefinition,
    timeout: float,
    *,
    attack_interval: float = 1.0,
    skill_cooldowns: Optional[Dict[UserBind, float]] = None,
    lure_interval: Optional[float] = None,
    extra_callbacks: Optional[Sequence[Callable[[float], None]]] = None,
    progress_parser: Optional[Callable[[str], Optional[int]]] = None,
) -> None:
    _prepare_for_combat(game)
    game.start_attack()
    action = _make_basic_combat_action(
        game,
        attack_interval=attack_interval,
        skill_cooldowns=skill_cooldowns,
        lure_interval=lure_interval,
        extra_callbacks=extra_callbacks,
    )

    try:
        _monitor_stage(
            game,
            vision,
            stage,
            timeout,
            action_callback=action,
            completion_keywords=stage.completion_keywords,
            progress_parser=progress_parser,
        )
    finally:
        game.stop_attack()


def _execute_restart_sequence(game: GameController, vision: VisionDetector) -> bool:
    frame = vision.capture_frame()
    start_pos = vision.locate_template(ResourceName.WUKONG_RESTART, frame=frame, threshold=0.8)
    fallback_start = (
        vision.get_global_pos(AUTOMATION_CONFIG.restart_button)
        if AUTOMATION_CONFIG.restart_button is not None
        else None
    )

    if start_pos is not None:
        logger.info("Wzorzec przycisku restartu odnaleziony – klikam.")
    elif fallback_start is not None:
        logger.info(
            "Nie znaleziono wzorca restartu – klikam zapasowe współrzędne %s.",
            AUTOMATION_CONFIG.restart_button,
        )
        start_pos = fallback_start
    else:
        logger.warning(
            "Brak szablonu i zapasowych współrzędnych przycisku restartu – oczekuję na kliknięcie ręczne."
        )
        return False

    game.click_at(start_pos)
    sleep(0.6)

    frame = vision.capture_frame()
    confirm_pos = vision.locate_template(ResourceName.WUKONG_RESTART_CONFIRM, frame=frame, threshold=0.8)
    fallback_confirm = (
        vision.get_global_pos(AUTOMATION_CONFIG.restart_confirm_button)
        if AUTOMATION_CONFIG.restart_confirm_button is not None
        else None
    )

    if confirm_pos is not None:
        logger.info("Znalazłem przycisk potwierdzenia restartu – klikam.")
    elif fallback_confirm is not None:
        logger.info(
            "Brak wzorca potwierdzenia – klikam zapasowe współrzędne %s.",
            AUTOMATION_CONFIG.restart_confirm_button,
        )
        confirm_pos = fallback_confirm
    else:
        logger.warning("Nie mogę potwierdzić restartu – brak współrzędnych zapasowych.")
        return False

    game.click_at(confirm_pos)
    sleep(0.5)
    return True


def _handle_slay_first_wave(
    game: GameController,
    vision: VisionDetector,
    stage: StageDefinition,
    timeout: float,
) -> None:
    _run_combat_stage(
        game,
        vision,
        stage,
        timeout,
        attack_interval=0.8,
        skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
    )


def _handle_destroy_first_metins(
    game: GameController,
    vision: VisionDetector,
    stage: StageDefinition,
    timeout: float,
) -> None:
    _run_combat_stage(
        game,
        vision,
        stage,
        timeout,
        attack_interval=0.9,
        skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
        progress_parser=_extract_remaining_count,
    )


def _handle_clear_second_wave(
    game: GameController,
    vision: VisionDetector,
    stage: StageDefinition,
    timeout: float,
) -> None:
    _run_combat_stage(
        game,
        vision,
        stage,
        timeout,
        attack_interval=0.8,
        skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
    )


def _handle_defeat_cloud_guardian(
    game: GameController,
    vision: VisionDetector,
    stage: StageDefinition,
    timeout: float,
) -> None:
    _run_combat_stage(
        game,
        vision,
        stage,
        timeout,
        attack_interval=0.7,
        skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
    )


def _handle_place_phoenix_eggs(
    game: GameController,
    vision: VisionDetector,
    stage: StageDefinition,
    timeout: float,
) -> None:
    last_egg_usage = {"time": 0.0}

    def _egg_callback(now: float) -> None:
        if now - last_egg_usage["time"] < 8.0:
            return
        _use_phoenix_eggs(game)
        last_egg_usage["time"] = now

    _run_combat_stage(
        game,
        vision,
        stage,
        timeout,
        attack_interval=0.8,
        skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
        extra_callbacks=(_egg_callback,),
        progress_parser=_extract_remaining_count,
    )


def _handle_destroy_phoenix_eggs(
    game: GameController,
    vision: VisionDetector,
    stage: StageDefinition,
    timeout: float,
) -> None:
    _run_combat_stage(
        game,
        vision,
        stage,
        timeout,
        attack_interval=0.75,
        skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
        progress_parser=_extract_remaining_count,
    )


def _handle_repel_three_waves(
    game: GameController,
    vision: VisionDetector,
    stage: StageDefinition,
    timeout: float,
) -> None:
    logger.info("Aktywuję pelerynę (F4), aby przyspieszyć pojawianie się fal.")
    game.tap_key(UserBind.MARMUREK)

    _run_combat_stage(
        game,
        vision,
        stage,
        timeout,
        attack_interval=0.85,
        skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
        lure_interval=25.0,
        progress_parser=_extract_remaining_count,
    )


def _handle_destroy_second_metin(
    game: GameController,
    vision: VisionDetector,
    stage: StageDefinition,
    timeout: float,
) -> None:
    _run_combat_stage(
        game,
        vision,
        stage,
        timeout,
        attack_interval=0.9,
        skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
        progress_parser=_extract_remaining_count,
    )


def _handle_defeat_flaming_phoenix(
    game: GameController,
    vision: VisionDetector,
    stage: StageDefinition,
    timeout: float,
) -> None:
    _run_combat_stage(
        game,
        vision,
        stage,
        timeout,
        attack_interval=0.7,
        skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
    )


def _handle_clear_final_wave(
    game: GameController,
    vision: VisionDetector,
    stage: StageDefinition,
    timeout: float,
) -> None:
    _run_combat_stage(
        game,
        vision,
        stage,
        timeout,
        attack_interval=0.8,
        skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
    )


def _handle_destroy_crimson_gourds(
    game: GameController,
    vision: VisionDetector,
    stage: StageDefinition,
    timeout: float,
) -> None:
    _run_combat_stage(
        game,
        vision,
        stage,
        timeout,
        attack_interval=0.85,
        skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
        progress_parser=_extract_remaining_count,
    )


def _handle_defeat_wukong(
    game: GameController,
    vision: VisionDetector,
    stage: StageDefinition,
    timeout: float,
) -> None:
    _run_combat_stage(
        game,
        vision,
        stage,
        timeout,
        attack_interval=0.65,
        skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
    )


def _handle_restart_expedition(
    game: GameController,
    vision: VisionDetector,
    stage: StageDefinition,
    timeout: float,
) -> None:
    _wait_for_stage_prompt(vision, stage, timeout)

    _execute_restart_sequence(game, vision)

    _monitor_stage(
        game,
        vision,
        stage,
        timeout,
        action_callback=None,
        completion_keywords=stage.completion_keywords,
        prompt_already_confirmed=True,
    )

STAGE_HANDLERS: Dict[str, StageHandler] = {
    "slay_first_wave": _handle_slay_first_wave,
    "destroy_first_metins": _handle_destroy_first_metins,
    "clear_second_wave": _handle_clear_second_wave,
    "defeat_cloud_guardian": _handle_defeat_cloud_guardian,
    "place_phoenix_eggs": _handle_place_phoenix_eggs,
    "destroy_phoenix_eggs": _handle_destroy_phoenix_eggs,
    "repel_three_waves": _handle_repel_three_waves,
    "destroy_second_metin": _handle_destroy_second_metin,
    "defeat_flaming_phoenix": _handle_defeat_flaming_phoenix,
    "clear_final_wave": _handle_clear_final_wave,
    "destroy_crimson_gourds": _handle_destroy_crimson_gourds,
    "defeat_wukong": _handle_defeat_wukong,
    "restart_expedition": _handle_restart_expedition,
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

    global AUTOMATION_CONFIG, _BUFFS_INITIALIZED
    AUTOMATION_CONFIG = _load_wukong_config()
    _BUFFS_INITIALIZED = False
    logger.debug("Załadowana konfiguracja WuKonga: %s", AUTOMATION_CONFIG)

    vision = VisionDetector()
    game = GameController(vision_detector=vision, start_delay=2, saved_credentials_idx=saved_credentials_idx)

    start_time = perf_counter()
    aborted = False
    for idx in range(stage, len(STAGES)):
        stage_def = STAGES[idx]
        timeout = stage_timeouts_tuple[idx]
        _log_stage_banner(idx, stage_def, timeout)
        handler = STAGE_HANDLERS.get(stage_def.key, _handle_generic_stage)
        try:
            handler(game, vision, stage_def, timeout)
        except StageTimeoutError as exc:
            logger.error(str(exc))
            aborted = True
            break

    elapsed = perf_counter() - start_time
    if aborted:
        logger.error("WuKong expedition przerwana po %.2fs z powodu przekroczonego limitu.", elapsed)
    else:
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