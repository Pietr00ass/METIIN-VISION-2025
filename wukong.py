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
from typing import Callable, Dict, NamedTuple, Optional, Sequence, Tuple

import click
import numpy as np
from loguru import logger
from ultralytics import YOLO
from ultralytics import checks as yolo_checks

from game_controller import GameController
from settings import GameBind, ResourceName, UserBind, MODELS_DIR
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


YOLO_MODEL_FILENAME = "wukong.pt"
DEFAULT_YOLO_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_YOLO_DEVICE = "cuda:0"

WUKONG_YOLO_RESOURCES: Tuple[ResourceName, ...] = (
    ResourceName.WUKONG_MOB,
    ResourceName.WUKONG_METIN,
    ResourceName.WUKONG_CRIMSON_GOURD,
    ResourceName.WUKONG_MONKEY_KING,
    ResourceName.WUKONG_CLOUD_GUARDIAN,
    ResourceName.WUKONG_PHOENIX_EGG,
    ResourceName.WUKONG_FLAMING_PHOENIX,
)


class DetectionResult(NamedTuple):
    positions: Tuple[Tuple[int, int], ...]
    method: str


DETECTION_METHOD_FRAME_MISSING = "frame_missing"
DETECTION_METHOD_YOLO = "yolo"
DETECTION_METHOD_TEMPLATE = "template"
DETECTION_METHOD_UNAVAILABLE = "unavailable"


_YOLO_MODEL: YOLO | None = None
_YOLO_CLASS_IDS: Dict[ResourceName, int] = {}
_YOLO_CONFIDENCE_THRESHOLD = DEFAULT_YOLO_CONFIDENCE_THRESHOLD
_YOLO_VERBOSE = False


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


def _resolve_yolo_class_map(model: YOLO) -> Dict[ResourceName, int]:
    raw_names = {}
    try:
        raw_names = getattr(model, "names", {}) or {}
    except AttributeError:
        raw_names = {}

    if not raw_names:
        raw_names = getattr(getattr(model, "model", None), "names", {}) or {}

    normalized = {str(name).lower(): int(idx) for idx, name in raw_names.items()}
    if not normalized:
        logger.warning("Model YOLO nie udostępnia nazw klas – mapowanie WuKonga będzie niedostępne.")
        return {}

    mapping: Dict[ResourceName, int] = {}
    missing: list[str] = []
    available = ", ".join(sorted(normalized))

    for resource in WUKONG_YOLO_RESOURCES:
        class_idx = normalized.get(resource.value.lower())
        if class_idx is None:
            missing.append(resource.value)
            continue
        mapping[resource] = class_idx

    available_display = available if available else "<brak>"

    logger.info(f"Klasy dostępne w modelu YOLO WuKonga: {available_display}.")

    if missing:
        logger.warning(
            f"Model YOLO WuKonga nie zawiera klas: {', '.join(sorted(missing))}. "
            f"Dostępne klasy: {available_display}."
        )

    if not mapping:
        logger.warning(
            f"Model YOLO WuKonga nie zawiera żadnych rozpoznawalnych klas WuKonga. "
            f"Dostępne klasy: {available_display}."
        )

    return mapping


def _initialize_yolo_model(device: str) -> None:
    global _YOLO_MODEL, _YOLO_CLASS_IDS

    if _YOLO_MODEL is not None:
        return

    try:
        yolo_checks()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(f"Weryfikacja środowiska YOLO zakończona ostrzeżeniem: {exc}")

    model_path = Path(MODELS_DIR) / YOLO_MODEL_FILENAME
    if not model_path.exists():
        logger.warning(
            f"Model YOLO WuKonga nie został znaleziony pod ścieżką {model_path} – "
            "powracam do detekcji szablonowej."
        )
        return

    try:
        model = YOLO(model_path)
    except Exception as exc:  # pragma: no cover - runtime environment specific
        logger.error("Nie udało się załadować modelu YOLO %s: %s", model_path, exc)
        return

    try:
        model.to(device)
    except Exception as exc:  # pragma: no cover - CUDA availability varies
        logger.warning(
            f"Nie można załadować modelu na urządzenie {device} ({exc}). Używam CPU."
        )
        model.to("cpu")

    _YOLO_MODEL = model
    _YOLO_CLASS_IDS = _resolve_yolo_class_map(model)

    if _YOLO_CLASS_IDS:
        logger.info(
            f"Załadowano model YOLO WuKonga – zmapowano {len(_YOLO_CLASS_IDS)} klas."
        )
    else:
        logger.warning("Model YOLO WuKonga załadowany, lecz nie znaleziono żadnych zgodnych klas.")


def _get_yolo_class_id(resource: ResourceName) -> Optional[int]:
    return _YOLO_CLASS_IDS.get(resource)


def _predict_yolo(frame: np.ndarray, *, conf: Optional[float] = None):
    if _YOLO_MODEL is None:
        return None

    inference_conf = _YOLO_CONFIDENCE_THRESHOLD if conf is None else max(float(conf), _YOLO_CONFIDENCE_THRESHOLD)

    try:
        results_list = _YOLO_MODEL.predict(
            source=VisionDetector.fill_non_clickable_wth_black(frame.copy()),
            conf=inference_conf,
            verbose=_YOLO_VERBOSE,
        )
    except Exception as exc:  # pragma: no cover - runtime specific
        logger.error("Błąd podczas inferencji modelu YOLO WuKonga: %s", exc)
        return None

    if not results_list:
        return None

    return results_list[0]


def _detect_with_yolo(
    vision: VisionDetector,
    frame: np.ndarray,
    resource: ResourceName,
    *,
    threshold: Optional[float] = None,
) -> Tuple[Tuple[int, int], ...]:
    class_id = _get_yolo_class_id(resource)
    if class_id is None:
        return tuple()

    result = _predict_yolo(frame, conf=threshold)
    if result is None or result.boxes is None or result.boxes.cls is None:
        return tuple()

    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = (
        result.boxes.conf.cpu().numpy()
        if getattr(result.boxes, "conf", None) is not None
        else np.ones_like(classes, dtype=float)
    )
    mask = classes == class_id

    if threshold is not None:
        mask &= confidences >= float(threshold)

    if not np.any(mask):
        return tuple()

    xywh = result.boxes.xywh.cpu().numpy()
    centers = xywh[mask, :2]
    return tuple(
        vision.get_global_pos((int(round(x)), int(round(y))))
        for x, y in centers
    )


def _detect_resource_positions(
    vision: VisionDetector,
    resource: ResourceName,
    *,
    frame: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
) -> DetectionResult:
    if frame is None:
        frame = vision.capture_frame()
    if frame is None:
        return DetectionResult(tuple(), DETECTION_METHOD_FRAME_MISSING)

    if _YOLO_MODEL is None:
        return DetectionResult(tuple(), DETECTION_METHOD_UNAVAILABLE)

    if _get_yolo_class_id(resource) is None:
        return DetectionResult(tuple(), DETECTION_METHOD_UNAVAILABLE)

    detections = _detect_with_yolo(vision, frame, resource, threshold=threshold)
    if detections:
        return DetectionResult(detections, DETECTION_METHOD_YOLO)

    return DetectionResult(tuple(), DETECTION_METHOD_YOLO)


def _detection_method_verbose_label(method: str) -> str:
    if method == DETECTION_METHOD_YOLO:
        return "modelem YOLO"
    if method == DETECTION_METHOD_TEMPLATE:
        return "szablonem"
    return "niezidentyfikowanym detektorem"


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


def _confirm_template_presence(
    vision: VisionDetector,
    resource: ResourceName,
    label: str,
    *,
    threshold: float = 0.82,
) -> None:
    detection = _detect_resource_positions(
        vision,
        resource,
        threshold=threshold,
    )
    method = detection.method

    if method == DETECTION_METHOD_FRAME_MISSING:
        logger.warning("Nie udało się pobrać klatki do potwierdzenia %s.", label)
        return

    if method == DETECTION_METHOD_UNAVAILABLE:
        logger.debug(
            "Pominięto wstępne potwierdzenie %s – brak detektora dla '%s'.",
            label,
            resource.value,
        )
        return

    method_verbose = _detection_method_verbose_label(method)

    if detection.positions:
        logger.info(
            "Potwierdzono obecność %s (%s; detekcje: %d).",
            label,
            method_verbose,
            len(detection.positions),
        )
    else:
        if method == DETECTION_METHOD_YOLO:
            logger.debug(f"Detekcja YOLO nie odnalazła jeszcze {label}.")
        else:
            logger.debug(f"Detekcja ({method_verbose}) nie odnalazła jeszcze {label}.")


def _make_template_presence_callback(
    vision: VisionDetector,
    resource: ResourceName,
    *,
    label: str,
    threshold: float = 0.82,
    interval: float = 4.0,
) -> Callable[[float], None]:
    state = {"last_check": 0.0, "present": None, "missing_logged": False}

    def _callback(now: float) -> None:
        if now - state["last_check"] < interval:
            return

        detection = _detect_resource_positions(
            vision,
            resource,
            threshold=threshold,
        )
        method = detection.method

        if method == DETECTION_METHOD_UNAVAILABLE:
            if not state["missing_logged"]:
                logger.debug(
                    "Pomijam monitorowanie %s – brak detektora dla '%s'.",
                    label,
                    resource.value,
                )
                state["missing_logged"] = True
            state["last_check"] = now
            return

        if method == DETECTION_METHOD_FRAME_MISSING:
            logger.warning("Nie udało się pobrać klatki do monitorowania %s.", label)
            state["last_check"] = now
            return

        present = bool(detection.positions)
        method_verbose = _detection_method_verbose_label(method)

        if state["present"] is None:
            if present:
                logger.info("Wykryto %s (%s).", label, method_verbose)
            else:
                if method == DETECTION_METHOD_YOLO:
                    logger.debug(f"Detekcja YOLO nie odnalazła jeszcze {label}.")
                else:
                    logger.debug(f"Detekcja ({method_verbose}) nie odnalazła jeszcze {label}.")
        elif present != state["present"]:
            if present:
                logger.info("%s ponownie widoczne (%s).", label.capitalize(), method_verbose)
            else:
                logger.success("%s nie są już wykrywane (%s).", label.capitalize(), method_verbose)

        state["present"] = present
        state["last_check"] = now

    return _callback


def _make_template_count_callback(
    vision: VisionDetector,
    resource: ResourceName,
    *,
    label: str,
    threshold: float = 0.82,
    interval: float = 4.0,
) -> Callable[[float], None]:
    state = {"last_check": 0.0, "count": None, "missing_logged": False}

    def _callback(now: float) -> None:
        if now - state["last_check"] < interval:
            return

        detection = _detect_resource_positions(
            vision,
            resource,
            threshold=threshold,
        )
        method = detection.method

        if method == DETECTION_METHOD_UNAVAILABLE:
            if not state["missing_logged"]:
                logger.debug(
                    "Pomijam monitorowanie liczby %s – brak detektora dla '%s'.",
                    label,
                    resource.value,
                )
                state["missing_logged"] = True
            state["last_check"] = now
            return

        if method == DETECTION_METHOD_FRAME_MISSING:
            logger.warning("Nie udało się pobrać klatki do monitorowania liczby %s.", label)
            state["last_check"] = now
            return

        count = len(detection.positions)
        method_verbose = _detection_method_verbose_label(method)

        if state["count"] is None:
            if count:
                logger.info("Wykryto %d %s (%s).", count, label, method_verbose)
            else:
                if method == DETECTION_METHOD_YOLO:
                    logger.debug(f"Detekcja YOLO nie odnalazła jeszcze {label}.")
                else:
                    logger.debug(f"Detekcja ({method_verbose}) nie odnalazła jeszcze {label}.")
        elif count != state["count"]:
            if count < state["count"]:
                logger.success(
                    "Pozostało %d %s (poprzednio %d; %s).",
                    count,
                    label,
                    state["count"],
                    method_verbose,
                )
            else:
                logger.warning(
                    "Liczba %s wzrosła z %d do %d (%s).",
                    label,
                    state["count"],
                    count,
                    method_verbose,
                )

        state["count"] = count
        state["last_check"] = now

    return _callback


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
        detection = _detect_resource_positions(
            vision,
            ResourceName.WUKONG_PHOENIX_EGG,
            frame=frame,
            threshold=0.82,
        )
        detected_positions = detection.positions
        method = detection.method
        if detected_positions:
            logger.info(
                "Wykryto %d jaj feniksa w ekwipunku (%s).",
                len(detected_positions),
                _detection_method_verbose_label(method),
            )
        else:
            if method == DETECTION_METHOD_UNAVAILABLE:
                logger.debug(
                    "Brak skonfigurowanego detektora jaj feniksa – korzystam z zapasowych współrzędnych.",
                )
            elif method == DETECTION_METHOD_YOLO:
                logger.debug(
                    "Detekcja YOLO nie znalazła jaj feniksa – używam pozycji zapasowych.",
                )
            else:
                logger.debug(
                    "Detekcja szablonowa nie znalazła jaj feniksa – używam pozycji zapasowych.",
                )

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
                "Nie znaleziono jaj poprzez automatyczną detekcję – klikam %d zaprogramowanych slotów.",
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
    _confirm_template_presence(
        vision,
        ResourceName.WUKONG_MOB,
        "mobów pierwszej fali WuKonga",
    )
    mob_tracker = _make_template_presence_callback(
        vision,
        ResourceName.WUKONG_MOB,
        label="mobów pierwszej fali WuKonga",
    )
    _run_combat_stage(
        game,
        vision,
        stage,
        timeout,
        attack_interval=0.8,
        skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
        extra_callbacks=(mob_tracker,),
    )


def _handle_destroy_first_metins(
    game: GameController,
    vision: VisionDetector,
    stage: StageDefinition,
    timeout: float,
) -> None:
    _confirm_template_presence(
        vision,
        ResourceName.WUKONG_METIN,
        "kamieni Metin WuKonga",
    )
    metin_tracker = _make_template_count_callback(
        vision,
        ResourceName.WUKONG_METIN,
        label="kamieni Metin WuKonga",
    )
    _run_combat_stage(
        game,
        vision,
        stage,
        timeout,
        attack_interval=0.9,
        skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
        progress_parser=_extract_remaining_count,
        extra_callbacks=(metin_tracker,),
    )


def _handle_clear_second_wave(
    game: GameController,
    vision: VisionDetector,
    stage: StageDefinition,
    timeout: float,
) -> None:
    _confirm_template_presence(
        vision,
        ResourceName.WUKONG_MOB,
        "mobów drugiej fali WuKonga",
    )
    mob_tracker = _make_template_presence_callback(
        vision,
        ResourceName.WUKONG_MOB,
        label="mobów drugiej fali WuKonga",
    )
    _run_combat_stage(
        game,
        vision,
        stage,
        timeout,
        attack_interval=0.8,
        skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
        extra_callbacks=(mob_tracker,),
    )


def _handle_defeat_cloud_guardian(
    game: GameController,
    vision: VisionDetector,
    stage: StageDefinition,
    timeout: float,
) -> None:
    _confirm_template_presence(
        vision,
        ResourceName.WUKONG_CLOUD_GUARDIAN,
        "Obrońcę Chmur WuKonga",
    )
    guardian_tracker = _make_template_presence_callback(
        vision,
        ResourceName.WUKONG_CLOUD_GUARDIAN,
        label="Obrońcę Chmur WuKonga",
    )
    _run_combat_stage(
        game,
        vision,
        stage,
        timeout,
        attack_interval=0.7,
        skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
        extra_callbacks=(guardian_tracker,),
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

    _confirm_template_presence(
        vision,
        ResourceName.WUKONG_MOB,
        "przeciwników w falach WuKonga",
    )
    mob_tracker = _make_template_presence_callback(
        vision,
        ResourceName.WUKONG_MOB,
        label="przeciwników w falach WuKonga",
    )
    _run_combat_stage(
        game,
        vision,
        stage,
        timeout,
        attack_interval=0.85,
        skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
        lure_interval=25.0,
        progress_parser=_extract_remaining_count,
        extra_callbacks=(mob_tracker,),
    )


def _handle_destroy_second_metin(
    game: GameController,
    vision: VisionDetector,
    stage: StageDefinition,
    timeout: float,
) -> None:
    _confirm_template_presence(
        vision,
        ResourceName.WUKONG_METIN,
        "ostatnich kamieni Metin WuKonga",
    )
    metin_tracker = _make_template_count_callback(
        vision,
        ResourceName.WUKONG_METIN,
        label="kamieni Metin WuKonga",
    )
    _run_combat_stage(
        game,
        vision,
        stage,
        timeout,
        attack_interval=0.9,
        skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
        progress_parser=_extract_remaining_count,
        extra_callbacks=(metin_tracker,),
    )


def _handle_defeat_flaming_phoenix(
    game: GameController,
    vision: VisionDetector,
    stage: StageDefinition,
    timeout: float,
) -> None:
    _confirm_template_presence(
        vision,
        ResourceName.WUKONG_FLAMING_PHOENIX,
        "Płomiennego Feniksa WuKonga",
    )
    phoenix_tracker = _make_template_presence_callback(
        vision,
        ResourceName.WUKONG_FLAMING_PHOENIX,
        label="Płomiennego Feniksa WuKonga",
    )
    _run_combat_stage(
        game,
        vision,
        stage,
        timeout,
        attack_interval=0.7,
        skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
        extra_callbacks=(phoenix_tracker,),
    )


def _handle_clear_final_wave(
    game: GameController,
    vision: VisionDetector,
    stage: StageDefinition,
    timeout: float,
) -> None:
    _confirm_template_presence(
        vision,
        ResourceName.WUKONG_MOB,
        "ostatnich przeciwników WuKonga",
    )
    mob_tracker = _make_template_presence_callback(
        vision,
        ResourceName.WUKONG_MOB,
        label="ostatnich przeciwników WuKonga",
    )
    _run_combat_stage(
        game,
        vision,
        stage,
        timeout,
        attack_interval=0.8,
        skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
        extra_callbacks=(mob_tracker,),
    )


def _handle_destroy_crimson_gourds(
    game: GameController,
    vision: VisionDetector,
    stage: StageDefinition,
    timeout: float,
) -> None:
    _confirm_template_presence(
        vision,
        ResourceName.WUKONG_CRIMSON_GOURD,
        "Karmazynowych Gurd",
    )
    gourd_tracker = _make_template_count_callback(
        vision,
        ResourceName.WUKONG_CRIMSON_GOURD,
        label="Karmazynowych Gurd",
    )
    _run_combat_stage(
        game,
        vision,
        stage,
        timeout,
        attack_interval=0.85,
        skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
        progress_parser=_extract_remaining_count,
        extra_callbacks=(gourd_tracker,),
    )


def _handle_defeat_wukong(
    game: GameController,
    vision: VisionDetector,
    stage: StageDefinition,
    timeout: float,
) -> None:
    _confirm_template_presence(
        vision,
        ResourceName.WUKONG_MONKEY_KING,
        "Małpiego Króla WuKonga",
    )
    boss_tracker = _make_template_presence_callback(
        vision,
        ResourceName.WUKONG_MONKEY_KING,
        label="Małpiego Króla WuKonga",
    )
    _run_combat_stage(
        game,
        vision,
        stage,
        timeout,
        attack_interval=0.65,
        skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
        extra_callbacks=(boss_tracker,),
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
    yolo_confidence_threshold: float = DEFAULT_YOLO_CONFIDENCE_THRESHOLD,
    yolo_device: str = DEFAULT_YOLO_DEVICE,
) -> None:
    """Run WuKong expedition automation starting from the requested stage."""

    log_level = log_level.upper()

    global _YOLO_CONFIDENCE_THRESHOLD, _YOLO_VERBOSE
    _YOLO_CONFIDENCE_THRESHOLD = float(yolo_confidence_threshold)
    _YOLO_VERBOSE = log_level in {"TRACE", "DEBUG"}

    _initialize_yolo_model(yolo_device)
    if _YOLO_MODEL is None:
        logger.warning(
            "Model YOLO WuKonga nie został załadowany – wykorzystywane będą jedynie dostępne szablony.",
        )

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
@click.option(
    "--yolo-confidence-threshold",
    default=DEFAULT_YOLO_CONFIDENCE_THRESHOLD,
    show_default=True,
    type=float,
    help="Minimalna pewność detekcji YOLO (0-1).",
)
@click.option(
    "--yolo-device",
    default=DEFAULT_YOLO_DEVICE,
    show_default=True,
    help="Urządzenie dla modelu YOLO (np. 'cuda:0' lub 'cpu').",
)
def main(
    stage: int,
    log_level: str,
    saved_credentials_idx: int,
    stage_timeouts: str,
    yolo_confidence_threshold: float,
    yolo_device: str,
) -> None:
    """CLI entrypoint mirroring :mod:`dung_polana`."""

    log_level = log_level.upper()
    setup_logger(script_name=Path(__file__).name, level=log_level)
    logger.warning("Starting the WuKong expedition bot...")
    run(
        stage,
        log_level,
        saved_credentials_idx,
        stage_timeouts,
        yolo_confidence_threshold,
        yolo_device,
    )


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("WuKong expedition bot interrupted by user.")