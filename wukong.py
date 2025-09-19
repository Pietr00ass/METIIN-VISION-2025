"""High level automation script for the WuKong expedition dungeon.

This module reshapes the previous experiment-driven implementation into a
single orchestration class inspired by ``dung_polana.py``.  The goal is to keep
all of the knowledge gathered while building the initial WuKong prototype, but
present it in a structure that mirrors the well-tested Valium Polana runner.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, sleep
from typing import Callable, Dict, Iterable, NamedTuple, Optional, Sequence, Tuple

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


class StageTimeoutError(RuntimeError):
    """Raised when a WuKong stage exceeds its allotted timeout."""


CONFIG_PATH = Path("data/wukong_config.json")
YOLO_MODEL_FILENAME = "wukong.pt"
DEFAULT_YOLO_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_YOLO_DEVICE = "cuda:0"

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


WUKONG_YOLO_ALIASES: Dict[ResourceName, Tuple[str, ...]] = {
    ResourceName.WUKONG_MOB: ("wukong_mob", "moby"),
    ResourceName.WUKONG_METIN: ("wukong_metin", "metin"),
    ResourceName.WUKONG_CRIMSON_GOURD: ("wukong_crimson_gourd", "karmazynowy_gurd"),
    ResourceName.WUKONG_MONKEY_KING: ("wukong_monkey_king", "wukong"),
    ResourceName.WUKONG_CLOUD_GUARDIAN: ("wukong_cloud_guardian", "obronca_chmur"),
    ResourceName.WUKONG_PHOENIX_EGG: ("wukong_phoenix_egg", "jajo", "jaja_feniksa"),
    ResourceName.WUKONG_FLAMING_PHOENIX: ("wukong_flaming_phoenix", "plomienny_feniks"),
}

WUKONG_RESOURCE_LABELS: Dict[ResourceName, str] = {
    ResourceName.WUKONG_MOB: "przeciwników WuKonga",
    ResourceName.WUKONG_METIN: "kamieni Metin WuKonga",
    ResourceName.WUKONG_CRIMSON_GOURD: "Karmazynowych Gurd WuKonga",
    ResourceName.WUKONG_MONKEY_KING: "Małpiego Króla WuKonga",
    ResourceName.WUKONG_CLOUD_GUARDIAN: "Obrońcę Chmur WuKonga",
    ResourceName.WUKONG_PHOENIX_EGG: "jaj Feniksa WuKonga",
    ResourceName.WUKONG_FLAMING_PHOENIX: "Płomiennego Feniksa WuKonga",
}

_OPTIONAL_YOLO_RESOURCES = {ResourceName.WUKONG_PHOENIX_EGG}


class DetectionResult(NamedTuple):
    positions: Tuple[Tuple[int, int], ...]
    method: str


DETECTION_METHOD_FRAME_MISSING = "frame_missing"
DETECTION_METHOD_YOLO = "yolo"
DETECTION_METHOD_TEMPLATE = "template"
DETECTION_METHOD_UNAVAILABLE = "unavailable"


REMAINING_PATTERN = re.compile(r"pozostał[oa]?\s*:?\s*(\d+)")
PROMPT_WAIT_LIMIT = 15.0

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

DEFAULT_SKILL_COOLDOWNS: Dict[UserBind, float] = {
    UserBind.WIREK: 9.0,
    UserBind.SZARZA: 12.0,
}


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


class WuKongAutomation:
    """Class orchestrating the WuKong expedition flow."""

    def __init__(
        self,
        *,
        log_level: str,
        saved_credentials_idx: int,
        stage_timeouts: Sequence[float] | str,
        yolo_confidence_threshold: float,
        yolo_device: str,
    ) -> None:
        self.log_level = log_level.upper()
        self.saved_credentials_idx = saved_credentials_idx
        self.stage_timeouts = self._parse_stage_timeouts(stage_timeouts)
        self._yolo_confidence_threshold = max(float(yolo_confidence_threshold), 0.0)
        self._yolo_device = yolo_device
        self._yolo_verbose = self.log_level in {"TRACE", "DEBUG"}

        self.config = self._load_wukong_config()
        logger.debug("Załadowana konfiguracja WuKonga: %s", self.config)

        self.vision = VisionDetector()
        self.game = GameController(
            vision_detector=self.vision,
            start_delay=2,
            saved_credentials_idx=self.saved_credentials_idx,
        )

        self._buffs_initialized = False
        self._yolo_model: YOLO | None = None
        self._yolo_class_ids: Dict[ResourceName, int] = {}

        self._initialize_yolo_model()

        self._stage_handlers: Dict[str, Callable[[StageDefinition, float], float]] = {
            "slay_first_wave": self._handle_slay_first_wave,
            "destroy_first_metins": self._handle_destroy_first_metins,
            "clear_second_wave": self._handle_clear_second_wave,
            "defeat_cloud_guardian": self._handle_defeat_cloud_guardian,
            "place_phoenix_eggs": self._handle_place_phoenix_eggs,
            "destroy_phoenix_eggs": self._handle_destroy_phoenix_eggs,
            "repel_three_waves": self._handle_repel_three_waves,
            "destroy_second_metin": self._handle_destroy_second_metin,
            "defeat_flaming_phoenix": self._handle_defeat_flaming_phoenix,
            "clear_final_wave": self._handle_clear_final_wave,
            "destroy_crimson_gourds": self._handle_destroy_crimson_gourds,
            "defeat_wukong": self._handle_defeat_wukong,
            "restart_expedition": self._handle_restart_expedition,
        }

        self._detection_log_state: Dict[ResourceName, Tuple[str, int]] = {}
        self._last_stage_durations: list[Tuple[str, float]] = []

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_stage_timeouts(stage_timeouts: Sequence[float] | str) -> Tuple[float, ...]:
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
        except ValueError as exc:
            raise ValueError(
                "Nie udało się sparsować stage_timeouts – upewnij się, że podajesz tylko liczby."
            ) from exc

        return parsed

    @staticmethod
    def _validate_start_stage(start_stage: int) -> None:
        if start_stage < 0 or start_stage >= len(STAGES):
            raise ValueError(
                f"Stage index {start_stage} spoza zakresu 0-{len(STAGES) - 1}."
            )

    def run(self, start_stage: int) -> None:
        self._validate_start_stage(start_stage)
        start_time = perf_counter()
        aborted = False
        self._last_stage_durations = []
        stage_durations: list[Tuple[int, float]] = []

        if self._yolo_model is None:
            logger.warning(
                "Model YOLO WuKonga nie został załadowany – wykorzystywane będą jedynie dostępne szablony."
            )

        for idx in range(start_stage, len(STAGES)):
            stage = STAGES[idx]
            timeout = self.stage_timeouts[idx]
            self._log_stage_banner(idx, stage, timeout)

            handler = self._stage_handlers.get(stage.key, self._handle_generic_stage)
            try:
                duration = handler(stage, timeout)
            except StageTimeoutError as exc:
                logger.error(str(exc))
                aborted = True
                break
            else:
                stage_durations.append((idx, duration))
                self._last_stage_durations.append((stage.key, duration))

        if stage_durations:
            logger.info("Podsumowanie ukończonych etapów WuKonga:")
            for idx, duration in stage_durations:
                stage_info = STAGES[idx]
                logger.info(
                    " %2d. %-35s — %5.1fs",
                    idx + 1,
                    stage_info.title,
                    duration,
                )

        elapsed = perf_counter() - start_time
        if aborted:
            logger.error("WuKong expedition przerwana po %.2fs z powodu przekroczonego limitu.", elapsed)
        else:
            logger.success("WuKong expedition flow finished in %.2fs", elapsed)

        self.game.reset_game_state()

    # ------------------------------------------------------------------
    # Configuration & model loading
    # ------------------------------------------------------------------

    def _load_wukong_config(self, path: Path = CONFIG_PATH) -> WuKongAutomationConfig:
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

        return WuKongAutomationConfig(
            egg_slots=_ensure_points("egg_slots"),
            restart_button=_ensure_point("restart_button"),
            restart_confirm_button=_ensure_point("restart_confirm_button"),
        )

    def _initialize_yolo_model(self) -> None:
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
            model.to(self._yolo_device)
        except Exception as exc:  # pragma: no cover - CUDA availability varies
            logger.warning(
                f"Nie można załadować modelu na urządzenie {self._yolo_device} ({exc}). Używam CPU."
            )
            model.to("cpu")

        self._yolo_model = model
        self._yolo_class_ids = self._resolve_yolo_class_map(model)

        if self._yolo_class_ids:
            logger.info(
                f"Załadowano model YOLO WuKonga – zmapowano {len(self._yolo_class_ids)} klas."
            )
        else:
            logger.warning("Model YOLO WuKonga załadowany, lecz nie znaleziono żadnych zgodnych klas.")

    def _resolve_yolo_class_map(self, model: YOLO) -> Dict[ResourceName, int]:
        raw_names: Dict[int, str] = {}
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
        available_display = ", ".join(sorted(normalized))
        alias_mismatches: list[Tuple[str, str]] = []
        missing_resources: list[Tuple[ResourceName, Tuple[str, ...]]] = []

        def _format_aliases(values: Iterable[str]) -> str:
            seen: set[str] = set()
            ordered: list[str] = []
            for candidate in values:
                normalized_alias = candidate.lower()
                if normalized_alias in seen:
                    continue
                seen.add(normalized_alias)
                ordered.append(candidate)
            return ", ".join(ordered)

        for resource, aliases in WUKONG_YOLO_ALIASES.items():
            canonical_alias = aliases[0]
            matched_alias = None
            for alias in aliases:
                class_idx = normalized.get(alias.lower())
                if class_idx is not None:
                    mapping[resource] = class_idx
                    matched_alias = alias
                    break

            if matched_alias is None:
                missing_resources.append((resource, aliases))
                continue

            if matched_alias.lower() != canonical_alias.lower():
                alias_mismatches.append((canonical_alias, matched_alias))

        if alias_mismatches:
            formatted = ", ".join(
                f"'{canonical}' dopasowano aliasem '{alias}'" for canonical, alias in alias_mismatches
            )
            logger.warning(
                f"Mapowanie YOLO WuKonga wykorzystało aliasy: {formatted}. Dostępne klasy: {available_display}"
            )

        if missing_resources:
            required = [item for item in missing_resources if item[0] not in _OPTIONAL_YOLO_RESOURCES]
            optional = [item for item in missing_resources if item[0] in _OPTIONAL_YOLO_RESOURCES]

            if required:
                formatted_required = ", ".join(
                    f"'{aliases[0]}' (aliasy: {_format_aliases(aliases)})" for _, aliases in required
                )
                logger.warning(
                    f"Klasy YOLO dla zasobów {formatted_required} nie zostały znalezione w modelu. "
                    f"Dostępne klasy: {available_display}"
                )

            if optional:
                formatted_optional = ", ".join(
                    f"'{aliases[0]}' (aliasy: {_format_aliases(aliases)})" for _, aliases in optional
                )
                logger.debug(
                    f"Opcjonalne zasoby WuKonga bez mapowania: {formatted_optional}. "
                    f"Dostępne klasy: {available_display}"
                )

        return mapping

    # ------------------------------------------------------------------
    # Detection utilities
    # ------------------------------------------------------------------

    def _get_yolo_class_id(self, resource: ResourceName) -> Optional[int]:
        return self._yolo_class_ids.get(resource)

    def _predict_yolo(self, frame: np.ndarray, *, conf: Optional[float] = None):
        if self._yolo_model is None:
            return None

        inference_conf = (
            self._yolo_confidence_threshold
            if conf is None
            else max(float(conf), self._yolo_confidence_threshold)
        )

        try:
            results_list = self._yolo_model.predict(
                source=VisionDetector.fill_non_clickable_wth_black(frame.copy()),
                conf=inference_conf,
                verbose=self._yolo_verbose,
            )
        except Exception as exc:  # pragma: no cover - runtime specific
            logger.error("Błąd podczas inferencji modelu YOLO WuKonga: %s", exc)
            return None

        if not results_list:
            return None

        return results_list[0]

    def _detect_with_yolo(
        self,
        frame: np.ndarray,
        resource: ResourceName,
        *,
        threshold: Optional[float] = None,
    ) -> Tuple[Tuple[int, int], ...]:
        class_id = self._get_yolo_class_id(resource)
        if class_id is None:
            return tuple()

        result = self._predict_yolo(frame, conf=threshold)
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
            self.vision.get_global_pos((int(round(x)), int(round(y))))
            for x, y in centers
        )

    def _detect_resource_positions(
        self,
        resource: ResourceName,
        *,
        frame: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> DetectionResult:
        fallback_threshold = threshold if threshold is not None else 0.85

        if frame is None:
            frame = self.vision.capture_frame()
        if frame is None:
            detection = DetectionResult(tuple(), DETECTION_METHOD_FRAME_MISSING)
            self._log_detection_result(resource, detection)
            return detection

        if self._yolo_model is not None and self._get_yolo_class_id(resource) is not None:
            detections = self._detect_with_yolo(frame, resource, threshold=threshold)
            if detections:
                detection = DetectionResult(detections, DETECTION_METHOD_YOLO)
                self._log_detection_result(resource, detection)
                return detection
            if resource.value in self.vision.target_templates:
                template_detections = self.vision.locate_all_templates(
                    resource,
                    frame=frame,
                    threshold=fallback_threshold,
                )
                if template_detections:
                    logger.debug(
                        "Model YOLO nie wykrył '%s' – korzystam z klasycznego wzorca jako wsparcia.",
                        resource.value,
                    )
                    detection = DetectionResult(template_detections, DETECTION_METHOD_TEMPLATE)
                    self._log_detection_result(resource, detection)
                    return detection
            detection = DetectionResult(tuple(), DETECTION_METHOD_YOLO)
            self._log_detection_result(resource, detection)
            return detection

        if resource.value in self.vision.target_templates:
            template_detections = self.vision.locate_all_templates(
                resource,
                frame=frame,
                threshold=fallback_threshold,
            )
            detection = DetectionResult(template_detections, DETECTION_METHOD_TEMPLATE)
            self._log_detection_result(resource, detection)
            return detection

        detection = DetectionResult(tuple(), DETECTION_METHOD_UNAVAILABLE)
        self._log_detection_result(resource, detection)
        return detection

    @staticmethod
    def _detection_method_verbose_label(method: str) -> str:
        if method == DETECTION_METHOD_YOLO:
            return "modelem YOLO"
        if method == DETECTION_METHOD_TEMPLATE:
            return "szablonem"
        return "niezidentyfikowanym detektorem"

    def _log_detection_result(
        self,
        resource: ResourceName,
        detection: DetectionResult,
    ) -> None:
        label = WUKONG_RESOURCE_LABELS.get(resource)
        if label is None:
            return

        count = len(detection.positions)
        state = (detection.method, count)
        if self._detection_log_state.get(resource) == state:
            return

        self._detection_log_state[resource] = state

        method = detection.method
        if method == DETECTION_METHOD_FRAME_MISSING:
            logger.warning("Nie udało się pobrać klatki do detekcji %s.", label)
            return

        if method == DETECTION_METHOD_UNAVAILABLE:
            logger.debug(
                "Detekcja %s jest niedostępna – brak odpowiedniego detektora.",
                label,
            )
            return

        method_verbose = self._detection_method_verbose_label(method)

        if count:
            logger.info("Wykryto %d %s (%s).", count, label, method_verbose)
        else:
            logger.debug("Nie wykryto %s (%s).", label, method_verbose)

    def _get_screen_center_global(self) -> Tuple[int, int]:
        """Return the global coordinates of the game window center."""

        return self.vision.get_global_pos(self.vision.center)

    def _make_engage_callback(
        self,
        resource: ResourceName,
        *,
        label: str,
        threshold: Optional[float] = None,
        detection_interval: float = 2.0,
        click_cooldown: float = 2.0,
    ) -> Callable[[float], None]:
        """Create a callback that keeps the character moving towards ``resource``.

        The callback relies on YOLO/template detections to find interesting
        objects on the screen.  When a detection is available, it clicks the
        closest match to the center of the screen which makes the character
        move towards the target instead of idling in place.
        """

        center_global = self._get_screen_center_global()
        state = {
            "next_attempt": 0.0,
            "last_click": 0.0,
            "last_target": None,
            "unavailable_logged": False,
            "attack_timestamp_updater": None,
        }

        def _callback(now: float) -> None:
            if now < state["next_attempt"]:
                return

            detection = self._detect_resource_positions(
                resource,
                threshold=threshold,
            )
            state["next_attempt"] = now + detection_interval

            method = detection.method
            if method == DETECTION_METHOD_UNAVAILABLE:
                if not state["unavailable_logged"]:
                    logger.debug(
                        "Automatyczne namierzanie %s niedostępne – brak detektora dla '%s'.",
                        label,
                        resource.value,
                    )
                    state["unavailable_logged"] = True
                return

            state["unavailable_logged"] = False

            positions = detection.positions
            if not positions:
                method_verbose = self._detection_method_verbose_label(method)
                logger.debug(
                    "Nie udało się zlokalizować %s podczas automatycznego namierzania (%s).",
                    label,
                    method_verbose,
                )
                return

            target = min(
                positions,
                key=lambda pos: (pos[0] - center_global[0]) ** 2 + (pos[1] - center_global[1]) ** 2,
            )
            method_verbose = self._detection_method_verbose_label(method)

            should_click = False
            if state["last_target"] != target:
                logger.info("Namierzono %s – przemieszczam się do celu (%s).", label, method_verbose)
                should_click = True
            elif now - state["last_click"] >= click_cooldown:
                logger.debug("Odświeżam ruch w kierunku %s (%s).", label, method_verbose)
                should_click = True

            if not should_click:
                return

            self.game.stop_attack()
            try:
                self.game.click_at(target)
            finally:
                self.game.start_attack(force=True)
                updater = state.get("attack_timestamp_updater")
                if updater is not None:
                    updater(now)
            state["last_target"] = target
            state["last_click"] = now

        def _set_attack_timestamp_updater(updater: Callable[[float], None]) -> None:
            state["attack_timestamp_updater"] = updater

        _callback.set_attack_timestamp_updater = _set_attack_timestamp_updater  # type: ignore[attr-defined]

        return _callback

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_text(text: str) -> str:
        return text.strip().lower()

    @staticmethod
    def _message_contains_keywords(message: str, keywords: Sequence[str]) -> bool:
        if not keywords:
            return False
        normalized = WuKongAutomation._normalize_text(message)
        return all(keyword in normalized for keyword in keywords)

    @staticmethod
    def _extract_remaining_count(message: str) -> Optional[int]:
        match = REMAINING_PATTERN.search(WuKongAutomation._normalize_text(message))
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:  # pragma: no cover - defensive
            return None

    # ------------------------------------------------------------------
    # Stage monitoring primitives
    # ------------------------------------------------------------------

    def _capture_dungeon_message(self) -> str:
        frame = self.vision.capture_frame()
        if frame is None:
            logger.warning("Brak klatki z gry – ponawiam próbę po krótkiej pauzie.")
            sleep(1.0)
            return ""
        return self.vision.get_dungeon_message(frame)

    def _wait_for_stage_prompt(self, stage: StageDefinition, timeout: float) -> None:
        if not stage.prompt_keywords:
            return

        deadline = perf_counter() + min(timeout, PROMPT_WAIT_LIMIT)
        while perf_counter() < deadline:
            message = self._capture_dungeon_message()
            if message and self._message_contains_keywords(message, stage.prompt_keywords):
                logger.info("Wykryto komunikat etapu: %s", message)
                return
            sleep(0.5)

        logger.warning(
            "Nie udało się potwierdzić komunikatu etapu '%s' w ciągu %.1fs.",
            stage.title,
            min(timeout, PROMPT_WAIT_LIMIT),
        )

    def _prepare_for_combat(self) -> None:
        if self._buffs_initialized:
            return

        logger.debug("Aktywuję premie bojowe przed pierwszym starciem.")
        self.game.use_boosters()
        self.game.toggle_passive_skills(reset_animation=False)
        self._buffs_initialized = True

    def _make_basic_combat_action(
        self,
        *,
        attack_interval: float = 1.0,
        skill_cooldowns: Optional[Dict[UserBind, float]] = None,
        lure_interval: Optional[float] = None,
        lure_key: UserBind = UserBind.MARMUREK,
        extra_callbacks: Optional[Sequence[Callable[[float], None]]] = None,
    ) -> Callable[[float], None]:
        last_attack_check = 0.0
        next_skill_use: Dict[UserBind, float] = {}
        if skill_cooldowns:
            next_skill_use = {skill: 0.0 for skill in skill_cooldowns}
        next_lure = 0.0 if lure_interval is not None else None

        def _update_attack_timestamp(timestamp: float) -> None:
            nonlocal last_attack_check
            last_attack_check = timestamp

        if extra_callbacks:
            for callback in extra_callbacks:
                setter = getattr(callback, "set_attack_timestamp_updater", None)
                if callable(setter):
                    setter(_update_attack_timestamp)

        def _action(now: float) -> None:
            nonlocal last_attack_check, next_lure

            if now - last_attack_check >= attack_interval:
                # Keep the attack key pressed – relying on ``tap_key`` would
                # release the key which stops the auto-attack.  ``force=True``
                # mirrors the behaviour from ``dung_polana.py`` where the
                # attack key press is reasserted periodically to recover from
                # any stray release events.
                self.game.start_attack(force=True)
                last_attack_check = now

            for skill, cooldown in (skill_cooldowns or {}).items():
                if now >= next_skill_use[skill]:
                    self.game.tap_key(skill)
                    next_skill_use[skill] = now + cooldown

            if lure_interval is not None and next_lure is not None and now >= next_lure:
                self.game.tap_key(lure_key)
                next_lure = now + lure_interval

            if extra_callbacks:
                for callback in extra_callbacks:
                    callback(now)

        return _action

    def _confirm_template_presence(
        self,
        resource: ResourceName,
        label: str,
        *,
        threshold: float = 0.82,
    ) -> None:
        detection = self._detect_resource_positions(
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

        if detection.positions:
            logger.info(
                "Potwierdzono obecność %s (%s; detekcje: %d).",
                label,
                self._detection_method_verbose_label(method),
                len(detection.positions),
            )
        else:
            if method == DETECTION_METHOD_YOLO:
                logger.debug("Detekcja YOLO nie odnalazła jeszcze %s.", label)
            else:
                logger.debug("Detekcja szablonowa nie odnalazła jeszcze %s.", label)

    def _make_template_presence_callback(
        self,
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

            detection = self._detect_resource_positions(
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
            method_verbose = self._detection_method_verbose_label(method)

            if state["present"] is None:
                if present:
                    logger.info("Wykryto %s (%s).", label, method_verbose)
                else:
                    if method == DETECTION_METHOD_YOLO:
                        logger.debug("Detekcja YOLO nie odnalazła jeszcze %s.", label)
                    else:
                        logger.debug("Detekcja szablonowa nie odnalazła jeszcze %s.", label)
            elif present != state["present"]:
                if present:
                    logger.info("%s ponownie widoczne (%s).", label.capitalize(), method_verbose)
                else:
                    logger.success("%s nie są już wykrywane (%s).", label.capitalize(), method_verbose)

            state["present"] = present
            state["last_check"] = now

        return _callback

    def _make_template_count_callback(
        self,
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

            detection = self._detect_resource_positions(
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

            count = len(detection.positions)
            method_verbose = self._detection_method_verbose_label(method)

            if state["count"] is None:
                if count:
                    logger.info("Wykryto %d %s (%s).", count, label, method_verbose)
                else:
                    if method == DETECTION_METHOD_YOLO:
                        logger.debug("Detekcja YOLO nie odnalazła jeszcze %s.", label)
                    else:
                        logger.debug("Detekcja szablonowa nie odnalazła jeszcze %s.", label)
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
        self,
        stage: StageDefinition,
        timeout: float,
        *,
        action_callback: Optional[Callable[[float], None]] = None,
        progress_parser: Optional[Callable[[str], Optional[int]]] = None,
        completion_keywords: Optional[Sequence[str]] = None,
        prompt_already_confirmed: bool = False,
    ) -> float:
        stage_start = perf_counter()
        if not prompt_already_confirmed:
            self._wait_for_stage_prompt(stage, timeout)

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

            message = self._capture_dungeon_message()
            if message:
                if message != last_message:
                    logger.debug("Komunikat lochów: %s", message)
                    last_message = message

                if self._message_contains_keywords(message, stage.prompt_keywords):
                    prompt_seen = True
                    if progress_parser is not None:
                        remaining = progress_parser(message)
                        if remaining is not None and remaining != last_progress:
                            logger.info("Pozostało: %s", remaining)
                            last_progress = remaining
                elif prompt_seen:
                    if completion_keywords and self._message_contains_keywords(message, completion_keywords):
                        logger.info("Wykryto komunikat kolejnego etapu: %s", message)
                    else:
                        logger.info("Komunikat etapu uległ zmianie: %s", message)
                    elapsed = now - stage_start
                    logger.success("Etap '%s' ukończony w %.1fs.", stage.title, elapsed)
                    return elapsed

            sleep(0.5)

    def _use_phoenix_eggs(self) -> None:
        logger.info("Otwieram ekwipunek w poszukiwaniu jaj feniksa.")
        self.game.tap_key(GameBind.EQ_MENU)
        sleep(0.4)

        frame = self.vision.capture_frame()
        detected_positions: Tuple[Tuple[int, int], ...] = tuple()

        if frame is None:
            logger.warning("Nie udało się pobrać klatki ekranu – używam pozycji zapasowych.")
        else:
            detection = self._detect_resource_positions(
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
                    self._detection_method_verbose_label(method),
                )
            else:
                if method == DETECTION_METHOD_UNAVAILABLE:
                    logger.debug(
                        "Brak skonfigurowanego detektora jaj feniksa – korzystam z zapasowych współrzędnych."
                    )
                elif method == DETECTION_METHOD_YOLO:
                    logger.debug(
                        "Detekcja YOLO nie znalazła jaj feniksa – używam pozycji zapasowych."
                    )
                else:
                    logger.debug(
                        "Detekcja szablonowa nie znalazła jaj feniksa – używam pozycji zapasowych."
                    )

        if not detected_positions:
            if not self.config.egg_slots:
                logger.warning(
                    "Brak skonfigurowanych slotów jaj feniksa – oczekuję na ręczne użycie przedmiotu."
                )
                fallback_positions: Tuple[Tuple[int, int], ...] = tuple()
            else:
                fallback_positions = tuple(
                    self.vision.get_global_pos(slot) for slot in self.config.egg_slots
                )
                logger.info(
                    "Nie znaleziono jaj poprzez automatyczną detekcję – klikam %d zaprogramowanych slotów.",
                    len(fallback_positions),
                )
            detected_positions = fallback_positions

        unique_positions = list(dict.fromkeys(detected_positions))
        for pos in unique_positions:
            self.game.click_at(pos, right=True)
            sleep(0.2)

        self.game.tap_key(GameBind.EQ_MENU)
        sleep(0.3)

    def _log_stage_banner(self, stage_index: int, stage: StageDefinition, timeout: float) -> None:
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

    def _handle_generic_stage(self, stage: StageDefinition, timeout: float) -> float:
        logger.warning(
            "Brak dedykowanego handlera dla etapu '%s' – działanie ograniczone do monitoringu.",
            stage.title,
        )
        return self._monitor_stage(
            stage,
            timeout,
            action_callback=None,
            completion_keywords=stage.completion_keywords,
            progress_parser=self._extract_remaining_count,
        )

    def _run_combat_stage(
        self,
        stage: StageDefinition,
        timeout: float,
        *,
        attack_interval: float = 1.0,
        skill_cooldowns: Optional[Dict[UserBind, float]] = None,
        lure_interval: Optional[float] = None,
        extra_callbacks: Optional[Sequence[Callable[[float], None]]] = None,
        progress_parser: Optional[Callable[[str], Optional[int]]] = None,
        prompt_confirmed: bool = False,
    ) -> float:
        self._prepare_for_combat()
        self.game.start_attack()
        action = self._make_basic_combat_action(
            attack_interval=attack_interval,
            skill_cooldowns=skill_cooldowns,
            lure_interval=lure_interval,
            extra_callbacks=extra_callbacks,
        )

        try:
            return self._monitor_stage(
                stage,
                timeout,
                action_callback=action,
                completion_keywords=stage.completion_keywords,
                progress_parser=progress_parser,
                prompt_already_confirmed=prompt_confirmed,
            )
        finally:
            self.game.stop_attack()

    def _execute_restart_sequence(self) -> bool:
        frame = self.vision.capture_frame()
        start_pos = self.vision.locate_template(ResourceName.WUKONG_RESTART, frame=frame, threshold=0.8)
        fallback_start = (
            self.vision.get_global_pos(self.config.restart_button)
            if self.config.restart_button is not None
            else None
        )

        if start_pos is not None:
            logger.info("Wzorzec przycisku restartu odnaleziony – klikam.")
        elif fallback_start is not None:
            logger.info(
                "Nie znaleziono wzorca restartu – klikam zapasowe współrzędne %s.",
                self.config.restart_button,
            )
            start_pos = fallback_start
        else:
            logger.warning(
                "Brak szablonu i zapasowych współrzędnych przycisku restartu – oczekuję na kliknięcie ręczne."
            )
            return False

        self.game.click_at(start_pos)
        sleep(0.6)

        frame = self.vision.capture_frame()
        confirm_pos = self.vision.locate_template(ResourceName.WUKONG_RESTART_CONFIRM, frame=frame, threshold=0.8)
        fallback_confirm = (
            self.vision.get_global_pos(self.config.restart_confirm_button)
            if self.config.restart_confirm_button is not None
            else None
        )

        if confirm_pos is not None:
            logger.info("Znalazłem przycisk potwierdzenia restartu – klikam.")
        elif fallback_confirm is not None:
            logger.info(
                "Brak wzorca potwierdzenia – klikam zapasowe współrzędne %s.",
                self.config.restart_confirm_button,
            )
            confirm_pos = fallback_confirm
        else:
            logger.warning("Nie mogę potwierdzić restartu – brak współrzędnych zapasowych.")
            return False

        self.game.click_at(confirm_pos)
        sleep(0.5)
        return True

    # ------------------------------------------------------------------
    # Stage specific handlers
    # ------------------------------------------------------------------

    def _handle_slay_first_wave(self, stage: StageDefinition, timeout: float) -> float:
        mob_engage = self._make_engage_callback(
            ResourceName.WUKONG_MOB,
            label="przeciwników WuKonga",
        )
        return self._run_combat_stage(
            stage,
            timeout,
            attack_interval=0.8,
            skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
            extra_callbacks=(mob_engage,),
            progress_parser=self._extract_remaining_count,
        )

    def _handle_destroy_first_metins(self, stage: StageDefinition, timeout: float) -> float:
        self._confirm_template_presence(
            ResourceName.WUKONG_METIN,
            "kamieni Metin WuKonga",
        )
        metin_tracker = self._make_template_count_callback(
            ResourceName.WUKONG_METIN,
            label="kamieni Metin WuKonga",
        )
        metin_engage = self._make_engage_callback(
            ResourceName.WUKONG_METIN,
            label="kamieni Metin WuKonga",
        )
        return self._run_combat_stage(
            stage,
            timeout,
            attack_interval=0.9,
            skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
            extra_callbacks=(metin_tracker, metin_engage),
            progress_parser=self._extract_remaining_count,
        )

    def _handle_clear_second_wave(self, stage: StageDefinition, timeout: float) -> float:
        self._confirm_template_presence(
            ResourceName.WUKONG_MOB,
            "mobów drugiej fali WuKonga",
        )
        mob_tracker = self._make_template_presence_callback(
            ResourceName.WUKONG_MOB,
            label="mobów drugiej fali WuKonga",
        )
        mob_engage = self._make_engage_callback(
            ResourceName.WUKONG_MOB,
            label="mobów drugiej fali WuKonga",
        )
        return self._run_combat_stage(
            stage,
            timeout,
            attack_interval=0.8,
            skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
            extra_callbacks=(mob_tracker, mob_engage),
        )

    def _handle_defeat_cloud_guardian(self, stage: StageDefinition, timeout: float) -> float:
        self._confirm_template_presence(
            ResourceName.WUKONG_CLOUD_GUARDIAN,
            "Obrońcę Chmur WuKonga",
        )
        guardian_tracker = self._make_template_presence_callback(
            ResourceName.WUKONG_CLOUD_GUARDIAN,
            label="Obrońcę Chmur WuKonga",
        )
        guardian_engage = self._make_engage_callback(
            ResourceName.WUKONG_CLOUD_GUARDIAN,
            label="Obrońcę Chmur WuKonga",
        )
        return self._run_combat_stage(
            stage,
            timeout,
            attack_interval=0.7,
            skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
            extra_callbacks=(guardian_tracker, guardian_engage),
        )

    def _handle_place_phoenix_eggs(self, stage: StageDefinition, timeout: float) -> float:
        last_egg_usage = {"time": 0.0}

        def _egg_callback(now: float) -> None:
            if now - last_egg_usage["time"] < 8.0:
                return
            self._use_phoenix_eggs()
            last_egg_usage["time"] = now

        mob_engage = self._make_engage_callback(
            ResourceName.WUKONG_MOB,
            label="przeciwników WuKonga",
        )
        return self._run_combat_stage(
            stage,
            timeout,
            attack_interval=0.8,
            skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
            extra_callbacks=(_egg_callback, mob_engage),
            progress_parser=self._extract_remaining_count,
        )

    def _handle_destroy_phoenix_eggs(self, stage: StageDefinition, timeout: float) -> float:
        egg_engage = self._make_engage_callback(
            ResourceName.WUKONG_PHOENIX_EGG,
            label="jaj Feniksa WuKonga",
        )
        mob_engage = self._make_engage_callback(
            ResourceName.WUKONG_MOB,
            label="przeciwników WuKonga",
        )
        return self._run_combat_stage(
            stage,
            timeout,
            attack_interval=0.75,
            skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
            extra_callbacks=(egg_engage, mob_engage),
            progress_parser=self._extract_remaining_count,
        )

    def _handle_repel_three_waves(self, stage: StageDefinition, timeout: float) -> float:
        logger.info("Aktywuję pelerynę (F4), aby przyspieszyć pojawianie się fal.")
        self.game.tap_key(UserBind.MARMUREK)

        self._confirm_template_presence(
            ResourceName.WUKONG_MOB,
            "przeciwników w falach WuKonga",
        )
        mob_tracker = self._make_template_presence_callback(
            ResourceName.WUKONG_MOB,
            label="przeciwników w falach WuKonga",
        )
        mob_engage = self._make_engage_callback(
            ResourceName.WUKONG_MOB,
            label="przeciwników w falach WuKonga",
        )
        return self._run_combat_stage(
            stage,
            timeout,
            attack_interval=0.85,
            skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
            lure_interval=25.0,
            progress_parser=self._extract_remaining_count,
            extra_callbacks=(mob_tracker, mob_engage),
        )

    def _handle_destroy_second_metin(self, stage: StageDefinition, timeout: float) -> float:
        self._confirm_template_presence(
            ResourceName.WUKONG_METIN,
            "ostatnich kamieni Metin WuKonga",
        )
        metin_tracker = self._make_template_count_callback(
            ResourceName.WUKONG_METIN,
            label="kamieni Metin WuKonga",
        )
        metin_engage = self._make_engage_callback(
            ResourceName.WUKONG_METIN,
            label="kamieni Metin WuKonga",
        )
        return self._run_combat_stage(
            stage,
            timeout,
            attack_interval=0.9,
            skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
            progress_parser=self._extract_remaining_count,
            extra_callbacks=(metin_tracker, metin_engage),
        )

    def _handle_defeat_flaming_phoenix(self, stage: StageDefinition, timeout: float) -> float:
        self._confirm_template_presence(
            ResourceName.WUKONG_FLAMING_PHOENIX,
            "Płomiennego Feniksa WuKonga",
        )
        phoenix_tracker = self._make_template_presence_callback(
            ResourceName.WUKONG_FLAMING_PHOENIX,
            label="Płomiennego Feniksa WuKonga",
        )
        phoenix_engage = self._make_engage_callback(
            ResourceName.WUKONG_FLAMING_PHOENIX,
            label="Płomiennego Feniksa WuKonga",
        )
        return self._run_combat_stage(
            stage,
            timeout,
            attack_interval=0.7,
            skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
            extra_callbacks=(phoenix_tracker, phoenix_engage),
        )

    def _handle_clear_final_wave(self, stage: StageDefinition, timeout: float) -> float:
        self._confirm_template_presence(
            ResourceName.WUKONG_MOB,
            "ostatnich przeciwników WuKonga",
        )
        mob_tracker = self._make_template_presence_callback(
            ResourceName.WUKONG_MOB,
            label="ostatnich przeciwników WuKonga",
        )
        mob_engage = self._make_engage_callback(
            ResourceName.WUKONG_MOB,
            label="ostatnich przeciwników WuKonga",
        )
        return self._run_combat_stage(
            stage,
            timeout,
            attack_interval=0.8,
            skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
            extra_callbacks=(mob_tracker, mob_engage),
        )

    def _handle_destroy_crimson_gourds(self, stage: StageDefinition, timeout: float) -> float:
        self._confirm_template_presence(
            ResourceName.WUKONG_CRIMSON_GOURD,
            "Karmazynowych Gurd",
        )
        gourd_tracker = self._make_template_count_callback(
            ResourceName.WUKONG_CRIMSON_GOURD,
            label="Karmazynowych Gurd",
        )
        gourd_engage = self._make_engage_callback(
            ResourceName.WUKONG_CRIMSON_GOURD,
            label="Karmazynowych Gurd",
        )
        return self._run_combat_stage(
            stage,
            timeout,
            attack_interval=0.85,
            skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
            progress_parser=self._extract_remaining_count,
            extra_callbacks=(gourd_tracker, gourd_engage),
        )

    def _handle_defeat_wukong(self, stage: StageDefinition, timeout: float) -> float:
        self._confirm_template_presence(
            ResourceName.WUKONG_MONKEY_KING,
            "Małpiego Króla WuKonga",
        )
        boss_tracker = self._make_template_presence_callback(
            ResourceName.WUKONG_MONKEY_KING,
            label="Małpiego Króla WuKonga",
        )
        boss_engage = self._make_engage_callback(
            ResourceName.WUKONG_MONKEY_KING,
            label="Małpiego Króla WuKonga",
        )
        return self._run_combat_stage(
            stage,
            timeout,
            attack_interval=0.65,
            skill_cooldowns=DEFAULT_SKILL_COOLDOWNS,
            extra_callbacks=(boss_tracker, boss_engage),
        )

    def _handle_restart_expedition(self, stage: StageDefinition, timeout: float) -> float:
        self._wait_for_stage_prompt(stage, timeout)
        if self._execute_restart_sequence():
            logger.success("Próba restartu wyprawy zakończona powodzeniem.")
        else:
            logger.warning("Restart wyprawy wymaga ręcznej interwencji.")
        return self._monitor_stage(
            stage,
            timeout,
            action_callback=None,
            completion_keywords=stage.completion_keywords,
            progress_parser=self._extract_remaining_count,
            prompt_already_confirmed=True,
        )

    @property
    def last_stage_durations(self) -> Tuple[Tuple[str, float], ...]:
        """Return a snapshot of the durations recorded during the last run."""

        return tuple(self._last_stage_durations)


def run(
    stage: int,
    log_level: str,
    saved_credentials_idx: int,
    stage_timeouts: Sequence[float] | str = DEFAULT_STAGE_TIMEOUTS,
    yolo_confidence_threshold: float = DEFAULT_YOLO_CONFIDENCE_THRESHOLD,
    yolo_device: str = DEFAULT_YOLO_DEVICE,
) -> None:
    automation = WuKongAutomation(
        log_level=log_level,
        saved_credentials_idx=saved_credentials_idx,
        stage_timeouts=stage_timeouts,
        yolo_confidence_threshold=yolo_confidence_threshold,
        yolo_device=yolo_device,
    )
    automation.run(stage)


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
