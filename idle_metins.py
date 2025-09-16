from pathlib import Path
from time import perf_counter, sleep

import click
import numpy as np
from loguru import logger
from torch import where as torch_where
from ultralytics import YOLO
from ultralytics import checks as yolo_checks

from game_controller import GameController, Key
from settings import MODELS_DIR, UserBind
from utils import channel_generator, setup_logger
from vision_detector import VisionDetector


def _allowlist_ultralytics_serialization_types():
    """Allow torch.load(weights_only=True) to deserialize Ultralytics + torch.nn modules."""

    try:
        from torch.serialization import add_safe_globals
    except ImportError:  # pragma: no cover - fallback for very old torch versions
        return

    import importlib
    import inspect
    import pkgutil
    from types import ModuleType

    safe_types = set()

    def register_package_types(root_module, package_prefix):
        if root_module is None:
            return

        visited_modules = set()

        def register(module):
            if module in visited_modules:
                return
            visited_modules.add(module)

            for attr in vars(module).values():
                if inspect.isclass(attr):
                    safe_types.add(attr)
                elif isinstance(attr, ModuleType):
                    module_name = getattr(attr, "__name__", "")
                    if module_name.startswith(package_prefix):
                        register(attr)

        register(root_module)

        if hasattr(root_module, "__path__"):
            prefix = root_module.__name__ + "."
            for module_info in pkgutil.walk_packages(root_module.__path__, prefix):
                try:
                    submodule = importlib.import_module(module_info.name)
                except Exception as exc:  # pragma: no cover - best effort: skip problematic modules
                    logger.debug(
                        f"Failed to import {module_info.name} while registering safe globals: {exc}"
                    )
                    continue

                register(submodule)

    try:
        from torch.nn.modules.container import Sequential

        safe_types.add(Sequential)
    except ImportError:  # pragma: no cover - defensive; Sequential is available in torch
        pass

    try:
        from torch.nn import modules as torch_nn_modules
    except ImportError:  # pragma: no cover - defensive: torch should always provide nn.modules
        torch_nn_modules = None
    register_package_types(torch_nn_modules, "torch.nn.modules")

    try:
        from ultralytics.nn.tasks import DetectionModel

        safe_types.add(DetectionModel)
    except ImportError:  # pragma: no cover - ultralytics not installed
        pass

    try:
        from ultralytics.nn import modules as yolo_modules
    except ImportError:  # pragma: no cover - ultralytics not installed
        yolo_modules = None
    register_package_types(yolo_modules, "ultralytics.nn")

    try:
        from ultralytics.nn import tasks as yolo_tasks
    except ImportError:  # pragma: no cover - ultralytics not installed
        yolo_tasks = None
    register_package_types(yolo_tasks, "ultralytics.nn")

    add_safe_globals(list(safe_types))


@click.command()
@click.option("--event", is_flag=True, help="Enable event mode (+2 metins)")
@click.option("--log-level",
              default="INFO",
              show_default=True,
              type=click.Choice(["TRACE", "DEBUG", "INFO"], case_sensitive=False),
              help="Set the logging level."
)
@click.option("--start", default=1, show_default=True, type=click.IntRange(1, 8), help="Start channel.")
@click.option("--saved_credentials_idx", default=1, type=int, show_default=True, help="Saved credentials index to use.")
def main(event, log_level, start, saved_credentials_idx):
    log_level = log_level.upper()
    yolo_checks()
    setup_logger(script_name=Path(__file__).name, level=log_level)
    run(event, log_level, start, saved_credentials_idx)


def run(event, log_level, start, saved_credentials_idx):
    _allowlist_ultralytics_serialization_types()
    yolo = YOLO(MODELS_DIR / "valium_idle_metiny_yolov8s.pt").to("cuda:0")
    yolo_verbose = log_level in ["TRACE", "DEBUG"]
    logger.info("YOLO model loaded.")

    vision = VisionDetector()
    logger.info("Vision detector loaded.")

    game = GameController(vision_detector=vision, start_delay=2, saved_credentials_idx=saved_credentials_idx)
    logger.info("Game controller loaded.")

    channel_gen = channel_generator(1, 8, start=start, step=3 if event else 1)

    YOLO_CONFIDENCE_THRESHOLD = 0.75
    CHANNEL_TIMEOUT = 20
    LOOKING_AROUND_MOVE_CAMERA_PRESS_TIME = 0.5
    WALK_TO_METIN_TIME = 1.25

    # METIN_CLS = 0  # smierci sohan
    # METIN_DESTROY_TIME = 8  # smierci sohan | poly + masne eq + IS

    METIN_CLS = 1  # upadku polana
    METIN_DESTROY_TIME = 1  # upadku polana | poly + masne eq + IS

    assert isinstance(METIN_CLS, int), "METIN_CLS must be an integer."

    game.calibrate_camera()
    game.move_camera_down(press_time=0.7)
    
    if event:
        idle_time = METIN_DESTROY_TIME + 0.25
    else:
        idle_time = METIN_DESTROY_TIME

    while game.is_running:
        channel = next(channel_gen)
        game.change_to_channel(channel, wait_after_change=1)

        t0 = perf_counter()
        metin_detected = False
        timed_out = False
        while not metin_detected:
            frame = vision.capture_frame()
            if frame is None:
                game.restart_game()
                continue

            if vision.logged_out(frame):
                logger.warning("Logged out. Re-logging...")
                game.login()
                sleep(3)  # wait out relogging blockage
                game.change_to_channel(next(channel_gen))  # change channel to reset dungeon map; handles the infinite relogging loop
                game.calibrate_camera()
                game.move_camera_down(press_time=0.7)
                continue

            if vision.is_loading(frame=vision.capture_frame()):
                sleep(10)
                if vision.is_loading(frame=vision.capture_frame()):
                    logger.warning("Loading is taking too long (>10s), something is wrong. Escaping to logging menu...")
                    game.tap_key(Key.esc, press_time=2)
                    sleep(5)
                    continue
                continue

            _, polymorphed = vision.is_polymorphed(frame=vision.capture_frame())
            if not polymorphed:
                game.toggle_skill(UserBind.AURA, reset_animation=False)
                game.use_polymorph()

            if perf_counter() - t0 > CHANNEL_TIMEOUT:
                timed_out = True
                logger.warning(f"Timeout ({CHANNEL_TIMEOUT}s). Switching to the next channel...")
                break

            latest_frame = vision.capture_frame()
            yolo_results = yolo.predict(
                source=VisionDetector.fill_non_clickable_wth_black(latest_frame),
                conf=YOLO_CONFIDENCE_THRESHOLD,
                verbose=yolo_verbose
            )[0]
            any_yolo_results = len(yolo_results.boxes.cls) > 0
            metins_idxs = torch_where(yolo_results.boxes.cls == METIN_CLS)
            metin_detected = metins_idxs[0].shape[0] > 0
            logger.debug(f"{metins_idxs=} {metin_detected=}")
            if not metin_detected:
                game.move_camera_right(press_time=LOOKING_AROUND_MOVE_CAMERA_PRESS_TIME)
                logger.warning(f"Metin not found. Looking around, retrying...")
                continue
        
        if timed_out and not metin_detected:
            continue

        metins_xywh = yolo_results.boxes.xywh[metins_idxs].cpu()
        metins_distance_to_center = np.linalg.norm(metins_xywh[:, :2] - np.array(vision.center), axis=1)
        closest_metin_idx = metins_distance_to_center.argmin()
        closest_metin_bbox_xywh = yolo_results.boxes.xywh[closest_metin_idx]
        closest_metin_bbox_center = closest_metin_bbox_xywh[:2]
        closest_metin_center_global = vision.get_global_pos(closest_metin_bbox_center)

        game.use_boosters()
        game.click_at(closest_metin_center_global)

        sleep(WALK_TO_METIN_TIME)

        game.start_attack()
        game.idle(idle_time, pickup=True)
        game.pickup()
        game.pickup()
        if event:
            game.pickup()
        game.stop_attack()

        butelka_dywizji_filled = vision.detect_butelka_dywizji_filled_message(frame=vision.capture_frame())
        if butelka_dywizji_filled:
            game.move_full_butelka_dywizji()
            game.use_next_butelka_dywizji()


if __name__ == '__main__':
    main()
    logger.success("Bot terminated.")
