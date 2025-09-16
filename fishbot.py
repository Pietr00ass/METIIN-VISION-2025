import sys
from datetime import timedelta
from pathlib import Path
from random import choice
from time import perf_counter, sleep
from typing import Sequence, Tuple
from warnings import filterwarnings

import click
import cv2
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from torch import where as torch_where
from ultralytics import YOLO
from ultralytics import checks as yolo_checks

from game_controller import GameController, Key
from settings import CAP_MAX_FPS, MODELS_DIR, WINDOW_HEIGHT, GameBind, UserBind
from utils import channel_generator, setup_logger
from vision_detector import VisionDetector


@click.command()
@click.option(
    "--stage", default=0, type=int, show_default=True, help="Stage to start from."
)
@click.option(
    "--log-level",
    default="TRACE",
    show_default=True,
    type=click.Choice(["TRACE", "DEBUG", "INFO"], case_sensitive=False),
    help="Set the logging level.",
)
@click.option(
    "--saved_credentials_idx",
    default=1,
    type=int,
    show_default=True,
    help="Saved credentials index to use.",
)
def main(stage, log_level, saved_credentials_idx):
    log_level = log_level.upper()
    setup_logger(script_name=Path(__file__).name, level=log_level)
    logger.warning("Starting the bot...")
    run(stage, log_level, saved_credentials_idx)


def _parse_window(window: Sequence[int] | str) -> Tuple[int, int, int, int]:
    """Parse window specification into (top, bottom, left, right)."""

    if isinstance(window, str):
        normalized = window.replace(" ", "")
        normalized = normalized.replace(";", ",")
        normalized = normalized.replace(":", ",")
        parts = [part for part in normalized.split(",") if part]
        if len(parts) != 4:
            raise ValueError(
                "Parametr fishing_window musi zawierać cztery wartości: górę, dół, lewą i prawą krawędź w pikselach."
            )
        try:
            top, bottom, left, right = map(int, parts)
        except ValueError as exc:
            raise ValueError(
                "Nie udało się sparsować fishing_window – wpisz liczby całkowite oddzielone przecinkami."
            ) from exc
    else:
        if len(tuple(window)) != 4:
            raise ValueError("fishing_window musi mieć cztery elementy.")
        top, bottom, left, right = map(int, window)

    if bottom <= top:
        raise ValueError("Dolna krawędź musi być większa od górnej w parametrze fishing_window.")
    if right <= left:
        raise ValueError("Prawa krawędź musi być większa od lewej w parametrze fishing_window.")

    return top, bottom, left, right


def run(
    stage,
    log_level,
    saved_credentials_idx,
    yolo_confidence_threshold: float = 0.95,
    fishing_window: Sequence[int] | str = (77, 304, 101, 379),
):
    yolo_checks()
    yolo_verbose = log_level in ["TRACE", "DEBUG"]
    yolo = YOLO(MODELS_DIR / "global_fishbot_yolov8s.pt").to("cuda:0")

    vision = VisionDetector()
    game = GameController(vision_detector=vision, start_delay=2, saved_credentials_idx=saved_credentials_idx)

    YOLO_CONFIDENCE_THRESHOLD = float(yolo_confidence_threshold)
    FISH_CLS = 0

    top, bottom, left, right = _parse_window(fishing_window)

    while game.is_running:
        frame = vision.capture_frame()

        fishing_view = frame[top:bottom, left:right]

        yolo_results = yolo.predict(
            source=fishing_view, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=yolo_verbose
        )[0]
        fish_detected = len(yolo_results.boxes.cls) > 0
        logger.debug(f"{fish_detected=}")

        if not fish_detected:
            logger.info("No fish detected.")
            sleep(0.1)
            continue

        fish_bbox_xywh = yolo_results.boxes.xywh[0]
        fish_bbox_center = fish_bbox_xywh[:2]

        fish_bbox_center_fixed = fish_bbox_center.cpu() + np.array([top, left])

        fish_bbox_center_global = vision.get_global_pos(fish_bbox_center_fixed)

        game.catch_fish(fish_bbox_center_global)

    if game.is_running:
        game.exit()
        exit()


if __name__ == "__main__":
    main()
    logger.success("Bot terminated.")
