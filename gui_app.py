"""GUI application for controlling bot modes and editing settings."""

from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any, Callable, Dict, List, Optional

from loguru import logger
from PyQt6 import QtCore, QtWidgets
from pynput.keyboard import Key

import dung_polana
import fishbot
import idle_metins
import settings
from settings import BotBind, GameBind, UserBind
from utils import setup_logger


CONFIG_FILE = Path("config.json")
LOG_FORMAT = "{time:HH:mm:ss} | {level:<8} | {message}"


class QtLogHandler(QtCore.QObject):
    """Bridge loguru logs to the GUI."""

    log_signal = QtCore.pyqtSignal(str)

    def emit(self, message: str) -> None:
        text = message.rstrip()
        if text:
            self.log_signal.emit(text)


class BotSignals(QtCore.QObject):
    """Signals emitted by :class:`BotRunner`."""

    started = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str, str)


@dataclass
class BotParameter:
    name: str
    label: str
    type: str
    default: Any = None
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    choices: Optional[List[str]] = None


@dataclass
class BotConfig:
    identifier: str
    title: str
    target: Callable[..., None]
    script_name: str
    parameters: List[BotParameter] = field(default_factory=list)


class BotRunner(QtCore.QObject):
    """Runs a bot in a background daemon thread."""

    def __init__(
        self,
        config: BotConfig,
        log_sink: Optional[Callable[[str], None]] = None,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self._thread: Optional[threading.Thread] = None
        self._log_sink = log_sink
        self.signals = BotSignals()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, parameters: Dict[str, Any]) -> None:
        if self.is_running():
            raise RuntimeError(f"Bot '{self.config.identifier}' is already running.")

        # copy to avoid accidental modifications by the UI thread
        params = dict(parameters)

        self._thread = threading.Thread(
            target=self._run,
            args=(params,),
            daemon=True,
            name=f"{self.config.identifier}_runner",
        )
        self._thread.start()

    def _run(self, params: Dict[str, Any]) -> None:
        self.signals.started.emit(self.config.identifier)

        log_level = str(params.get("log_level", "INFO")).upper()
        try:
            setup_logger(script_name=self.config.script_name, level=log_level)
            if self._log_sink is not None:
                logger.add(
                    self._log_sink,
                    format=LOG_FORMAT,
                    level="TRACE",
                    enqueue=True,
                )
            logger.info("Starting bot '{}' with parameters: {}", self.config.identifier, params)
            self.config.target(**params)
            logger.success("Bot '{}' finished successfully.", self.config.identifier)
            self.signals.finished.emit(self.config.identifier)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Bot '{}' crashed:", self.config.identifier)
            self.signals.failed.emit(self.config.identifier, str(exc))


class BotControlWidget(QtWidgets.QGroupBox):
    """UI control for a single bot configuration."""

    start_requested = QtCore.pyqtSignal(str, dict)

    def __init__(self, config: BotConfig, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(config.title, parent)
        self.config = config
        self.inputs: Dict[str, QtWidgets.QWidget] = {}
        self.param_types: Dict[str, str] = {}
        self.status_label = QtWidgets.QLabel("Nieaktywny")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout()
        form_layout = QtWidgets.QFormLayout()

        for param in self.config.parameters:
            widget: QtWidgets.QWidget
            if param.type == "int":
                spin = QtWidgets.QSpinBox()
                spin.setMinimum(param.minimum if param.minimum is not None else -10_000)
                spin.setMaximum(param.maximum if param.maximum is not None else 10_000)
                if param.default is not None:
                    spin.setValue(int(param.default))
                widget = spin
            elif param.type == "bool":
                checkbox = QtWidgets.QCheckBox()
                checkbox.setChecked(bool(param.default))
                widget = checkbox
            elif param.type == "choice":
                combo = QtWidgets.QComboBox()
                combo.addItems(param.choices or [])
                if param.default is not None:
                    index = combo.findText(str(param.default))
                    if index >= 0:
                        combo.setCurrentIndex(index)
                widget = combo
            else:
                line = QtWidgets.QLineEdit()
                if param.default is not None:
                    line.setText(str(param.default))
                widget = line

            self.inputs[param.name] = widget
            self.param_types[param.name] = param.type
            form_layout.addRow(param.label + ":", widget)

        layout.addLayout(form_layout)

        buttons_layout = QtWidgets.QHBoxLayout()
        start_button = QtWidgets.QPushButton("Uruchom")
        start_button.clicked.connect(self._emit_start)
        buttons_layout.addWidget(start_button)
        buttons_layout.addStretch(1)
        buttons_layout.addWidget(QtWidgets.QLabel("Status:"))
        buttons_layout.addWidget(self.status_label)

        layout.addLayout(buttons_layout)
        self.setLayout(layout)

    def _emit_start(self) -> None:
        self.start_requested.emit(self.config.identifier, self.parameters())

    def parameters(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for name, widget in self.inputs.items():
            param_type = self.param_types[name]
            if param_type == "int":
                assert isinstance(widget, QtWidgets.QSpinBox)
                params[name] = widget.value()
            elif param_type == "bool":
                assert isinstance(widget, QtWidgets.QCheckBox)
                params[name] = widget.isChecked()
            elif param_type == "choice":
                assert isinstance(widget, QtWidgets.QComboBox)
                params[name] = widget.currentText()
            else:
                assert isinstance(widget, QtWidgets.QLineEdit)
                params[name] = widget.text()
        return params

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)


class PathInput(QtWidgets.QWidget):
    """Composite widget allowing to edit and browse filesystem paths."""

    def __init__(
        self,
        initial: PurePath | str,
        select_directory: bool,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._select_directory = select_directory

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._line_edit = QtWidgets.QLineEdit(str(initial))
        browse_button = QtWidgets.QPushButton("Przeglądaj…")
        browse_button.clicked.connect(self._browse)

        layout.addWidget(self._line_edit)
        layout.addWidget(browse_button)

    def text(self) -> str:
        return self._line_edit.text()

    def setText(self, text: str) -> None:  # noqa: N802 (Qt naming convention)
        self._line_edit.setText(text)

    def _browse(self) -> None:
        current_text = self._line_edit.text()
        start_path = current_text if current_text else str(Path.cwd())

        if self._select_directory:
            directory = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Wybierz folder",
                start_path,
            )
            if directory:
                self._line_edit.setText(directory)
            return

        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Wybierz plik",
            start_path,
            "Wszystkie pliki (*)",
        )
        if file_path:
            self._line_edit.setText(file_path)


class SettingsPanel(QtWidgets.QWidget):
    """Panel for editing values defined in :mod:`settings`."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.scalar_inputs: Dict[str, QtWidgets.QWidget] = {}
        self.scalar_types: Dict[str, type] = {}
        self.enum_tables: Dict[str, QtWidgets.QTableWidget] = {}
        self._build_ui()
        self.refresh_from_settings()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout()

        general_group = QtWidgets.QGroupBox("Parametry ogólne")
        general_form = QtWidgets.QFormLayout()

        for name in sorted(dir(settings)):
            if not name.isupper():
                continue
            value = getattr(settings, name)
            if isinstance(value, (int, float, str, bool, PurePath)):
                widget: QtWidgets.QWidget
                if isinstance(value, bool):
                    checkbox = QtWidgets.QCheckBox()
                    widget = checkbox
                elif isinstance(value, PurePath):
                    widget = PathInput(value, self._is_directory_path(name, value))
                else:
                    line_edit = QtWidgets.QLineEdit()
                    widget = line_edit
                self.scalar_inputs[name] = widget
                self.scalar_types[name] = type(value)
                general_form.addRow(name + ":", widget)

        general_group.setLayout(general_form)
        layout.addWidget(general_group)

        for enum_cls, title in (
            (GameBind, "GameBind"),
            (UserBind, "UserBind"),
            (BotBind, "BotBind"),
        ):
            group = QtWidgets.QGroupBox(f"Bindy {title}")
            table = QtWidgets.QTableWidget(len(list(enum_cls)), 2)
            table.setHorizontalHeaderLabels(["Akcja", "Wartość"])
            table.horizontalHeader().setStretchLastSection(True)
            table.verticalHeader().setVisible(False)
            table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.AllEditTriggers)
            for row, member in enumerate(enum_cls):
                name_item = QtWidgets.QTableWidgetItem(member.name)
                name_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
                table.setItem(row, 0, name_item)
                value_item = QtWidgets.QTableWidgetItem(self._format_enum_value(member.value))
                table.setItem(row, 1, value_item)
            group_layout = QtWidgets.QVBoxLayout()
            group_layout.addWidget(table)
            group.setLayout(group_layout)
            layout.addWidget(group)
            self.enum_tables[enum_cls.__name__] = table

        buttons_layout = QtWidgets.QHBoxLayout()
        apply_button = QtWidgets.QPushButton("Zastosuj")
        apply_button.clicked.connect(self.apply_changes)
        save_button = QtWidgets.QPushButton("Zapisz konfigurację")
        save_button.clicked.connect(self.save_to_file)
        load_button = QtWidgets.QPushButton("Wczytaj konfigurację")
        load_button.clicked.connect(self.load_from_file)
        buttons_layout.addWidget(apply_button)
        buttons_layout.addWidget(save_button)
        buttons_layout.addWidget(load_button)
        buttons_layout.addStretch(1)

        layout.addLayout(buttons_layout)
        layout.addStretch(1)
        self.setLayout(layout)

    def refresh_from_settings(self) -> None:
        for name, widget in self.scalar_inputs.items():
            value = getattr(settings, name)
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, PathInput):
                widget.setText(str(value))
            elif isinstance(widget, QtWidgets.QLineEdit):
                widget.setText(str(value))

        for enum_name, table in self.enum_tables.items():
            enum_cls = getattr(settings, enum_name)
            for row, member in enumerate(enum_cls):
                value_item = table.item(row, 1)
                if value_item is not None:
                    value_item.setText(self._format_enum_value(member.value))

    def apply_changes(self) -> None:
        try:
            scalar_values = self._collect_scalar_values()
            enum_values = self._collect_enum_values()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Błąd", str(exc))
            return

        for name, value in scalar_values.items():
            setattr(settings, name, value)
            self.scalar_types[name] = type(value)

        for enum_name, values in enum_values.items():
            enum_cls = getattr(settings, enum_name)
            for member_name, value in values.items():
                self._update_enum_value(enum_cls, member_name, value)

        logger.success("Zaktualizowano ustawienia w module settings.")
        self.refresh_from_settings()

    def save_to_file(self) -> None:
        try:
            scalar_values = self._collect_scalar_values()
            enum_values = self._collect_enum_values()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Błąd", str(exc))
            return

        data = {
            "scalars": {
                name: str(value) if isinstance(value, PurePath) else value
                for name, value in scalar_values.items()
            },
            "enums": {
                enum_name: {
                    member: self._serialize_enum_value(value)
                    for member, value in values.items()
                }
                for enum_name, values in enum_values.items()
            },
        }

        CONFIG_FILE.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
        logger.success("Zapisano konfigurację do pliku %s.", CONFIG_FILE)

    def load_from_file(self) -> None:
        if not CONFIG_FILE.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "Błąd",
                f"Plik {CONFIG_FILE} nie istnieje.",
            )
            return

        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        try:
            scalars = data.get("scalars", {})
            enums = data.get("enums", {})
            for name, value in scalars.items():
                if name in self.scalar_inputs:
                    value_type = self.scalar_types.get(name)
                    if isinstance(value_type, type) and issubclass(value_type, PurePath):
                        setattr(settings, name, Path(str(value)))
                    else:
                        setattr(settings, name, value)
                    self.scalar_types[name] = type(getattr(settings, name))

            for enum_name, members in enums.items():
                if enum_name in self.enum_tables:
                    enum_cls = getattr(settings, enum_name)
                    for member_name, raw_value in members.items():
                        value = self._deserialize_enum_value(enum_cls, member_name, raw_value)
                        self._update_enum_value(enum_cls, member_name, value)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Błąd", str(exc))
            return

        logger.success("Wczytano konfigurację z pliku %s.", CONFIG_FILE)
        self.refresh_from_settings()

    def _collect_scalar_values(self) -> Dict[str, Any]:
        values: Dict[str, Any] = {}
        for name, widget in self.scalar_inputs.items():
            value_type = self.scalar_types[name]
            if isinstance(widget, QtWidgets.QCheckBox):
                values[name] = bool(widget.isChecked())
                continue

            if isinstance(widget, PathInput):
                raw_text = widget.text()
                text = raw_text.strip()
            else:
                assert isinstance(widget, QtWidgets.QLineEdit)
                raw_text = widget.text()
                text = raw_text.strip()
            try:
                if value_type is int:
                    values[name] = int(text)
                elif value_type is float:
                    values[name] = float(text)
                elif value_type is bool:
                    values[name] = text.lower() in {"1", "true", "tak", "yes"}
                elif isinstance(value_type, type) and issubclass(value_type, PurePath):
                    if not text:
                        raise ValueError(f"Ścieżka dla {name} nie może być pusta.")
                    values[name] = Path(raw_text)
                else:
                    values[name] = text
            except ValueError as exc:
                raise ValueError(f"Niepoprawna wartość dla {name}: '{text}'.") from exc
        return values

    def _is_directory_path(self, name: str, path_value: PurePath) -> bool:
        upper_name = name.upper()
        if upper_name.endswith("_CMD"):
            return False
        if upper_name.endswith(("_DIR", "_DIRS")):
            return True
        if upper_name.endswith(("_FILE", "_PATH", "_FPATH")):
            return False
        return path_value.suffix == ""

    def _collect_enum_values(self) -> Dict[str, Dict[str, Any]]:
        values: Dict[str, Dict[str, Any]] = {}
        for enum_name, table in self.enum_tables.items():
            enum_cls = getattr(settings, enum_name)
            member_values: Dict[str, Any] = {}
            for row in range(table.rowCount()):
                member_item = table.item(row, 0)
                value_item = table.item(row, 1)
                if member_item is None or value_item is None:
                    continue
                member_name = member_item.text()
                text_value = value_item.text().strip()
                try:
                    parsed = self._parse_enum_value(enum_cls, member_name, text_value)
                except ValueError as exc:
                    raise ValueError(str(exc)) from exc
                member_values[member_name] = parsed
            values[enum_name] = member_values
        return values

    def _parse_enum_value(self, enum_cls: type, member_name: str, text_value: str) -> Any:
        member = getattr(enum_cls, member_name)
        current_value = member.value

        if isinstance(current_value, Key):
            key_name = text_value
            if text_value.startswith("Key."):
                key_name = text_value.split(".", 1)[1]
            try:
                return getattr(Key, key_name)
            except AttributeError as exc:
                raise ValueError(f"Niepoprawny klawisz dla {member_name}: '{text_value}'.") from exc

        if isinstance(current_value, bool):
            return text_value.lower() in {"1", "true", "tak", "yes"}

        if isinstance(current_value, int):
            return int(text_value)

        if isinstance(current_value, float):
            return float(text_value)

        return text_value

    def _update_enum_value(self, enum_cls: type, member_name: str, value: Any) -> None:
        member = getattr(enum_cls, member_name)
        old_value = member.value
        if old_value == value:
            return

        # Update the Enum member value
        if old_value in enum_cls._value2member_map_:  # type: ignore[attr-defined]
            del enum_cls._value2member_map_[old_value]  # type: ignore[attr-defined]
        member._value_ = value  # type: ignore[attr-defined]
        enum_cls._value2member_map_[value] = member  # type: ignore[attr-defined]

    def _format_enum_value(self, value: Any) -> str:
        if isinstance(value, Key):
            return value.name
        return str(value)

    def _serialize_enum_value(self, value: Any) -> Any:
        if isinstance(value, Key):
            return f"Key.{value.name}"
        return value

    def _deserialize_enum_value(self, enum_cls: type, member_name: str, raw_value: Any) -> Any:
        if isinstance(raw_value, str) and raw_value.startswith("Key."):
            return self._parse_enum_value(enum_cls, member_name, raw_value)
        return self._parse_enum_value(enum_cls, member_name, str(raw_value))


class MainWindow(QtWidgets.QMainWindow):
    """Main window of the GUI application."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("METIIN Vision GUI")
        self.resize(900, 700)

        self.log_handler = QtLogHandler()
        self.log_handler.log_signal.connect(self._append_log)
        logger.add(self.log_handler.emit, format=LOG_FORMAT, level="TRACE", enqueue=True)

        self.bot_configs = self._create_bot_configs()
        self.bot_runners: Dict[str, BotRunner] = {}
        self.bot_controls: Dict[str, BotControlWidget] = {}
        self.active_bot: Optional[str] = None
        self.log_messages: List[str] = []
        self.log_filter_edit: Optional[QtWidgets.QLineEdit] = None
        self.autoscroll_checkbox: Optional[QtWidgets.QCheckBox] = None
        self.log_count_label: Optional[QtWidgets.QLabel] = None
        self.log_view: Optional[QtWidgets.QPlainTextEdit] = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._create_bots_tab(), "Tryby")

        self.settings_panel = SettingsPanel()
        tabs.addTab(self.settings_panel, "Ustawienia")

        tabs.addTab(self._create_logs_tab(), "Logi")

        self.setCentralWidget(tabs)

    def _create_bots_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        info_label = QtWidgets.QLabel(
            "Uruchomienie nowego trybu zatrzyma możliwość uruchomienia innych trybów\n"
            "do czasu zakończenia aktualnie działającego bota."
        )
        layout.addWidget(info_label)

        for config in self.bot_configs:
            control = BotControlWidget(config)
            control.start_requested.connect(self._start_bot)
            layout.addWidget(control)
            self.bot_controls[config.identifier] = control
            self.bot_runners[config.identifier] = BotRunner(
                config=config, log_sink=self.log_handler.emit
            )
            runner = self.bot_runners[config.identifier]
            runner.signals.started.connect(self._on_bot_started)
            runner.signals.finished.connect(self._on_bot_finished)
            runner.signals.failed.connect(self._on_bot_failed)

        layout.addStretch(1)
        widget.setLayout(layout)
        return widget

    def _create_logs_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        controls_layout = QtWidgets.QHBoxLayout()

        filter_label = QtWidgets.QLabel("Filtr:")
        controls_layout.addWidget(filter_label)

        self.log_filter_edit = QtWidgets.QLineEdit()
        self.log_filter_edit.setPlaceholderText("Wpisz fragment tekstu, aby przefiltrować logi")
        self.log_filter_edit.textChanged.connect(self._refresh_log_view)
        controls_layout.addWidget(self.log_filter_edit)

        self.autoscroll_checkbox = QtWidgets.QCheckBox("Autoprzewijanie")
        self.autoscroll_checkbox.setChecked(True)
        controls_layout.addWidget(self.autoscroll_checkbox)

        clear_button = QtWidgets.QPushButton("Wyczyść")
        clear_button.clicked.connect(self._clear_logs)
        controls_layout.addWidget(clear_button)

        save_button = QtWidgets.QPushButton("Zapisz do pliku")
        save_button.clicked.connect(self._save_logs)
        controls_layout.addWidget(save_button)

        self.log_count_label = QtWidgets.QLabel("Wpisy: 0")
        controls_layout.addWidget(self.log_count_label)

        controls_layout.addStretch(1)

        layout.addLayout(controls_layout)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view)

        widget.setLayout(layout)
        self._refresh_log_view()
        return widget

    def _create_bot_configs(self) -> List[BotConfig]:
        return [
            BotConfig(
                identifier="dung_polana",
                title="Upadła Polana",
                target=dung_polana.run,
                script_name=Path(dung_polana.__file__).name,
                parameters=[
                    BotParameter(
                        name="stage",
                        label="Stage",
                        type="int",
                        default=0,
                        minimum=0,
                        maximum=5,
                    ),
                    BotParameter(
                        name="log_level",
                        label="Poziom logów",
                        type="choice",
                        choices=["TRACE", "DEBUG", "INFO"],
                        default="INFO",
                    ),
                    BotParameter(
                        name="saved_credentials_idx",
                        label="Index konta",
                        type="int",
                        default=1,
                        minimum=1,
                        maximum=10,
                    ),
                ],
            ),
            BotConfig(
                identifier="idle_metins",
                title="Idle Metins",
                target=idle_metins.run,
                script_name=Path(idle_metins.__file__).name,
                parameters=[
                    BotParameter(
                        name="event",
                        label="Tryb eventu",
                        type="bool",
                        default=False,
                    ),
                    BotParameter(
                        name="log_level",
                        label="Poziom logów",
                        type="choice",
                        choices=["TRACE", "DEBUG", "INFO"],
                        default="INFO",
                    ),
                    BotParameter(
                        name="start",
                        label="Kanał startowy",
                        type="int",
                        default=1,
                        minimum=1,
                        maximum=8,
                    ),
                    BotParameter(
                        name="saved_credentials_idx",
                        label="Index konta",
                        type="int",
                        default=1,
                        minimum=1,
                        maximum=10,
                    ),
                ],
            ),
            BotConfig(
                identifier="fishbot",
                title="Fishbot",
                target=fishbot.run,
                script_name=Path(fishbot.__file__).name,
                parameters=[
                    BotParameter(
                        name="stage",
                        label="Stage",
                        type="int",
                        default=0,
                        minimum=0,
                        maximum=5,
                    ),
                    BotParameter(
                        name="log_level",
                        label="Poziom logów",
                        type="choice",
                        choices=["TRACE", "DEBUG", "INFO"],
                        default="TRACE",
                    ),
                    BotParameter(
                        name="saved_credentials_idx",
                        label="Index konta",
                        type="int",
                        default=1,
                        minimum=1,
                        maximum=10,
                    ),
                ],
            ),
        ]

    def _start_bot(self, bot_id: str, params: Dict[str, Any]) -> None:
        if self.active_bot and self.active_bot != bot_id:
            QtWidgets.QMessageBox.warning(
                self,
                "Bot aktywny",
                "Inny bot jest aktualnie uruchomiony. Zaczekaj na zakończenie pracy.",
            )
            return

        runner = self.bot_runners[bot_id]
        if runner.is_running():
            QtWidgets.QMessageBox.information(
                self,
                "Bot już działa",
                "Wybrany bot jest już uruchomiony.",
            )
            return

        try:
            runner.start(params)
        except Exception as exc:  # pylint: disable=broad-except
            QtWidgets.QMessageBox.critical(
                self,
                "Błąd",
                f"Nie udało się uruchomić bota: {exc}",
            )
            return

        self.active_bot = bot_id
        self.bot_controls[bot_id].set_status("Uruchomiony")

    def _on_bot_started(self, bot_id: str) -> None:
        logger.info("Bot '{}' został uruchomiony.", bot_id)

    def _on_bot_finished(self, bot_id: str) -> None:
        logger.success("Bot '{}' zakończył działanie.", bot_id)
        self.bot_controls[bot_id].set_status("Zakończony")
        if self.active_bot == bot_id:
            self.active_bot = None

    def _on_bot_failed(self, bot_id: str, reason: str) -> None:
        logger.error("Bot '{}' zakończył się błędem: {}", bot_id, reason)
        self.bot_controls[bot_id].set_status("Błąd")
        if self.active_bot == bot_id:
            self.active_bot = None

    @QtCore.pyqtSlot(str)
    def _append_log(self, message: str) -> None:
        self.log_messages.append(message)
        self._refresh_log_view()

    def _filtered_messages(self) -> List[str]:
        if not self.log_filter_edit:
            return list(self.log_messages)

        filter_text = self.log_filter_edit.text().strip().lower()
        if not filter_text:
            return list(self.log_messages)
        return [msg for msg in self.log_messages if filter_text in msg.lower()]

    def _refresh_log_view(self) -> None:
        if not self.log_view:
            return

        scrollbar = self.log_view.verticalScrollBar()
        previous_value = scrollbar.value()
        filtered = self._filtered_messages()
        self.log_view.setPlainText("\n".join(filtered))

        if self.log_count_label:
            total = len(self.log_messages)
            filtered_count = len(filtered)
            self.log_count_label.setText(f"Wpisy: {filtered_count} / {total}")

        if self.autoscroll_checkbox and self.autoscroll_checkbox.isChecked():
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(previous_value)

    def _clear_logs(self) -> None:
        self.log_messages.clear()
        self._refresh_log_view()

    def _save_logs(self) -> None:
        if not self.log_messages:
            QtWidgets.QMessageBox.information(
                self,
                "Brak danych",
                "Brak logów do zapisania.",
            )
            return

        filtered_messages = self._filtered_messages()
        default_path = str(Path.cwd() / "logi.txt")
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Zapisz logi",
            default_path,
            "Pliki tekstowe (*.txt);;Wszystkie pliki (*)",
        )

        if not file_path:
            return

        try:
            Path(file_path).write_text("\n".join(filtered_messages), encoding="utf-8")
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Błąd zapisu",
                f"Nie udało się zapisać logów: {exc}",
            )
            return

        QtWidgets.QMessageBox.information(
            self,
            "Zapisano",
            f"Logi zostały zapisane do pliku: {file_path}",
        )


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
