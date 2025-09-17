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
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[str]] = None
    help_text: Optional[str] = None
    editable_label: bool = True
    placeholder: Optional[str] = None


@dataclass
class BotConfig:
    identifier: str
    title: str
    target: Callable[..., None]
    script_name: str
    parameters: List[BotParameter] = field(default_factory=list)
    title_editable: bool = False
    instructions: Optional[str] = None


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
        self.param_labels: Dict[str, QtWidgets.QLabel] = {}
        self.label_edits: Dict[str, QtWidgets.QLineEdit] = {}
        self.status_label = QtWidgets.QLabel("Nieaktywny")
        self.title_edit: Optional[QtWidgets.QLineEdit] = None
        self.instructions_label: Optional[QtWidgets.QLabel] = None
        self.default_title = config.title
        self.default_labels = {param.name: param.label for param in config.parameters}
        self.default_values = {param.name: param.default for param in config.parameters}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(10)

        if self.config.title_editable:
            title_layout = QtWidgets.QHBoxLayout()
            title_label = QtWidgets.QLabel("Nazwa trybu:")
            self.title_edit = QtWidgets.QLineEdit(self.config.title)
            self.title_edit.setPlaceholderText("Pozostaw puste, aby użyć nazwy domyślnej")
            self.title_edit.textChanged.connect(self._on_title_changed)
            title_layout.addWidget(title_label)
            title_layout.addWidget(self.title_edit)
            layout.addLayout(title_layout)

        if self.config.instructions:
            instruction_label = QtWidgets.QLabel(self.config.instructions)
            instruction_label.setWordWrap(True)
            instruction_label.setStyleSheet("color: #444; font-size: 12px;")
            layout.addWidget(instruction_label)
            self.instructions_label = instruction_label

        for param in self.config.parameters:
            layout.addWidget(self._create_parameter_block(param))

        buttons_layout = QtWidgets.QHBoxLayout()
        start_button = QtWidgets.QPushButton("Uruchom")
        start_button.clicked.connect(self._emit_start)
        buttons_layout.addWidget(start_button)
        buttons_layout.addStretch(1)
        buttons_layout.addWidget(QtWidgets.QLabel("Status:"))
        buttons_layout.addWidget(self.status_label)

        layout.addLayout(buttons_layout)
        self.setLayout(layout)

    def _create_parameter_block(self, param: BotParameter) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(4)

        label = QtWidgets.QLabel(param.label)
        label.setWordWrap(True)
        label.setStyleSheet("font-weight: bold;")
        container_layout.addWidget(label)
        self.param_labels[param.name] = label

        widget = self._create_input_widget(param)
        container_layout.addWidget(widget)

        if param.help_text:
            help_label = QtWidgets.QLabel(param.help_text)
            help_label.setWordWrap(True)
            help_label.setStyleSheet("color: #555; font-size: 11px;")
            container_layout.addWidget(help_label)

        if param.editable_label:
            rename_edit = QtWidgets.QLineEdit(param.label)
            rename_edit.setPlaceholderText("Własna etykieta (opcjonalnie)")
            rename_edit.textChanged.connect(lambda text, name=param.name: self._on_label_changed(name, text))
            container_layout.addWidget(rename_edit)
            self.label_edits[param.name] = rename_edit

        container.setLayout(container_layout)
        return container

    def _create_input_widget(self, param: BotParameter) -> QtWidgets.QWidget:
        widget: QtWidgets.QWidget
        if param.type == "int":
            spin = QtWidgets.QSpinBox()
            spin.setMinimum(int(param.minimum) if param.minimum is not None else -10_000)
            spin.setMaximum(int(param.maximum) if param.maximum is not None else 10_000)
            if param.step is not None:
                spin.setSingleStep(int(param.step))
            if param.default is not None:
                spin.setValue(int(param.default))
            widget = spin
        elif param.type == "float":
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setMinimum(float(param.minimum) if param.minimum is not None else -10_000.0)
            spin.setMaximum(float(param.maximum) if param.maximum is not None else 10_000.0)
            spin.setSingleStep(float(param.step) if param.step is not None else 0.1)
            if param.default is not None:
                spin.setValue(float(param.default))
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
            if param.placeholder:
                line.setPlaceholderText(param.placeholder)
            if param.default is not None:
                line.setText(str(param.default))
            widget = line

        if param.help_text:
            widget.setToolTip(param.help_text)

        self.inputs[param.name] = widget
        self.param_types[param.name] = param.type
        return widget

    def _on_title_changed(self, text: str) -> None:
        final_title = text.strip() or self.default_title
        self.setTitle(final_title)

    def _on_label_changed(self, name: str, text: str) -> None:
        label = self.param_labels.get(name)
        default_label = self.default_labels.get(name, "")
        if label is not None:
            label.setText(text.strip() or default_label)

    def _emit_start(self) -> None:
        self.start_requested.emit(self.config.identifier, self.parameters())

    def parameters(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for name, widget in self.inputs.items():
            params[name] = self._get_widget_value(widget, self.param_types[name])
        return params

    def _get_widget_value(self, widget: QtWidgets.QWidget, param_type: str) -> Any:
        if param_type == "int":
            assert isinstance(widget, QtWidgets.QSpinBox)
            return widget.value()
        if param_type == "float":
            assert isinstance(widget, QtWidgets.QDoubleSpinBox)
            return widget.value()
        if param_type == "bool":
            assert isinstance(widget, QtWidgets.QCheckBox)
            return widget.isChecked()
        if param_type == "choice":
            assert isinstance(widget, QtWidgets.QComboBox)
            return widget.currentText()
        assert isinstance(widget, QtWidgets.QLineEdit)
        return widget.text().strip()

    def set_parameter_value(self, name: str, value: Any) -> None:
        if name not in self.inputs:
            return
        widget = self.inputs[name]
        param_type = self.param_types[name]
        if param_type == "int":
            assert isinstance(widget, QtWidgets.QSpinBox)
            widget.setValue(int(value))
        elif param_type == "float":
            assert isinstance(widget, QtWidgets.QDoubleSpinBox)
            widget.setValue(float(value))
        elif param_type == "bool":
            assert isinstance(widget, QtWidgets.QCheckBox)
            widget.setChecked(bool(value))
        elif param_type == "choice":
            assert isinstance(widget, QtWidgets.QComboBox)
            index = widget.findText(str(value))
            if index >= 0:
                widget.setCurrentIndex(index)
        else:
            assert isinstance(widget, QtWidgets.QLineEdit)
            widget.setText("" if value is None else str(value))

    def apply_customization(self, data: Dict[str, Any]) -> None:
        title = data.get("title")
        if title is not None:
            if self.title_edit is not None:
                self.title_edit.setText(title)
            else:
                self.setTitle(title or self.default_title)

        for name, label_value in data.get("labels", {}).items():
            edit = self.label_edits.get(name)
            if edit is not None:
                edit.setText(label_value)
            elif name in self.param_labels:
                self.param_labels[name].setText(label_value)

        for name, value in data.get("values", {}).items():
            self.set_parameter_value(name, value)

    def get_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}

        if self.title_edit is not None:
            title_text = self.title_edit.text().strip()
            if title_text and title_text != self.default_title:
                state["title"] = title_text

        label_overrides: Dict[str, str] = {}
        for name, edit in self.label_edits.items():
            text = edit.text().strip()
            default_label = self.default_labels.get(name, "")
            if text and text != default_label:
                label_overrides[name] = text
        if label_overrides:
            state["labels"] = label_overrides

        value_overrides: Dict[str, Any] = {}
        for name, widget in self.inputs.items():
            param_type = self.param_types[name]
            current_value = self._get_widget_value(widget, param_type)
            default_value = self.default_values.get(name)

            if param_type == "int":
                if default_value is None:
                    if current_value != 0:
                        value_overrides[name] = int(current_value)
                elif int(current_value) != int(default_value):
                    value_overrides[name] = int(current_value)
            elif param_type == "float":
                if default_value is None:
                    if abs(float(current_value)) > 1e-6:
                        value_overrides[name] = float(current_value)
                elif abs(float(current_value) - float(default_value)) > 1e-6:
                    value_overrides[name] = float(current_value)
            elif param_type == "bool":
                if bool(current_value) != bool(default_value):
                    value_overrides[name] = bool(current_value)
            elif param_type == "choice":
                current_text = str(current_value)
                default_text = str(default_value) if default_value is not None else ""
                if current_text != default_text:
                    value_overrides[name] = current_text
            else:
                current_text = str(current_value).strip()
                default_text = "" if default_value is None else str(default_value).strip()
                if current_text != default_text:
                    value_overrides[name] = current_text

        if value_overrides:
            state["values"] = value_overrides

        return state

    def reset_to_defaults(self) -> None:
        if self.title_edit is not None:
            self.title_edit.blockSignals(True)
            self.title_edit.setText(self.default_title)
            self.title_edit.blockSignals(False)
            self.setTitle(self.default_title)
        else:
            self.setTitle(self.default_title)

        for name, label in self.param_labels.items():
            label.setText(self.default_labels.get(name, label.text()))

        for name, edit in self.label_edits.items():
            edit.blockSignals(True)
            edit.setText(self.default_labels.get(name, ""))
            edit.blockSignals(False)

        for name, widget in self.inputs.items():
            default_value = self.default_values.get(name)
            if default_value is None and self.param_types[name] not in {"int", "float", "bool"}:
                self.set_parameter_value(name, "")
            elif default_value is not None:
                self.set_parameter_value(name, default_value)

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

        try:
            existing = (
                json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                if CONFIG_FILE.exists()
                else {}
            )
        except json.JSONDecodeError as exc:
            logger.warning("Plik %s zawiera niepoprawny JSON – tworzenie nowego pliku. (%s)", CONFIG_FILE, exc)
            existing = {}

        existing["scalars"] = {
            name: str(value) if isinstance(value, PurePath) else value
            for name, value in scalar_values.items()
        }
        existing["enums"] = {
            enum_name: {
                member: self._serialize_enum_value(value)
                for member, value in values.items()
            }
            for enum_name, values in enum_values.items()
        }

        try:
            CONFIG_FILE.write_text(
                json.dumps(existing, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Błąd zapisu",
                f"Nie udało się zapisać konfiguracji: {exc}",
            )
            return

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

        self.gui_config: Dict[str, Any] = self._load_gui_config()
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
        outer_layout = QtWidgets.QVBoxLayout()

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)

        inner_widget = QtWidgets.QWidget()
        inner_layout = QtWidgets.QVBoxLayout()

        info_label = QtWidgets.QLabel(
            "Instrukcje pracy z trybami:\n"
            "- każdy tryb można dowolnie skonfigurować przed uruchomieniem (wartości liczbowe podawaj w sekundach).\n"
            "- zmiany nazw pól i wartości zapisujesz przyciskiem 'Zapisz konfigurację GUI'.\n"
            "- w danej chwili uruchomiony może być tylko jeden tryb."
        )
        info_label.setWordWrap(True)
        inner_layout.addWidget(info_label)

        for config in self.bot_configs:
            control = BotControlWidget(config)
            control.start_requested.connect(self._start_bot)
            inner_layout.addWidget(control)
            self.bot_controls[config.identifier] = control
            self.bot_runners[config.identifier] = BotRunner(
                config=config, log_sink=self.log_handler.emit
            )
            runner = self.bot_runners[config.identifier]
            runner.signals.started.connect(self._on_bot_started)
            runner.signals.finished.connect(self._on_bot_finished)
            runner.signals.failed.connect(self._on_bot_failed)

            saved_state = self.gui_config.get(config.identifier)
            if isinstance(saved_state, dict):
                control.apply_customization(saved_state)

        buttons_layout = QtWidgets.QHBoxLayout()
        save_button = QtWidgets.QPushButton("Zapisz konfigurację GUI")
        save_button.clicked.connect(self._save_gui_config)
        buttons_layout.addWidget(save_button)

        reset_button = QtWidgets.QPushButton("Przywróć domyślne GUI")
        reset_button.clicked.connect(self._reset_gui_defaults)
        buttons_layout.addWidget(reset_button)
        buttons_layout.addStretch(1)

        inner_layout.addLayout(buttons_layout)

        inner_layout.addStretch(1)

        inner_widget.setLayout(inner_layout)
        scroll_area.setWidget(inner_widget)
        outer_layout.addWidget(scroll_area)
        widget.setLayout(outer_layout)
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
                title_editable=True,
                instructions=(
                    "Konfiguruj czasy i progi detekcji dla lochu Upadła Polana. "
                    "Wszystkie wartości czasowe podawaj w sekundach. Timeouty etapów wpisz "
                    "jako sześć liczb oddzielonych przecinkami (kolejno: wejście, 200 potworów, "
                    "minibossy, metiny, drop run, boss)."
                ),
                parameters=[
                    BotParameter(
                        name="stage",
                        label="Początkowy etap",
                        type="int",
                        default=0,
                        minimum=0,
                        maximum=5,
                        help_text="Numer etapu, od którego bot ma rozpocząć (0=wejście, 5=boss).",
                    ),
                    BotParameter(
                        name="log_level",
                        label="Poziom logów",
                        type="choice",
                        choices=["TRACE", "DEBUG", "INFO"],
                        default="INFO",
                        help_text="Określa szczegółowość logów zapisywanych w zakładce Logi.",
                    ),
                    BotParameter(
                        name="saved_credentials_idx",
                        label="Slot danych logowania",
                        type="int",
                        default=1,
                        minimum=1,
                        maximum=10,
                        help_text="Numer slotu zapisanych danych logowania używany podczas automatycznego logowania.",
                    ),
                    BotParameter(
                        name="yolo_confidence_threshold",
                        label="Próg YOLO (globalny)",
                        type="float",
                        default=0.7,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        help_text="Minimalna pewność detekcji YOLO dla wszystkich obiektów (0-1).",
                    ),
                    BotParameter(
                        name="yolo_metin_confidence_threshold",
                        label="Próg YOLO dla metinów",
                        type="float",
                        default=0.8,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        help_text="Minimalna pewność detekcji YOLO dla metinów w etapach 3-5.",
                    ),
                    BotParameter(
                        name="nonsense_msg_similarity_threshold",
                        label="Próg wiadomości 'nonsense'",
                        type="float",
                        default=0.0,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        help_text="Minimalne podobieństwo wiadomości, aby uznać ją za bezużyteczną (0-1).",
                    ),
                    BotParameter(
                        name="stage_timeouts",
                        label="Timeouty etapów",
                        type="str",
                        default="60,120,120,300,180,420",
                        help_text=(
                            "Czasy limitów dla etapów w sekundach – podaj sześć liczb oddzielonych przecinkami "
                            "(wejście, 200 potworów, minibossy, metiny, drop run, boss)."
                        ),
                        placeholder="np. 60,120,120,300,180,420",
                    ),
                    BotParameter(
                        name="walk_time_to_metin",
                        label="Czas podejścia do metina",
                        type="float",
                        default=10.0,
                        minimum=0.0,
                        maximum=60.0,
                        step=0.5,
                        help_text="Ile sekund bot czeka, aż postać podejdzie do wskazanego metina.",
                    ),
                    BotParameter(
                        name="metin_destroy_time",
                        label="Czas niszczenia metina",
                        type="float",
                        default=11.0,
                        minimum=0.0,
                        maximum=120.0,
                        step=0.5,
                        help_text="Czas w sekundach przeznaczony na niszczenie jednego metina.",
                    ),
                    BotParameter(
                        name="loading_timeout",
                        label="Limit ekranu ładowania",
                        type="float",
                        default=10.0,
                        minimum=0.0,
                        maximum=120.0,
                        step=0.5,
                        help_text="Maksymalny czas oczekiwania na ekran ładowania zanim bot podejmie akcję awaryjną.",
                    ),
                    BotParameter(
                        name="stage_200_mobs_idle_time",
                        label="Idle – 200 potworów",
                        type="float",
                        default=16.0,
                        minimum=0.0,
                        maximum=120.0,
                        step=0.5,
                        help_text="Czas bezczynności (z podnoszeniem) podczas etapu 200 potworów.",
                    ),
                    BotParameter(
                        name="stage_item_drop_idle_time",
                        label="Idle – drop run",
                        type="float",
                        default=16.0,
                        minimum=0.0,
                        maximum=120.0,
                        step=0.5,
                        help_text="Czas bezczynności na etapie zdobywania run.",
                    ),
                    BotParameter(
                        name="stage_boss_walk_time",
                        label="Czas dojścia do bossa",
                        type="float",
                        default=3.0,
                        minimum=0.0,
                        maximum=60.0,
                        step=0.5,
                        help_text="Ile sekund bot daje postaci na podejście do bossa przed atakiem.",
                    ),
                    BotParameter(
                        name="reenter_wait",
                        label="Przerwa przed ponownym wejściem",
                        type="float",
                        default=2.0,
                        minimum=0.0,
                        maximum=120.0,
                        step=0.5,
                        help_text="Czas oczekiwania przed ponownym wejściem do lochu po ukończeniu runu.",
                    ),
                ],
            ),
            BotConfig(
                identifier="idle_metins",
                title="Idle Metins",
                target=idle_metins.run,
                script_name=Path(idle_metins.__file__).name,
                title_editable=True,
                instructions=(
                    "Automatyczne farmienie metinów na mapie. Czasy podawaj w sekundach, "
                    "progi YOLO w przedziale 0-1. W trybie eventowym bot przeskakuje co trzy kanały."
                ),
                parameters=[
                    BotParameter(
                        name="event",
                        label="Tryb eventu",
                        type="bool",
                        default=False,
                        help_text="Włącza tryb eventowy (skok co 3 kanały oraz dodatkowe podnoszenie dropu).",
                    ),
                    BotParameter(
                        name="log_level",
                        label="Poziom logów",
                        type="choice",
                        choices=["TRACE", "DEBUG", "INFO"],
                        default="INFO",
                        help_text="Określa szczegółowość logów zapisywanych w zakładce Logi.",
                    ),
                    BotParameter(
                        name="start",
                        label="Kanał startowy",
                        type="int",
                        default=1,
                        minimum=1,
                        maximum=8,
                        help_text="Kanał, od którego rozpoczyna się cykl wyszukiwania metinów.",
                    ),
                    BotParameter(
                        name="saved_credentials_idx",
                        label="Slot danych logowania",
                        type="int",
                        default=1,
                        minimum=1,
                        maximum=10,
                        help_text="Numer slotu zapisanych danych logowania używany podczas automatycznego logowania.",
                    ),
                    BotParameter(
                        name="yolo_confidence_threshold",
                        label="Próg YOLO",
                        type="float",
                        default=0.75,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        help_text="Minimalna pewność detekcji YOLO dla metinów (0-1).",
                    ),
                    BotParameter(
                        name="channel_timeout",
                        label="Limit czasu na kanał",
                        type="float",
                        default=20.0,
                        minimum=0.0,
                        maximum=180.0,
                        step=0.5,
                        help_text="Maksymalny czas (s) spędzony na kanale bez znalezienia metina.",
                    ),
                    BotParameter(
                        name="walk_to_metin_time",
                        label="Czas podejścia do metina",
                        type="float",
                        default=1.25,
                        minimum=0.0,
                        maximum=30.0,
                        step=0.1,
                        help_text="Czas oczekiwania po kliknięciu metina, aż postać do niego podejdzie.",
                    ),
                    BotParameter(
                        name="metin_destroy_time",
                        label="Czas niszczenia metina",
                        type="float",
                        default=1.0,
                        minimum=0.0,
                        maximum=60.0,
                        step=0.1,
                        help_text="Podstawowy czas niszczenia metina (bot sam doliczy rezerwę w trybie eventu).",
                    ),
                    BotParameter(
                        name="looking_around_move_camera_press_time",
                        label="Czas obrotu kamerą",
                        type="float",
                        default=0.5,
                        minimum=0.0,
                        maximum=10.0,
                        step=0.1,
                        help_text="Jak długo (s) bot obraca kamerę w poszukiwaniu metina zanim ponowi YOLO.",
                    ),
                ],
            ),
            BotConfig(
                identifier="fishbot",
                title="Fishbot",
                target=fishbot.run,
                script_name=Path(fishbot.__file__).name,
                title_editable=True,
                instructions=(
                    "Ustaw okno łowienia oraz próg detekcji ryb. Okno podaj jako cztery liczby (góra, dół, lewa, prawa) "
                    "odnoszące się do obrazu przechwytywanego z gry."
                ),
                parameters=[
                    BotParameter(
                        name="stage",
                        label="Etap testowy",
                        type="int",
                        default=0,
                        minimum=0,
                        maximum=5,
                        help_text="Parametr pomocniczy – pozostaw 0, chyba że testujesz inne scenariusze.",
                    ),
                    BotParameter(
                        name="log_level",
                        label="Poziom logów",
                        type="choice",
                        choices=["TRACE", "DEBUG", "INFO"],
                        default="TRACE",
                        help_text="Określa szczegółowość logów zapisywanych w zakładce Logi.",
                    ),
                    BotParameter(
                        name="saved_credentials_idx",
                        label="Slot danych logowania",
                        type="int",
                        default=1,
                        minimum=1,
                        maximum=10,
                        help_text="Numer slotu zapisanych danych logowania używany podczas automatycznego logowania.",
                    ),
                    BotParameter(
                        name="yolo_confidence_threshold",
                        label="Próg YOLO",
                        type="float",
                        default=0.95,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        help_text="Minimalna pewność detekcji YOLO dla ryby (0-1).",
                    ),
                    BotParameter(
                        name="fishing_window",
                        label="Okno łowienia",
                        type="str",
                        default="77,304,101,379",
                        placeholder="góra,dół,lewa,prawa",
                        help_text="Wymiary wycinka obrazu do analizy w pikselach (top,bottom,left,right).",
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

    def _load_gui_config(self) -> Dict[str, Any]:
        if not CONFIG_FILE.exists():
            return {}

        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Nie udało się wczytać konfiguracji GUI: {}", exc)
            return {}

        gui_data = data.get("gui", {})
        if isinstance(gui_data, dict):
            return gui_data
        return {}

    def _collect_gui_config(self) -> Dict[str, Any]:
        collected: Dict[str, Any] = {}
        for identifier, control in self.bot_controls.items():
            state = control.get_state()
            if state:
                collected[identifier] = state
        return collected

    def _save_gui_config(self) -> None:
        gui_data = self._collect_gui_config()

        try:
            existing = (
                json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                if CONFIG_FILE.exists()
                else {}
            )
        except json.JSONDecodeError as exc:
            logger.warning("Plik %s zawiera niepoprawny JSON – tworzenie nowego pliku. (%s)", CONFIG_FILE, exc)
            existing = {}

        existing["gui"] = gui_data

        try:
            CONFIG_FILE.write_text(
                json.dumps(existing, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Błąd zapisu",
                f"Nie udało się zapisać konfiguracji GUI: {exc}",
            )
            return

        self.gui_config = gui_data
        logger.success("Zapisano konfigurację GUI do pliku %s.", CONFIG_FILE)
        QtWidgets.QMessageBox.information(
            self,
            "Zapisano",
            "Konfiguracja GUI została zapisana.",
        )

    def _reset_gui_defaults(self) -> None:
        for control in self.bot_controls.values():
            control.reset_to_defaults()

        self.gui_config = {}
        logger.info("Przywrócono domyślne ustawienia GUI w bieżącej sesji.")
        QtWidgets.QMessageBox.information(
            self,
            "Przywrócono",
            "Przywrócono domyślne nazwy i wartości GUI. Zapisz konfigurację, aby zachować zmiany.",
        )


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
