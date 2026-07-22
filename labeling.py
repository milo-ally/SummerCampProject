import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QImage,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


APP_TITLE = "Road Calibration Annotator"
DEFAULT_TARGET_WIDTH_M = 24.0
DEFAULT_TARGET_LENGTH_M = 249.0
DEFAULT_FRAME_SAMPLES = 24
HANDLE_RADIUS = 6


@dataclass
class FrameCandidate:
    index: int
    score: float
    frame: np.ndarray


@dataclass
class DistanceAnnotation:
    name: str
    p1: QPointF
    p2: QPointF
    distance_m: float = 0.0
    item: QGraphicsLineItem | None = field(default=None, repr=False)
    label_item: QGraphicsTextItem | None = field(default=None, repr=False)


@dataclass
class RoadRegion:
    region_id: str
    name: str
    target_width_m: float
    target_length_m: float
    polygon_item: QGraphicsPolygonItem | None = field(default=None, repr=False)
    handles: list["RoadHandleItem"] = field(default_factory=list, repr=False)


def default_output_json_path(video_path: str) -> Path:
    path = Path(video_path)
    return path.parent / path.stem / f"{path.stem}.calibration.json"


def frame_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(gray.mean())
    brightness_penalty = abs(brightness - 128.0)
    return float(sharpness - brightness_penalty)


class FrameExtractionWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(object, object)
    failed = Signal(str)

    def __init__(self, video_path: Path, samples: int):
        super().__init__()
        self.video_path = video_path
        self.samples = samples

    def run(self) -> None:
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            self.failed.emit(f"Could not open video: {self.video_path}")
            return

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            frame_count = self.samples
        sample_count = max(1, min(self.samples, frame_count))
        indexes = np.linspace(0, frame_count - 1, sample_count, dtype=int)
        candidates: list[FrameCandidate] = []

        for offset, index in enumerate(indexes, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok:
                candidates.append(
                    FrameCandidate(
                        index=int(index),
                        score=frame_score(frame),
                        frame=frame,
                    )
                )
            percent = int(offset * 100 / sample_count)
            self.progress.emit(percent, f"Parsing video frames... {percent}%")

        capture.release()
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        self.finished.emit(self.video_path, candidates[: min(8, len(candidates))])


def cv_to_pixmap(frame: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    bytes_per_line = channels * width
    image = QImage(
        rgb.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888,
    )
    return QPixmap.fromImage(image.copy())


def ordered_quad_points(points: list[QPointF]) -> list[QPointF]:
    if len(points) != 4:
        raise ValueError("Road polygon must contain exactly four points.")
    center_x = sum(point.x() for point in points) / 4.0
    center_y = sum(point.y() for point in points) / 4.0
    sorted_points = sorted(
        points,
        key=lambda point: np.arctan2(point.y() - center_y, point.x() - center_x),
    )
    top_points = sorted(sorted_points[:2], key=lambda point: point.x())
    bottom_points = sorted(sorted_points[2:], key=lambda point: point.x(), reverse=True)
    return [top_points[0], top_points[1], bottom_points[0], bottom_points[1]]


class RoadHandleItem(QGraphicsEllipseItem):
    def __init__(
        self,
        editor: "CalibrationScene",
        region: RoadRegion,
        index: int,
        position: QPointF,
    ):
        super().__init__(
            -HANDLE_RADIUS,
            -HANDLE_RADIUS,
            HANDLE_RADIUS * 2,
            HANDLE_RADIUS * 2,
        )
        self.editor = editor
        self.region = region
        self.index = index
        self.setPos(position)
        self.setBrush(QBrush(QColor(255, 214, 64)))
        self.setPen(QPen(QColor(20, 20, 20), 1))
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setZValue(4)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.editor.update_region_polygon(self.region)
        return super().itemChange(change, value)

    def mousePressEvent(self, event) -> None:
        if self.region in self.editor.road_regions:
            self.editor.set_active_region(self.editor.road_regions.index(self.region))
            self.editor.parent_window.refresh_region_list()
        super().mousePressEvent(event)


class CalibrationScene(QGraphicsScene):
    def __init__(self, parent_window: "CalibrationWindow"):
        super().__init__(parent_window)
        self.parent_window = parent_window
        self.pixmap_item: QGraphicsPixmapItem | None = None
        self.road_regions: list[RoadRegion] = []
        self.active_region_index = -1
        self.distance_annotations: list[DistanceAnnotation] = []
        self.drag_start: QPointF | None = None
        self.preview_line: QGraphicsLineItem | None = None
        self.mode = "select"

    def set_frame(self, frame: np.ndarray) -> None:
        self.clear()
        self.distance_annotations.clear()
        self.road_regions.clear()
        self.active_region_index = -1
        pixmap = cv_to_pixmap(frame)
        self.pixmap_item = self.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))
        self.add_default_region(pixmap.width(), pixmap.height())

    def add_default_region(
        self,
        width: int | None = None,
        height: int | None = None,
    ) -> RoadRegion:
        if width is None or height is None:
            rect = self.sceneRect()
            width = max(1, int(rect.width()))
            height = max(1, int(rect.height()))
        region_number = len(self.road_regions) + 1
        offset = min(0.05 * (region_number - 1), 0.18)
        points = [
            QPointF(width * (0.32 + offset), height * (0.35 + offset * 0.2)),
            QPointF(width * (0.68 + offset), height * (0.35 + offset * 0.2)),
            QPointF(width * (0.86 + offset), height * (0.92 - offset * 0.1)),
            QPointF(width * (0.14 + offset), height * (0.92 - offset * 0.1)),
        ]
        region = RoadRegion(
            region_id=f"region_{region_number}",
            name=f"region_{region_number}",
            target_width_m=DEFAULT_TARGET_WIDTH_M,
            target_length_m=DEFAULT_TARGET_LENGTH_M,
        )
        region.polygon_item = self.addPolygon(
            QPolygonF(points),
            QPen(QColor(230, 32, 32), 2),
            QBrush(QColor(230, 32, 32, 40)),
        )
        region.polygon_item.setZValue(2)
        region.polygon_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        for index, point in enumerate(points):
            handle = RoadHandleItem(self, region, index, point)
            region.handles.append(handle)
            self.addItem(handle)
        self.road_regions.append(region)
        self.active_region_index = len(self.road_regions) - 1
        self.refresh_region_styles()
        return region

    def active_region(self) -> RoadRegion | None:
        if 0 <= self.active_region_index < len(self.road_regions):
            return self.road_regions[self.active_region_index]
        return None

    def set_active_region(self, index: int) -> None:
        if 0 <= index < len(self.road_regions):
            self.active_region_index = index
            self.refresh_region_styles()

    def remove_active_region(self) -> None:
        region = self.active_region()
        if region is None or len(self.road_regions) <= 1:
            return
        if region.polygon_item:
            self.removeItem(region.polygon_item)
        for handle in region.handles:
            self.removeItem(handle)
        self.road_regions.remove(region)
        self.active_region_index = min(
            self.active_region_index,
            len(self.road_regions) - 1,
        )
        self.refresh_region_styles()

    def refresh_region_styles(self) -> None:
        colors = [
            QColor(230, 32, 32),
            QColor(66, 133, 244),
            QColor(52, 168, 83),
            QColor(251, 188, 5),
            QColor(171, 71, 188),
        ]
        for index, region in enumerate(self.road_regions):
            color = colors[index % len(colors)]
            active = index == self.active_region_index
            if region.polygon_item:
                region.polygon_item.setPen(QPen(color, 4 if active else 2))
                region.polygon_item.setBrush(
                    QBrush(QColor(color.red(), color.green(), color.blue(), 44))
                )
            for handle in region.handles:
                handle.setBrush(
                    QBrush(QColor(255, 214, 64) if active else QColor(220, 220, 220))
                )

    def update_region_polygon(self, region: RoadRegion) -> None:
        if not region.polygon_item or len(region.handles) != 4:
            return
        region.polygon_item.setPolygon(
            QPolygonF([handle.pos() for handle in region.handles])
        )

    def mousePressEvent(self, event):
        if self.mode == "line" and event.button() == Qt.MouseButton.LeftButton:
            self.drag_start = event.scenePos()
            self.preview_line = self.addLine(
                self.drag_start.x(),
                self.drag_start.y(),
                self.drag_start.x(),
                self.drag_start.y(),
                QPen(QColor(66, 133, 244), 2),
            )
            self.preview_line.setZValue(3)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.mode == "line" and self.drag_start and self.preview_line:
            position = event.scenePos()
            self.preview_line.setLine(
                self.drag_start.x(),
                self.drag_start.y(),
                position.x(),
                position.y(),
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.mode == "line" and self.drag_start and self.preview_line:
            end = event.scenePos()
            start = self.drag_start
            self.removeItem(self.preview_line)
            self.preview_line = None
            self.drag_start = None
            if np.hypot(end.x() - start.x(), end.y() - start.y()) >= 5:
                self.add_distance_annotation(start, end)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def add_distance_annotation(self, start: QPointF, end: QPointF) -> None:
        annotation = DistanceAnnotation(
            name=f"distance_{len(self.distance_annotations) + 1}",
            p1=start,
            p2=end,
        )
        line_item = self.addLine(
            start.x(),
            start.y(),
            end.x(),
            end.y(),
            QPen(QColor(66, 133, 244), 3),
        )
        line_item.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        line_item.setZValue(3)
        label_item = self.addText(annotation.name)
        label_item.setDefaultTextColor(QColor(66, 133, 244))
        label_item.setPos((start.x() + end.x()) / 2.0, (start.y() + end.y()) / 2.0)
        label_item.setZValue(4)
        annotation.item = line_item
        annotation.label_item = label_item
        self.distance_annotations.append(annotation)
        self.parent_window.refresh_distance_list()
        self.parent_window.select_annotation(annotation)

    def selected_distance_annotation(self) -> DistanceAnnotation | None:
        selected_items = set(self.selectedItems())
        for annotation in self.distance_annotations:
            if annotation.item in selected_items:
                return annotation
        return None

    def delete_annotation(self, annotation: DistanceAnnotation) -> None:
        if annotation.item:
            self.removeItem(annotation.item)
        if annotation.label_item:
            self.removeItem(annotation.label_item)
        self.distance_annotations.remove(annotation)
        self.parent_window.refresh_distance_list()


class CalibrationView(QGraphicsView):
    def __init__(self, scene: CalibrationScene):
        super().__init__(scene)
        self.auto_fit = True
        self.setMinimumSize(360, 260)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    def fit_scene(self) -> None:
        scene_rect = self.scene().sceneRect()
        if scene_rect.isValid() and not scene_rect.isEmpty():
            self.fitInView(scene_rect, Qt.AspectRatioMode.KeepAspectRatio)
            self.auto_fit = True

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.auto_fit:
            self.fit_scene()

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(zoom_factor, zoom_factor)
            self.auto_fit = False
            event.accept()
            return
        super().wheelEvent(event)


class CalibrationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.source_video_path: Path | None = None
        self.output_json_path: Path | None = None
        self.frame_samples = DEFAULT_FRAME_SAMPLES
        self.frame_candidates: list[FrameCandidate] = []
        self.current_frame: np.ndarray | None = None
        self.selected_annotation: DistanceAnnotation | None = None
        self.extract_thread: QThread | None = None
        self.extract_worker: FrameExtractionWorker | None = None

        self.scene = CalibrationScene(self)
        self.scene.selectionChanged.connect(self.scene_selection_changed)
        self.view = CalibrationView(self.scene)
        self.view.setRenderHints(self.view.renderHints())
        self.view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)

        self.video_path_edit = QLineEdit()
        self.video_path_edit.setReadOnly(True)
        self.frame_combo = QComboBox()
        self.region_list = QListWidget()
        self.region_name_edit = QLineEdit()
        self.width_spin = QDoubleSpinBox()
        self.length_spin = QDoubleSpinBox()
        self.annotation_list = QListWidget()
        self.annotation_name_edit = QLineEdit()
        self.distance_spin = QDoubleSpinBox()
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Open a video to begin.")

        self.setup_ui()

    def setup_ui(self) -> None:
        toolbar = QToolBar("Tools")
        self.addToolBar(toolbar)
        open_action = QAction("Open Video", self)
        open_action.triggered.connect(self.choose_video)
        toolbar.addAction(open_action)
        select_action = QAction("Select", self)
        select_action.triggered.connect(lambda: self.set_mode("select"))
        toolbar.addAction(select_action)
        line_action = QAction("Draw Distance Line", self)
        line_action.triggered.connect(lambda: self.set_mode("line"))
        toolbar.addAction(line_action)
        save_action = QAction("Save Calibration", self)
        save_action.triggered.connect(self.save_calibration)
        toolbar.addAction(save_action)
        fit_action = QAction("Fit Frame", self)
        fit_action.triggered.connect(self.view.fit_scene)
        toolbar.addAction(fit_action)

        self.width_spin.setRange(0.1, 10000.0)
        self.width_spin.setDecimals(3)
        self.width_spin.setValue(DEFAULT_TARGET_WIDTH_M)
        self.width_spin.setSuffix(" m")
        self.length_spin.setRange(0.1, 10000.0)
        self.length_spin.setDecimals(3)
        self.length_spin.setValue(DEFAULT_TARGET_LENGTH_M)
        self.length_spin.setSuffix(" m")
        self.distance_spin.setRange(0.0, 10000.0)
        self.distance_spin.setDecimals(3)
        self.distance_spin.setSuffix(" m")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.hide()
        self.status_label.setWordWrap(True)
        self.video_path_edit.setMinimumWidth(0)
        self.frame_combo.setMinimumWidth(0)

        self.frame_combo.currentIndexChanged.connect(self.change_frame)
        self.region_list.currentRowChanged.connect(self.region_row_changed)
        self.region_name_edit.editingFinished.connect(self.apply_region_edits)
        self.width_spin.valueChanged.connect(self.apply_region_edits)
        self.length_spin.valueChanged.connect(self.apply_region_edits)
        self.annotation_list.currentRowChanged.connect(self.annotation_row_changed)
        self.annotation_name_edit.editingFinished.connect(self.apply_annotation_edits)
        self.distance_spin.valueChanged.connect(self.apply_annotation_edits)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.choose_video)
        add_region_button = QPushButton("Add region")
        add_region_button.clicked.connect(self.add_region)
        delete_region_button = QPushButton("Delete region")
        delete_region_button.clicked.connect(self.delete_active_region)
        line_button = QPushButton("Draw distance line")
        line_button.clicked.connect(lambda: self.set_mode("line"))
        select_button = QPushButton("Select/edit")
        select_button.clicked.connect(lambda: self.set_mode("select"))
        delete_button = QPushButton("Delete selected line")
        delete_button.clicked.connect(self.delete_selected_annotation)
        save_button = QPushButton("Save calibration JSON")
        save_button.clicked.connect(self.save_calibration)

        right_panel = QWidget()
        right_panel.setMinimumWidth(260)
        right_panel.setMaximumWidth(430)
        right_layout = QVBoxLayout(right_panel)
        form = QFormLayout()
        form.addRow("Video", self.video_path_edit)
        form.addRow("", browse_button)
        form.addRow("Effective frame", self.frame_combo)
        right_layout.addLayout(form)
        right_layout.addWidget(QLabel("Regions"))
        right_layout.addWidget(self.region_list)
        right_layout.addWidget(add_region_button)
        right_layout.addWidget(delete_region_button)
        region_form = QFormLayout()
        region_form.addRow("Region name", self.region_name_edit)
        region_form.addRow("Road width", self.width_spin)
        region_form.addRow("Road length", self.length_spin)
        right_layout.addLayout(region_form)
        right_layout.addWidget(select_button)
        right_layout.addWidget(line_button)
        right_layout.addWidget(QLabel("Distance annotations"))
        right_layout.addWidget(self.annotation_list)
        annotation_form = QFormLayout()
        annotation_form.addRow("Name", self.annotation_name_edit)
        annotation_form.addRow("Distance", self.distance_spin)
        right_layout.addLayout(annotation_form)
        right_layout.addWidget(delete_button)
        right_layout.addWidget(save_button)
        right_layout.addWidget(self.progress_bar)
        right_layout.addWidget(self.status_label)
        right_layout.addStretch(1)

        central = QWidget()
        layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.view)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([860, 320])
        layout.addWidget(splitter)
        self.setCentralWidget(central)
        self.setMinimumSize(720, 480)
        self.resize(1100, 720)

    def set_mode(self, mode: str) -> None:
        self.scene.mode = mode
        self.view.setDragMode(
            QGraphicsView.DragMode.NoDrag
            if mode == "line"
            else QGraphicsView.DragMode.RubberBandDrag
        )
        self.status_label.setText(
            "Draw a distance line by dragging on the frame."
            if mode == "line"
            else "Select a line or drag road polygon handles."
        )

    def choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose video",
            str(Path.cwd()),
            "Videos (*.mp4 *.avi *.mov *.mkv);;All files (*.*)",
        )
        if path:
            self.load_video(Path(path))

    def load_video(self, video_path: Path) -> None:
        if self.extract_thread and self.extract_thread.isRunning():
            QMessageBox.information(
                self,
                "Video parsing",
                "A video is already being parsed. Please wait for it to finish.",
            )
            return

        self.video_path_edit.setText(str(video_path))
        self.source_video_path = video_path
        self.output_json_path = default_output_json_path(str(video_path))
        self.frame_combo.blockSignals(True)
        self.frame_combo.clear()
        self.frame_combo.blockSignals(False)
        self.frame_candidates = []
        self.scene.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.status_label.setText("Parsing video frames... 0%")
        self.set_controls_enabled(False)

        self.extract_thread = QThread(self)
        self.extract_worker = FrameExtractionWorker(video_path, self.frame_samples)
        self.extract_worker.moveToThread(self.extract_thread)
        self.extract_thread.started.connect(self.extract_worker.run)
        self.extract_worker.progress.connect(self.update_parse_progress)
        self.extract_worker.finished.connect(self.finish_video_loading)
        self.extract_worker.failed.connect(self.fail_video_loading)
        self.extract_worker.finished.connect(self.extract_thread.quit)
        self.extract_worker.failed.connect(self.extract_thread.quit)
        self.extract_thread.finished.connect(self.extract_worker.deleteLater)
        self.extract_thread.finished.connect(self.cleanup_extract_thread)
        self.extract_thread.start()

    def set_controls_enabled(self, enabled: bool) -> None:
        self.frame_combo.setEnabled(enabled)
        self.region_list.setEnabled(enabled)
        self.region_name_edit.setEnabled(enabled)
        self.width_spin.setEnabled(enabled)
        self.length_spin.setEnabled(enabled)
        self.annotation_list.setEnabled(enabled)
        self.annotation_name_edit.setEnabled(enabled)
        self.distance_spin.setEnabled(enabled)

    def update_parse_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)

    def finish_video_loading(
        self,
        video_path: Path,
        candidates: list[FrameCandidate],
    ) -> None:
        if not candidates:
            self.fail_video_loading("No readable frames found.")
            return
        self.frame_candidates = candidates
        self.frame_combo.blockSignals(True)
        for candidate in self.frame_candidates:
            self.frame_combo.addItem(
                f"frame {candidate.index} | score {candidate.score:.1f}",
                candidate.index,
            )
        self.frame_combo.blockSignals(False)
        self.frame_combo.setCurrentIndex(0)
        self.set_frame_candidate(0)
        self.progress_bar.setValue(100)
        self.progress_bar.hide()
        self.set_controls_enabled(True)
        self.status_label.setText(
            f"Loaded {video_path.name}. Pick an effective frame, then drag road handles."
        )

    def fail_video_loading(self, message: str) -> None:
        self.progress_bar.hide()
        self.set_controls_enabled(True)
        QMessageBox.critical(self, "Video error", message)
        self.status_label.setText("Video parsing failed.")

    def cleanup_extract_thread(self) -> None:
        self.extract_thread = None
        self.extract_worker = None

    def change_frame(self, index: int) -> None:
        if index >= 0:
            self.set_frame_candidate(index)

    def set_frame_candidate(self, index: int) -> None:
        candidate = self.frame_candidates[index]
        self.current_frame = candidate.frame
        self.scene.set_frame(candidate.frame)
        self.view.fit_scene()
        self.refresh_region_list()
        self.refresh_distance_list()
        self.status_label.setText(
            f"Loaded frame {candidate.index}. Drag yellow road handles; draw lines for measured distances."
        )

    def refresh_region_list(self) -> None:
        self.region_list.blockSignals(True)
        self.region_list.clear()
        for region in self.scene.road_regions:
            self.region_list.addItem(QListWidgetItem(region.name))
        self.region_list.setCurrentRow(self.scene.active_region_index)
        self.region_list.blockSignals(False)
        self.load_active_region_to_form()

    def load_active_region_to_form(self) -> None:
        region = self.scene.active_region()
        if region is None:
            self.region_name_edit.clear()
            return
        self.region_name_edit.setText(region.name)
        self.width_spin.blockSignals(True)
        self.length_spin.blockSignals(True)
        self.width_spin.setValue(region.target_width_m)
        self.length_spin.setValue(region.target_length_m)
        self.width_spin.blockSignals(False)
        self.length_spin.blockSignals(False)

    def region_row_changed(self, row: int) -> None:
        self.scene.set_active_region(row)
        self.load_active_region_to_form()

    def apply_region_edits(self) -> None:
        region = self.scene.active_region()
        if region is None:
            return
        region.name = self.region_name_edit.text().strip() or region.name
        region.target_width_m = float(self.width_spin.value())
        region.target_length_m = float(self.length_spin.value())
        row = self.scene.active_region_index
        if row >= 0 and row < self.region_list.count():
            self.region_list.item(row).setText(region.name)

    def add_region(self) -> None:
        if self.current_frame is None:
            return
        self.scene.add_default_region()
        self.refresh_region_list()

    def delete_active_region(self) -> None:
        self.scene.remove_active_region()
        self.refresh_region_list()

    def refresh_distance_list(self) -> None:
        self.annotation_list.blockSignals(True)
        self.annotation_list.clear()
        for annotation in self.scene.distance_annotations:
            text = f"{annotation.name}: {annotation.distance_m:.3f} m"
            self.annotation_list.addItem(QListWidgetItem(text))
        self.annotation_list.blockSignals(False)

    def select_annotation(self, annotation: DistanceAnnotation | None) -> None:
        self.selected_annotation = annotation
        if annotation is None:
            self.annotation_name_edit.clear()
            self.distance_spin.setValue(0.0)
            return
        self.annotation_name_edit.setText(annotation.name)
        self.distance_spin.blockSignals(True)
        self.distance_spin.setValue(annotation.distance_m)
        self.distance_spin.blockSignals(False)
        row = self.scene.distance_annotations.index(annotation)
        self.annotation_list.blockSignals(True)
        self.annotation_list.setCurrentRow(row)
        self.annotation_list.blockSignals(False)

    def annotation_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self.scene.distance_annotations):
            self.select_annotation(None)
            return
        annotation = self.scene.distance_annotations[row]
        if annotation.item:
            annotation.item.setSelected(True)
        self.select_annotation(annotation)

    def scene_selection_changed(self) -> None:
        annotation = self.scene.selected_distance_annotation()
        if annotation is not None:
            self.select_annotation(annotation)
            return
        selected_items = set(self.scene.selectedItems())
        for index, region in enumerate(self.scene.road_regions):
            if region.polygon_item in selected_items:
                self.scene.set_active_region(index)
                self.refresh_region_list()
                return

    def apply_annotation_edits(self) -> None:
        annotation = self.selected_annotation or self.scene.selected_distance_annotation()
        if annotation is None:
            return
        annotation.name = self.annotation_name_edit.text().strip() or annotation.name
        annotation.distance_m = float(self.distance_spin.value())
        if annotation.label_item:
            annotation.label_item.setPlainText(
                f"{annotation.name}: {annotation.distance_m:.3f} m"
            )
        self.refresh_distance_list()

    def delete_selected_annotation(self) -> None:
        annotation = self.selected_annotation or self.scene.selected_distance_annotation()
        if annotation is None:
            return
        self.scene.delete_annotation(annotation)
        self.select_annotation(None)

    def road_source_points(self, region: RoadRegion) -> list[list[float]]:
        points = [handle.pos() for handle in region.handles]
        ordered = ordered_quad_points(points)
        return [[round(point.x(), 3), round(point.y(), 3)] for point in ordered]

    def target_points(self, region: RoadRegion) -> list[list[float]]:
        width = float(region.target_width_m)
        length = float(region.target_length_m)
        return [[0.0, 0.0], [width, 0.0], [width, length], [0.0, length]]

    def regions_payload(self) -> list[dict[str, object]]:
        payload = []
        for region in self.scene.road_regions:
            payload.append(
                {
                    "region_id": region.region_id,
                    "name": region.name,
                    "source": self.road_source_points(region),
                    "target": self.target_points(region),
                    "target_width_m": region.target_width_m,
                    "target_length_m": region.target_length_m,
                }
            )
        return payload

    def distance_annotations_payload(self) -> list[dict[str, object]]:
        payload = []
        for annotation in self.scene.distance_annotations:
            payload.append(
                {
                    "name": annotation.name,
                    "source": [
                        [round(annotation.p1.x(), 3), round(annotation.p1.y(), 3)],
                        [round(annotation.p2.x(), 3), round(annotation.p2.y(), 3)],
                    ],
                    "distance_m": annotation.distance_m,
                }
            )
        return payload

    def save_calibration(self) -> None:
        if not self.source_video_path:
            QMessageBox.warning(self, "Missing video", "Open a video first.")
            return
        output_path = self.output_json_path
        if output_path is None:
            output_path = default_output_json_path(str(self.source_video_path))
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save calibration JSON",
            str(output_path),
            "JSON (*.json);;All files (*.*)",
        )
        if not path:
            return
        selected_frame_index = self.frame_combo.currentData()
        regions = self.regions_payload()
        if not regions:
            QMessageBox.warning(self, "Missing region", "Add at least one region.")
            return
        first_region = regions[0]
        payload = {
            "source_video_path": str(self.source_video_path),
            "frame_index": int(selected_frame_index),
            "source": first_region["source"],
            "target": first_region["target"],
            "target_width_m": first_region["target_width_m"],
            "target_length_m": first_region["target_length_m"],
            "regions": regions,
            "distance_annotations": self.distance_annotations_payload(),
            "notes": {
                "regions": "Each region is independently ordered as top-left, top-right, bottom-right, bottom-left at export time.",
                "source": "Legacy alias for the first region.",
                "target": "Road-plane coordinates in meters.",
            },
        }
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
        self.output_json_path = output_path
        self.status_label.setText(f"Saved calibration: {output_path}")


def main() -> None:
    app = QApplication(sys.argv)
    window = CalibrationWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
