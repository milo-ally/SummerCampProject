import argparse
import json
from pathlib import Path


WINDOW_NAME = "source_calibration"
POINT_COUNT = 4
POINT_NAMES = ["top-left", "top-right", "bottom-right", "bottom-left"]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pick SOURCE points from a video frame.")
    parser.add_argument("--source-video-path", required=True, type=str)
    parser.add_argument("--display-width", default=1280, type=int)
    parser.add_argument("--frame-index", default=0, type=int)
    parser.add_argument("--padding", default=300, type=int)
    parser.add_argument("--output-json-path", type=str)
    return parser.parse_args()


def read_frame(video_path: str, frame_index: int):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()

    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index}: {video_path}")
    return frame


def format_source(points: list[tuple[int, int]]) -> str:
    rows = ",\n        ".join(f"[{x}, {y}]" for x, y in points)
    return f"SOURCE = np.array(\n    [\n        {rows}\n    ]\n)"


def save_calibration_source(path: str, points: list[tuple[int, int]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    calibration = {}
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as file:
            calibration = json.load(file)

    calibration["source"] = [[x, y] for x, y in points]
    calibration.setdefault(
        "target",
        [
            [0, 0],
            [24, 0],
            [24, 249],
            [0, 249],
        ],
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(calibration, file, indent=2)
        file.write("\n")


def default_output_json_path(video_path: str) -> Path:
    path = Path(video_path)
    return path.parent / path.stem / f"{path.stem}.calibration.json"


if __name__ == "__main__":
    args = parse_arguments()

    import cv2
    import numpy as np

    frame = read_frame(args.source_video_path, args.frame_index)
    original_height, original_width = frame.shape[:2]

    display_width = min(args.display_width, original_width)
    display_height = int(original_height * display_width / original_width)
    display_size = (display_width, display_height)
    scale_x = original_width / display_width
    scale_y = original_height / display_height
    padding = args.padding
    points: list[tuple[int, int]] = []
    mouse_position: list[tuple[int, int] | None] = [None]

    def to_display(point: tuple[int, int]) -> tuple[int, int]:
        x, y = point
        return round(x / scale_x) + padding, round(y / scale_y) + padding

    def to_original(x: int, y: int) -> tuple[int, int]:
        return round((x - padding) * scale_x), round((y - padding) * scale_y)

    def constrain_point(point: tuple[int, int], flags: int) -> tuple[int, int]:
        if not points or not flags & cv2.EVENT_FLAG_SHIFTKEY:
            return point
        previous_x, previous_y = points[-1]
        x, _ = point
        return x, previous_y

    def draw() -> np.ndarray:
        resized_frame = cv2.resize(frame, display_size, interpolation=cv2.INTER_AREA)
        canvas = np.zeros(
            (display_height + padding * 2, display_width + padding * 2, 3),
            dtype=np.uint8,
        )
        canvas[padding : padding + display_height, padding : padding + display_width] = (
            resized_frame
        )
        for index, (x, y) in enumerate(points, start=1):
            display_point = to_display((x, y))
            cv2.circle(canvas, display_point, 5, (0, 255, 255), -1)
            cv2.putText(
                canvas,
                str(index),
                (display_point[0] + 8, display_point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
        if len(points) == POINT_COUNT:
            polygon = np.array([to_display(point) for point in points], dtype=np.int32)
            cv2.polylines(canvas, [polygon], True, (0, 0, 255), 2)
        if points and mouse_position[0] and len(points) < POINT_COUNT:
            preview_point = to_display(mouse_position[0])
            cv2.line(canvas, to_display(points[-1]), preview_point, (0, 0, 255), 1)
        if len(points) < POINT_COUNT:
            cv2.putText(
                canvas,
                f"Click {len(points) + 1}/4: {POINT_NAMES[len(points)]} | Ctrl+Z/z undo | r reset | Shift straight",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
        return canvas

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        mouse_position[0] = constrain_point(to_original(x, y), flags)

        if event != cv2.EVENT_LBUTTONDOWN or len(points) >= POINT_COUNT:
            return
        points.append(mouse_position[0])
        if len(points) == POINT_COUNT:
            print(format_source(points))
            output_json_path = Path(args.output_json_path) if args.output_json_path else default_output_json_path(args.source_video_path)
            save_calibration_source(str(output_json_path), points)
            print(f"Saved calibration source to {output_json_path}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        cv2.imshow(WINDOW_NAME, draw())
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("r"):
            points.clear()
        if key in (26, ord("z")) and points:
            points.pop()

    cv2.destroyAllWindows()
