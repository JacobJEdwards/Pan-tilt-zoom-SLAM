import cv2 as cv
import numpy as np
import threading
import time
from sequence_manager import SequenceManager
from ptz_slam import PtzSlam
from util import save_camera_pose
from ptz_camera import PTZCamera
from image_process import detect_players
from pathlib import Path

IMAGE_DIR = Path("../pre_processing/frames")
OUTPUT_FILE = Path("./my_football_realtime_result.mat")
PROCESSING_INTERVAL = 10
BB_INTERVAL = 5
FPS = 25

class SharedData:
    def __init__(self):
        self.latest_pan = None
        self.latest_tilt = None
        self.latest_zoom = None
        self.latest_bounding_boxes = []
        self.processing_active = True
        self.lock = threading.Lock()

shared_data = SharedData()

def slam_worker(slam: PtzSlam, first_camera: PTZCamera, image_files: list[Path], processing_interval: int) -> None:
    pan_list = [first_camera.get_ptz()[0]]
    tilt_list = [first_camera.get_ptz()[1]]
    zoom_list = [first_camera.get_ptz()[2]]

    for i in range(1, len(image_files)):
        with shared_data.lock:
            if not shared_data.processing_active:
                break

        frame_color = cv.imread(str(image_files[i]))
        if frame_color is None:
            continue

        if i % PROCESSING_INTERVAL == 0:
            print(f"===== Processing frame {i} =====")

            slam.tracking(next_img_color=frame_color, bad_tracking_percentage=80)

            with shared_data.lock:
                shared_data.latest_bounding_boxes = detect_players(frame_color)

            if slam.tracking_lost:
                print("Tracking lost! Relocalizing...")
                relocalized_camera = slam.relocalize(frame_color, slam.current_camera, bounding_box=None)
                slam.init_system(frame_color, relocalized_camera, bounding_box=None)
            elif slam.new_keyframe:
                print("Adding a new keyframe.")
                slam.add_keyframe(cv.cvtColor(frame_color, cv.COLOR_BGR2GRAY), slam.current_camera, i)

        latest_camera = slam.cameras[-1]
        pan, tilt, zoom = latest_camera.get_ptz()

        pan_list.append(pan)
        tilt_list.append(tilt)
        zoom_list.append(zoom)

        with shared_data.lock:
            shared_data.latest_pan = pan
            shared_data.latest_tilt = tilt
            shared_data.latest_zoom = zoom

    with shared_data.lock:
        shared_data.processing_active = False

    save_camera_pose(pan_list, tilt_list, zoom_list, OUTPUT_FILE)
    print(f"SLAM processing complete. Results saved to {OUTPUT_FILE}")

def run_realtime(image_dir: Path = IMAGE_DIR, fps: int = FPS, bb_interval: int = BB_INTERVAL, processing_interval:
int = PROCESSING_INTERVAL) -> None:
    try:
        image_files = sorted([f for f in image_dir.iterdir() if f.suffix in ['.jpg','.png']])
        if not image_files:
            print(f"Error: No image files found in {image_files}")
            exit()
    except FileNotFoundError:
        print(f"Error: The directory {image_dir} does not exist.")
        exit()

    sequence = SequenceManager()
    sequence.length = len(image_files)
    slam = PtzSlam()

    first_frame_for_size = cv.imread(str(image_files[0]))
    height, width, _ = first_frame_for_size.shape
    u, v = width / 2, height / 2

    camera_center = np.array([53.8528, -8.37071, 15.0785])
    base_rotation = np.identity(3)
    first_camera = PTZCamera((u, v), camera_center, base_rotation)
    initial_pan, initial_tilt, initial_focal_length = 0.0, 0.0, 3000.0
    first_camera.set_ptz((initial_pan, initial_tilt, initial_focal_length))

    slam.init_system(first_frame_for_size, first_camera)
    slam.add_keyframe(cv.cvtColor(first_frame_for_size, cv.COLOR_BGR2GRAY), first_camera, 0)

    with shared_data.lock:
        shared_data.latest_pan, shared_data.latest_tilt, shared_data.latest_zoom = first_camera.get_ptz()
        shared_data.latest_bounding_boxes = detect_players(first_frame_for_size)

    slam_thread = threading.Thread(target=slam_worker, args=(slam, first_camera, image_files, processing_interval))
    slam_thread.start()

    start_time = time.time()
    i = 0
    previous_bounding_boxes = None
    while i < len(image_files):
        target_time = start_time + i / fps
        current_time = time.time()
        if current_time < target_time:
            time.sleep(target_time - current_time)

        frame_color = cv.imread(str(image_files[i]))
        if frame_color is None:
            break

        with shared_data.lock:
            pan = shared_data.latest_pan
            tilt = shared_data.latest_tilt
            zoom = shared_data.latest_zoom
            active = shared_data.processing_active

        if i % bb_interval == 0:
            bounding_boxes = detect_players(frame_color)
            previous_bounding_boxes = bounding_boxes
        else:
            bounding_boxes = previous_bounding_boxes if previous_bounding_boxes is not None else detect_players(frame_color)

        vis_frame = frame_color.copy()

        for box in bounding_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv.putText(vis_frame, f"Frame: {i}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(vis_frame, f"Pan: {pan:.10f}", (20, 80), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv.putText(vis_frame, f"Tilt: {tilt:.10f}", (20, 110), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv.putText(vis_frame, f"Zoom: {zoom:.10f}", (20, 140), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv.imshow("PTZ SLAM - Real Time Simulation", vis_frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q') or not active:
            with shared_data.lock:
                shared_data.processing_active = False
            break

        i += 1

    with shared_data.lock:
        shared_data.processing_active = False

    slam_thread.join()
    cv.destroyAllWindows()

if __name__ == "__main__":
    run_realtime()