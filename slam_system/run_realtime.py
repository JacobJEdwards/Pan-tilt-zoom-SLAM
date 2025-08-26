import cv2 as cv
import numpy as np
import os
import threading
import time
from sequence_manager import SequenceManager
from ptz_slam import PtzSlam
from util import save_camera_pose
from ptz_camera import PTZCamera

IMAGE_DIR = "../pre_processing/frames"
OUTPUT_FILE = "./my_football_realtime_result.mat"
PROCESSING_INTERVAL = 10
FPS = 30  

class SharedData:
    def __init__(self):
        self.latest_pan = None
        self.latest_tilt = None
        self.latest_zoom = None
        self.processing_active = True
        self.lock = threading.Lock()

shared_data = SharedData()

def slam_worker(slam, sequence, first_camera, image_files):
    pan_list = [first_camera.get_ptz()[0]]
    tilt_list = [first_camera.get_ptz()[1]]
    zoom_list = [first_camera.get_ptz()[2]]

    for i in range(1, sequence.length):
        with shared_data.lock:
            if not shared_data.processing_active:
                break

        frame_color = cv.imread(image_files[i])
        if frame_color is None:
            continue

        if i % PROCESSING_INTERVAL == 0:
            print(f"===== Processing frame {i} =====")
            frame_gray = cv.cvtColor(frame_color, cv.COLOR_BGR2GRAY)

            slam.tracking(next_img=frame_gray, bad_tracking_percentage=80, bounding_box=None)

            if slam.tracking_lost:
                print("Tracking lost! Relocalizing...")
                relocalized_camera = slam.relocalize(frame_gray, slam.current_camera)
                slam.init_system(frame_gray, relocalized_camera, bounding_box=None)
            elif slam.new_keyframe:
                print("Adding a new keyframe.")
                slam.add_keyframe(frame_gray, slam.current_camera, i)

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

if __name__ == "__main__":
    try:
        image_files = sorted([os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))])
        if not image_files:
            print(f"Error: No image files found in {IMAGE_DIR}")
            exit()
    except FileNotFoundError:
        print(f"Error: The directory {IMAGE_DIR} does not exist.")
        exit()

    sequence = SequenceManager()
    sequence.length = len(image_files)
    slam = PtzSlam()

    first_frame_for_size = cv.imread(image_files[0])
    height, width, _ = first_frame_for_size.shape
    u, v = width / 2, height / 2

    camera_center = np.array([53.8528, -8.37071, 15.0785])
    base_rotation = np.identity(3)
    first_camera = PTZCamera((u, v), camera_center, base_rotation)
    initial_pan, initial_tilt, initial_focal_length = 0.0, 0.0, 3000.0
    first_camera.set_ptz((initial_pan, initial_tilt, initial_focal_length))

    first_frame_gray = cv.cvtColor(first_frame_for_size, cv.COLOR_BGR2GRAY)
    slam.init_system(first_frame_gray, first_camera)
    slam.add_keyframe(first_frame_gray, first_camera, 0)

    with shared_data.lock:
        shared_data.latest_pan, shared_data.latest_tilt, shared_data.latest_zoom = first_camera.get_ptz()

    slam_thread = threading.Thread(target=slam_worker, args=(slam, sequence, first_camera, image_files))
    slam_thread.start()

    start_time = time.time()
    i = 0
    while i < len(image_files):
        target_time = start_time + i / FPS
        current_time = time.time()
        if current_time < target_time:
            time.sleep(target_time - current_time)

        frame_color = cv.imread(image_files[i])
        if frame_color is None:
            break

        with shared_data.lock:
            pan = shared_data.latest_pan
            tilt = shared_data.latest_tilt
            zoom = shared_data.latest_zoom
            active = shared_data.processing_active

        vis_frame = frame_color.copy()
        cv.putText(vis_frame, f"Frame: {i}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(vis_frame, f"Pan: {pan:.2f}", (20, 80), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv.putText(vis_frame, f"Tilt: {tilt:.2f}", (20, 110), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv.putText(vis_frame, f"Zoom: {zoom:.2f}", (20, 140), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv.imshow("PTZ SLAM - Real Time Simulation", vis_frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            with shared_data.lock:
                shared_data.processing_active = False
            break

        i += 1

    with shared_data.lock:
        shared_data.processing_active = False

    slam_thread.join()
    cv.destroyAllWindows()