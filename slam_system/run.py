from sequence_manager import SequenceManager
from ptz_camera import PTZCamera
from ptz_slam import PtzSlam
from util import save_camera_pose
import numpy as np

sequence = SequenceManager(
    image_path="../pre_processing/frames",
)
sequence.length = 1502

slam = PtzSlam()

u, v = 1280 / 2, 720 / 2
camera_center = np.array([53.8528, -8.37071, 15.0785])
base_rotation = np.identity(3)

first_camera = PTZCamera((u, v), camera_center, base_rotation)

initial_pan = 60.0
initial_tilt = -5.0
initial_focal_length = 3000.0
first_camera.set_ptz((initial_pan, initial_tilt, initial_focal_length))

first_img = sequence.get_image_gray(index=0, dataset_type=3)
first_bounding_box = sequence.get_bounding_box_mask(0)

slam.init_system(first_img, first_camera, bounding_box=first_bounding_box)
slam.add_keyframe(first_img, first_camera, 0)

pan_list = [first_camera.get_ptz()[0]]
tilt_list = [first_camera.get_ptz()[1]]
zoom_list = [first_camera.get_ptz()[2]]

for i in range(1, sequence.length):
    print(f"===== Processing frame {i} =====")
    img = sequence.get_image_gray(index=i, dataset_type=3)
    bounding_box = sequence.get_bounding_box_mask(i)

    slam.tracking(next_img=img, bad_tracking_percentage=80, bounding_box=bounding_box)

    if slam.tracking_lost:
        print("Tracking lost! Relocalizing...")
        relocalized_camera = slam.relocalize(img, slam.current_camera)
        slam.init_system(img, relocalized_camera, bounding_box=bounding_box)
    elif slam.new_keyframe:
        print("Adding a new keyframe.")
        slam.add_keyframe(img, slam.current_camera, i)

    pan_list.append(slam.cameras[i].pan)
    tilt_list.append(slam.cameras[i].tilt)
    zoom_list.append(slam.cameras[i].focal_length)

save_camera_pose(pan_list, tilt_list, zoom_list, "./my_football_result.mat")
print("SLAM processing complete. Results saved to my_football_result.mat")