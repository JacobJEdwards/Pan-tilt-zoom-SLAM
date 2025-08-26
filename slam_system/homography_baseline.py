"""
Baseline:
Homography based frame-to-frame matching

"""


from sequence_manager import SequenceManager
from ptz_camera import estimate_camera_from_homography
from image_process import *
from util import *
from visualize import *


class HomographyTracking:
    def __init__(self, first_frame, first_frame_matrix):
        self.current_frame = first_frame
        self.first_matrix = first_frame_matrix

        self.accumulate_matrix = [
            first_frame_matrix,
        ]
        self.each_homography = [
            None,
        ]

    def tracking(self, next_frame):
        kp1 = detect_sift(self.current_frame, 100)

        # kp1 = add_gauss(kp1, 2, 1280, 720)

        local_index, kp2 = optical_flow_matching(self.current_frame, next_frame, kp1)

        # kp2, des2 = detect_compute_sift(next_frame, 0)
        kp1 = kp1[local_index]

        # kp2 = add_gauss(kp2, 2, 1280, 720)

        self.current_frame = next_frame
        _, homography = homography_ransac(kp1, kp2, return_matrix=True)
        # homography = compute_homography(kp1, des1, kp2, des2)

        self.each_homography.append(homography)
        self.accumulate_matrix.append(np.dot(homography, self.accumulate_matrix[-1]))


def soccer3():
    sequence = SequenceManager(
        "../../dataset/soccer_dataset/seq3/seq3_ground_truth.mat",
        "../../dataset/soccer_dataset/seq3/seq3_330",
        "../../dataset/soccer_dataset/seq3/seq3_ground_truth.mat",
        "../../dataset/soccer_dataset/seq3/seq3_player_bounding_box.mat",
    )

    line_index, points = load_model(
        "../../dataset/soccer_dataset/highlights_soccer_model.mat"
    )

    first_frame_ptz = (
        sequence.ground_truth_pan[0],
        sequence.ground_truth_tilt[0],
        sequence.ground_truth_f[0],
    )

    first_camera = sequence.camera
    first_camera.set_ptz(first_frame_ptz)

    # 3*4 projection matrix for 1st frame
    first_frame_mat = first_camera.projection_matrix
    first_frame = sequence.get_image_gray(index=0, dataset_type=1)

    img = project_with_homography(first_frame_mat, points, line_index, first_frame)

    # cv.imshow("image", img)
    # cv.waitKey()

    tracking_obj = HomographyTracking(first_frame, first_frame_mat)

    points3d_on_field = uniform_point_sample_on_field(118, 70, 50, 25)

    pan = [first_frame_ptz[0]]
    tilt = [first_frame_ptz[1]]
    f = [first_frame_ptz[2]]

    for i in range(1, sequence.length):
        next_frame = sequence.get_image_gray(index=i, dataset_type=1)
        tracking_obj.tracking(next_frame)

        # img = project_with_homography(tracking_obj.accumulate_matrix[-1], points, line_index, next_frame)

        # compute ptz

        first_camera.set_ptz((pan[-1], tilt[-1], f[-1]))
        pose = estimate_camera_from_homography(
            tracking_obj.accumulate_matrix[-1], first_camera, points3d_on_field
        )

        print("-----" + str(i) + "--------")

        print(tracking_obj.each_homography[-1])

        print(pose)

        # first_camera.set_ptz(pose)
        # img2 = project_with_PTZCamera(first_camera, points, line_index, next_frame)

        print("%f" % (pose[0] - sequence.ground_truth_pan[i]))
        print("%f" % (pose[1] - sequence.ground_truth_tilt[i]))
        print("%f" % (pose[2] - sequence.ground_truth_f[i]))

        pan.append(pose[0])
        tilt.append(pose[1])
        f.append(pose[2])

        # cv.imshow("image", img)
        # cv.imshow("image2", img2)
        # cv.waitKey(0)

    save_camera_pose(
        np.array(pan),
        np.array(tilt),
        np.array(f),
        "C:/graduate_design/experiment_result/baseline2/2-gauss.mat",
    )


def synthesized_test():
    sequence = SequenceManager(
        annotation_path="../../dataset/basketball/ground_truth.mat",
        image_path="../../dataset/synthesized/images",
    )

    gt_pan, gt_tilt, gt_f = load_camera_pose(
        "../../dataset/synthesized/synthesize_ground_truth.mat", separate=True
    )

    line_index, points = load_model("../../dataset/basketball/basketball_model.mat")

    begin_frame = 2400

    first_frame_ptz = (gt_pan[begin_frame], gt_tilt[begin_frame], gt_f[begin_frame])

    first_camera = sequence.camera
    first_camera.set_ptz(first_frame_ptz)

    # 3*4 projection matrix for 1st frame
    first_frame_mat = first_camera.projection_matrix
    first_frame = sequence.get_image_gray(index=begin_frame, dataset_type=2)

    # img = project_with_homography(first_frame_mat, points, line_index, first_frame)

    # cv.imshow("image", img)
    # cv.waitKey()

    tracking_obj = HomographyTracking(first_frame, first_frame_mat)

    points3d_on_field = uniform_point_sample_on_field(25, 18, 25, 18)

    pan = [first_frame_ptz[0]]
    tilt = [first_frame_ptz[1]]
    f = [first_frame_ptz[2]]

    for i in range(2400, 3000):
        next_frame = sequence.get_image_gray(index=i, dataset_type=2)
        tracking_obj.tracking(next_frame)

        # img = project_with_homography(tracking_obj.accumulate_matrix[-1], points, line_index, next_frame)

        # compute ptz

        first_camera.set_ptz((pan[-1], tilt[-1], f[-1]))
        pose = estimate_camera_from_homography(
            tracking_obj.accumulate_matrix[-1], first_camera, points3d_on_field
        )

        print("-----" + str(i) + "--------")

        # print(tracking_obj.each_homography[-1])

        print(pose)

        # first_camera.set_ptz(pose)
        # img2 = project_with_PTZCamera(first_camera, points, line_index, next_frame)

        print("%f" % (pose[0] - gt_pan[i]))
        print("%f" % (pose[1] - gt_tilt[i]))
        print("%f" % (pose[2] - gt_f[i]))

        pan.append(pose[0])
        tilt.append(pose[1])
        f.append(pose[2])

        # cv.imshow("image", img)
        # cv.imshow("image2", img2)
        # cv.waitKey(0)

    save_camera_pose(
        np.array(pan),
        np.array(tilt),
        np.array(f),
        "C:/graduate_design/experiment_result/baseline2/2-gauss.mat",
    )


if __name__ == "__main__":
    synthesized_test()
