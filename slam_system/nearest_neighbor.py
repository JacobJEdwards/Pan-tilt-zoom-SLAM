"""
Created by Jikai Lu, 2019.7

A nearest neighbor search based relocalization solution.
"""

import numpy as np
import scipy.io as sio
import cv2 as cv

from scipy.spatial import KDTree
from scene_map import Map
from ptz_camera import PTZCamera
from scipy.optimize import least_squares
from transformation import TransFunction


class NNBasedMap(Map):
    def __init__(self):
        super(NNBasedMap, self).__init__("sift")
        # [N, 128] descriptor array
        self.global_des = np.empty((0, 128), dtype=np.float32)

        # init KDTree (replaces FLANN)
        self.tree = None

    def build_kdtree(self):
        if self.global_des.shape[0] > 0:
            self.tree = KDTree(self.global_des)
        else:
            self.tree = None

    def find_nearest(self, des):
        if self.tree is None:
            self.build_kdtree()

        dist, result = self.tree.query(des, k=1)
        result = result.flatten()
        dist = dist.flatten()

        matched_keypoint_index = []
        matched_ray_index = []
        for i in range(len(result)):
            if dist[i] < 2000:
                matched_keypoint_index.append(i)
                matched_ray_index.append(result[i])

        return matched_keypoint_index, matched_ray_index

    def add_keyframe_without_ba(self, keyframe, verbose=False):
        # keyframe.convert_keypoint_to_array(norm=False)
        super(NNBasedMap, self).add_keyframe_without_ba(keyframe, verbose)

        camera = PTZCamera(
            (keyframe.u, keyframe.v), keyframe.center, keyframe.base_rotation
        )
        camera.set_ptz((keyframe.pan, keyframe.tilt, keyframe.f))

        rays = camera.back_project_to_rays(keyframe.feature_pts)

        self.global_ray = np.vstack([self.global_ray, rays])
        self.global_des = np.vstack([self.global_des, keyframe.feature_des])

    def add_keyframes(self, keyframe_list):
        for keyframe in keyframe_list:
            self.add_keyframe_without_ba(keyframe)

        self.build_kdtree()

    @staticmethod
    def compute_residual(pose, rays, points, u, v):
        """
        :param pose: shape [3] array of camera pose
        :param rays: [N, 2] array of corresponding rays
        :param points: [N, 2] array of corresponding points
        :param u: camera u
        :param v: camera v
        :return: reprojection error of these points
        """

        residual = np.zeros(2 * len(rays))

        for i in range(len(rays)):
            reproject_x, reproject_y = TransFunction.from_ray_to_image(
                u, v, pose[2], pose[0], pose[1], rays[i, 0], rays[i, 1]
            )
            residual[2 * i] = reproject_x - points[i, 0]
            residual[2 * i + 1] = reproject_y - points[i, 1]

        return residual

    def relocalize(self, keyframe):
        keypoint_index, ray_index = self.find_nearest(keyframe.feature_des)

        rays = self.global_ray[ray_index]

        # camera = PTZCamera((keyframe.u, keyframe.v), keyframe.center, keyframe.base_rotation)

        pose = np.array([keyframe.pan, keyframe.tilt, keyframe.f])

        # fuck = self.compute_residual(pose, rays, keyframe.feature_pts, keyframe.u, keyframe.v)

        optimized_pose = least_squares(
            NNBasedMap.compute_residual,
            pose,
            verbose=2,
            x_scale="jac",
            ftol=1e-4,
            args=(rays, keyframe.feature_pts[keypoint_index], keyframe.u, keyframe.v),
        )

        # save to mat
        keyframe_data = dict()

        keyframe_data["rays"] = rays
        keyframe_data["keypoints"] = keyframe.feature_pts[keypoint_index]

        # convert the base rotation to (3, 1)
        save_br = np.ndarray([3, 1])
        if keyframe.base_rotation.shape == (3, 3):
            save_br, _ = cv.Rodrigues(keyframe.base_rotation, save_br)
        else:
            save_br = keyframe.base_rotation
        save_br = save_br.ravel()

        keyframe_data["camera"] = np.array(
            [
                keyframe.u,
                keyframe.v,
                keyframe.f,
                save_br[0],
                save_br[1],
                save_br[2],
                keyframe.center[0],
                keyframe.center[1],
                keyframe.center[2],
            ]
        ).reshape(-1, 1)

        keyframe_data["ptz"] = np.array(
            [keyframe.pan, keyframe.tilt, keyframe.f]
        ).reshape(-1, 1)

        sio.savemat(
            "../nn_test_data/outliers-0/" + str(keyframe.img_index) + ".mat",
            mdict=keyframe_data,
        )

        return optimized_pose.x


def ut_estimateCameraRANSAC():
    for i in range(0, 3600, 120):
        file_name = "C:/graduate_design/Pan-tilt-zoom-SLAM/nn_test_data/outliers-0/{}.mat".format(
            i
        )
        data = sio.loadmat(file_name)

        ptz_gt = data["ptz"]
        init_ptz = np.array([0, 0, 3000])

        rays = data["rays"]
        keypoints = data["keypoints"]

        optimized_pose = least_squares(
            NNBasedMap.compute_residual,
            init_ptz,
            x_scale="jac",
            ftol=1e-4,
            args=(rays, keypoints, 640, 360),
        )

        print("estimated ptz is {}".format(optimized_pose.x))
        print("ground truth ptz is {}".format(ptz_gt))


if __name__ == "__main__":
    ut_estimateCameraRANSAC()
