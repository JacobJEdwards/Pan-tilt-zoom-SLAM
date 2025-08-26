# random forest as map
from typing import Self
import numpy as np
from ctypes import cdll
from ctypes import c_void_p
from ctypes import c_char_p
import platform

system = platform.system()

# @todo hardcode library
if system == "Windows":
    lib = cdll.LoadLibrary(
        "C:/graduate_design/Pan-tilt-zoom-SLAM/slam_system/rf_map/build/x64/Debug/rf_map_python.dll"
    )
else:
    lib = cdll.LoadLibrary(
        "/Users/jimmy/Code/ptz_slam/Pan-tilt-zoom-SLAM/slam_system/rf_map/build/librf_map_python.dylib"
    )


class RFMap:
    rf_file: str
    rf_map: c_void_p

    def __init__(self: Self, rf_file: str) -> None:
        self.rf_file = rf_file
        self.rf_map = lib.RFMap_new()

        lib.RFMap_new.restype = c_void_p

        print("rf_map value 1 {}".format(self.rf_map))

    def create_map(self: Self, feature_label_files: str, tree_param_file: str) -> None:
        """
        :param tree_param_file:
        :param feature_label_files: .mat file has 'keypoint', 'descriptor' and 'ptz'
        :return:
        """
        fl_file = feature_label_files.encode()
        tr_file = tree_param_file.encode()
        rf_file = self.rf_file.encode()
        lib.createMap.argtypes = [c_void_p, c_char_p, c_char_p, c_char_p]

        # print('rf_map point in python {}'.format(self.rf_map))
        print("rf_map value 2 {}".format(self.rf_map))
        lib.createMap(self.rf_map, fl_file, tr_file, rf_file)
        print("rf_map value 3 {}".format(self.rf_map))

    def relocalization(self: Self, feature_location_file: str, init_pan_tilt_zoom: str) -> np.ndarray:
        """
        :param feature_location_file: .mat file has 'keypoint' and 'descriptor'
        :param init_pan_tilt_zoom, 3 x 1, initial camera parameter
        :return:
        """
        feature_location_file = feature_location_file.encode()
        test_parameter_file = "".encode()
        pan_tilt_zoom = np.zeros((3, 1))
        for i in range(3):
            pan_tilt_zoom[i] = init_pan_tilt_zoom[i]

        lib.relocalizeCamera.argtypes = [c_void_p, c_char_p, c_char_p, c_void_p]

        print("rf_map value 4 {}".format(self.rf_map))
        lib.relocalizeCamera(
            self.rf_map,
            feature_location_file,
            test_parameter_file,
            c_void_p(pan_tilt_zoom.ctypes.data),
        )
        return pan_tilt_zoom

    @staticmethod
    def estimateCameraRANSAC(keypoint_ray_file_name: str, init_pan_tilt_zoom: np.ndarray) -> np.ndarray:
        """
        :param keypoint_ray_file_name: .mat file has 'keypoints' and 'rays'
        :param init_pan_tilt_zoom: 3 x 1, initial camera parameter
        :return:
        """

        keypoint_ray_file_name = keypoint_ray_file_name.encode()
        pan_tilt_zoom = np.zeros((3, 1))
        for i in range(3):
            pan_tilt_zoom[i] = init_pan_tilt_zoom[i]

        lib.estimateCameraRANSAC.argtypes = [c_char_p, c_void_p]

        lib.estimateCameraRANSAC(
            keypoint_ray_file_name, c_char_p(pan_tilt_zoom.ctypes.data)
        )

        return pan_tilt_zoom
