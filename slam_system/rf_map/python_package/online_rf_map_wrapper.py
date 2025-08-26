# online random forest as map
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
        "/Users/jacobedwards/dev/Pan-tilt-zoom-SLAM/slam_system/rf_map/build/librf_map_python.dylib"
    )


class OnlineRFMap:
    rf_file: str
    rf_map: c_void_p

    def __init__(self: Self, rf_file: str) -> None:
        self.rf_file = rf_file

        lib.OnlineRFMap_new.restype = c_void_p
        self.rf_map = lib.OnlineRFMap_new()
        # print('rf_map value 1 {}'.format(self.rf_map))

    def create_map(self: Self, feature_label_file: str, tree_param_file: str) -> None:
        """
        :param feature_label_file: a .mat file has 'keypoint', 'descriptor' and 'ptz'
        :param tree_param_file:
        :return:
        """

        fl_file = feature_label_file.encode()
        tr_file = tree_param_file.encode()
        rf_file = self.rf_file.encode()
        lib.createOnlineMap.argtypes = [c_void_p, c_char_p, c_char_p, c_char_p]

        # print('rf_map point in python {}'.format(self.rf_map))
        # print('rf_map value 2 {}'.format(self.rf_map))
        lib.createOnlineMap(self.rf_map, fl_file, tr_file, rf_file)

    def update_map(self: Self, feature_label_file: str) -> None:
        """
        add a feature label file to the model
        :param feature_label_file:
        :return:
        """

        fl_file = feature_label_file.encode()
        rf_file = self.rf_file.encode()

        lib.updateOnlineMap.argtypes = [c_void_p, c_char_p, c_char_p]
        lib.updateOnlineMap(self.rf_map, fl_file, rf_file)

    def relocalization(self: Self, feature_location_file: str, init_pan_tilt_zoom: np.ndarray) -> np.ndarray:
        """
        :param feature_location_file: .mat file has 'keypoint' and 'descriptor'
        :param init_pan_tilt_zoom, 3 x 1, initial camera parameter
        :return:
        """
        feature_location_file = feature_location_file.encode()
        test_parameter_file = "".encode()
        pan_tilt_zoom = init_pan_tilt_zoom.reshape((3, 1))

        lib.relocalizeCameraOnline.argtypes = [c_void_p, c_char_p, c_char_p, c_void_p]

        # print('rf_map value 4 {}'.format(self.rf_map))
        lib.relocalizeCameraOnline(
            self.rf_map,
            feature_location_file,
            test_parameter_file,
            c_void_p(pan_tilt_zoom.ctypes.data),
        )
        return pan_tilt_zoom

