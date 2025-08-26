# random forest as map
from typing import Self

import numpy as np
from ctypes import cdll
from ctypes import c_void_p
from ctypes import c_char_p
from ctypes import Structure
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


class BTDTRegressor(Structure):
    pass


class RFMap:
    """
    The map is saved on files.
    """
    rf_file_name: str
    rf: c_void_p | None # a point to random forest

    def __init__(self: Self, rf_file_name: str) -> None:
        self.rf_file_name = rf_file_name
        self.rf = None

    def createMap(self: Self, feature_label_files: str, tree_param_file: str) -> None:
        """
        :param tree_param_file:
        :param feature_label_files: .mat file has 'keypoint', 'descriptor' and 'ptz'
        :return:
        """
        fl_file = feature_label_files.encode()
        tr_file = tree_param_file.encode()
        rf_file = self.rf_file_name.encode()
        lib.createMap.argtypes = [c_char_p, c_char_p, c_char_p]
        lib.createMap(fl_file, tr_file, rf_file)

    def relocalization(self: Self, feature_location_file: str, init_pan_tilt_zoom: np.ndarray) -> np.ndarray:
        model_name = self.rf_file_name.encode()
        feature_location_file = feature_location_file.encode()
        test_parameter_file = "".encode()
        pan_tilt_zoom = init_pan_tilt_zoom.reshape((3, 1))

        lib.relocalizeCamera.argtrypes = [c_char_p, c_char_p, c_char_p, c_void_p]
        lib.relocalizeCamera(
            model_name,
            feature_location_file,
            test_parameter_file,
            c_void_p(pan_tilt_zoom.ctypes.data),
        )
        return pan_tilt_zoom
