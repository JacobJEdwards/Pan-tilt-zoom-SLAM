"""
general image processing functions

Create by Jimmy and Luke, 2018.9
"""

import cv2 as cv
import numpy as np
import random


def detect_sift(im: np.ndarray, nfeatures: int=50) -> np.ndarray:
    if len(im.shape) == 3 and im.shape[0] == 3:
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT.create(nfeatures=nfeatures)
    kp = sift.detect(im, None)

    if 0 < nfeatures < len(kp):
        kp = kp[:nfeatures]

    sift_pts = np.zeros((len(kp), 2), dtype=np.float32)
    for i in range(len(kp)):
        sift_pts[i][0] = kp[i].pt[0]
        sift_pts[i][1] = kp[i].pt[1]

    return sift_pts


def detect_orb(im: np.ndarray, nfeatures: int=1000) -> np.ndarray:
    """
    :param im: gray or color image
    :param nfeatures:
    :return:  N x 2 matrix, ORB keypoint location in the image
    """
    assert nfeatures > 0
    if len(im.shape) == 3 and im.shape[0] == 3:
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    orb = cv.ORB.create(nfeatures)
    keypoints = orb.detect(im, None)

    N = len(keypoints)
    pts = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        pts[i][0], pts[i][1] = keypoints[i].pt[0], keypoints[i].pt[1]
    return pts


def detect_compute_sift(im: np.ndarray, nfeatures: int, verbose: bool=False) -> tuple[list, np.ndarray]:
    """
    :param im: RGB or gray image
    :param nfeatures:
    :return: two lists of key_point (2 dimension), and descriptor (128 dimension)
    """
    # pre-processing if input is color image
    assert isinstance(im, np.ndarray)
    # if len(im.shape) == 3 and im.shape[0] == 3:
    #     im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT.create(nfeatures=nfeatures)
    key_point, descriptor = sift.detectAndCompute(im, None)

    """SIFT may detect more keypoint than set"""

    if 0 < nfeatures < len(key_point):
        key_point = key_point[:nfeatures]
        descriptor = descriptor[:nfeatures]

    if verbose:
        print("detect: %d SIFT keypoints." % len(key_point))

    return key_point, descriptor


def detect_compute_sift_array(im: np.ndarray, nfeatures: int, norm: bool=True) -> tuple[np.ndarray, np.ndarray]:
    """
    an option for SIFT keypoints detection if arrays are needed.
    :param im: RGB or gray image
    :param nfeatures: number of SIFT keypoints
    :return: two numpy array of shape (N, 2) and (N, 128)
    """
    keyframe_kp, keyframe_des = detect_compute_sift(im, nfeatures)

    array_pts = np.zeros((len(keyframe_kp), 2), dtype=np.float64)
    for i in range(len(keyframe_kp)):
        array_pts[i][0] = keyframe_kp[i].pt[0]
        array_pts[i][1] = keyframe_kp[i].pt[1]

    if norm:
        norm = np.linalg.norm(keyframe_des, axis=1).reshape(-1, 1)
        array_des = np.divide(keyframe_des, norm).astype(np.float64)
    else:
        array_des = keyframe_des

    return array_pts, array_des


def detect_compute_orb(im: np.ndarray, nfeatures: int=1000, verbose: bool=False) -> tuple[list, np.ndarray]:
    """
    :param im: gray or color image
    :param nfeatures: 0 for zero features
    :param verbose:
    :return:
    """

    assert nfeatures > 0

    orb = cv.ORB.create(nfeatures)
    key_point = orb.detect(im, None)
    key_point, descriptor = orb.compute(im, key_point)

    if len(key_point) > nfeatures:
        key_point = key_point[:nfeatures]
        descriptor = descriptor[:nfeatures]

    if verbose:
        print("detect: %d ORB keypoints." % len(key_point))
    return list(key_point), descriptor

def detect_compute_latch(im: np.ndarray, nfeatures: int=1500, verbose: bool=False) -> tuple[list, np.ndarray]:
    """
    :param im: RGB or gray image
    :param nfeatures:
    :return: two lists of key_point (2 dimension), and descriptor (128 dimension)
    """
    # pre-processing if input is color image
    if len(im.shape) == 3 and im.shape[0] == 3:
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    orb = cv.ORB.create(nfeatures)
    kp_orb = orb.detect(im, None)

    latch = cv.LATCH.create(64)
    kp_latch, des_latch = latch.compute(im, kp_orb)

    """LATCH may detect more keypoint than set"""

    if nfeatures > 0 and len(kp_latch) > nfeatures:
        kp_latch = kp_latch[:nfeatures]
        des_latch = des_latch[:nfeatures]

    if verbose == True:
        print("detect: %d LATCH keypoints." % len(kp_latch))

    return kp_latch, des_latch


def keypoints_masking(kp: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    use bounding box to remove keypoints on players
    :param kp: list [N] of keypoints object (need use .pt to access point location)
    :param mask: bounding box mask
    :return: index array for keypoints out of players
    """
    inner_index = np.ndarray([0], dtype=np.int32)
    for i in range(len(kp)):
        if isinstance(kp, np.ndarray):
            x, y = int(kp[i, 0]), int(kp[i, 1])
        else:
            x, y = int(kp[i].pt[0]), int(kp[i].pt[1])

        if mask[y, x] == 1:
            inner_index = np.append(inner_index, i)
    return inner_index


def match_sift_features(
    keypiont1: np.ndarray,
        descriptor1,
        keypoint2,
        descriptor2: np.ndarray,
        pts_array: bool=False, verbose: bool=False
):
    # from https://opencv-python-tutroals.readthedocs.io/en
    # /latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
    """
    :param keypiont1: list of keypoints
    :param descriptor1:
    :param keypoint2:
    :param descriptor2:
    :param verbose:
    :return: matched 2D points, and matched descriptor index
    : pts1, index1, pts2, index2
    """

    bf = cv.BFMatcher()
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)  # (query_data, train_data)

    # step 1: apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if verbose == True:
        print("%d matches passed the ratio test" % len(good))

    N = len(good)
    if N <= 8:
        print("warning: match sift features failed, not enough matching")
        return None, [], None, []

    pts1 = np.zeros((N, 2))
    pts2 = np.zeros((N, 2))
    index1 = np.zeros((N), dtype=np.int32)
    index2 = np.zeros((N), dtype=np.int32)
    for i in range(N):
        idx1, idx2 = good[i].queryIdx, good[i].trainIdx  # query is from the first image
        index1[i], index2[i] = idx1, idx2

        if pts_array:
            pts1[i] = keypiont1[idx1]
            pts2[i] = keypoint2[idx2]
        else:
            pts1[i] = keypiont1[idx1].pt
            pts2[i] = keypoint2[idx2].pt

    # step 2: apply homography constraint
    # inlier index from homography estimation
    inlier_index = homography_ransac(pts1, pts2, 1.0)

    if verbose == True:
        print("%d matches passed the homography ransac" % len(inlier_index))

    pts1, pts2 = pts1[inlier_index, :], pts2[inlier_index, :]
    index1 = index1[inlier_index].tolist()
    index2 = index2[inlier_index].tolist()

    return pts1, index1, pts2, index2


def match_orb_features(keypiont1: list, descriptor1: np.ndarray, keypoint2: list
                       , descriptor2: np.ndarray, verbose: bool=False) -> tuple[np.ndarray, list, np.ndarray, list]:
    """
    :param keypiont1: list of keypoints
    :param descriptor1:
    :param keypoint2:
    :param descriptor2:
    :param verbose:
    :return: matched 2D points, and matched descriptor index
    """
    assert len(keypiont1) >= 4  # assume homography matching

    # step 1: matching by hamming distance
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptor1, descriptor2)

    # step 2: remove outlier using RANSAC  @todo this code is same (redundant) as in match_sift_features
    N = len(matches)
    pts1, pts2 = np.zeros((N, 2)), np.zeros((N, 2))
    index1 = np.zeros((N), dtype=np.int32)
    index2 = np.zeros((N), dtype=np.int32)
    for i in range(N):
        idx1, idx2 = (
            matches[i].queryIdx,
            matches[i].trainIdx,
        )  # query is from the first image
        index1[i], index2[i] = idx1, idx2
        pts1[i] = keypiont1[idx1].pt
        pts2[i] = keypoint2[idx2].pt

    # inlier index from homography estimation
    inlier_index = homography_ransac(pts1, pts2, 1.0)

    if verbose == True:
        print("%d matches passed the homography ransac" % len(inlier_index))

    pts1, pts2 = pts1[inlier_index, :], pts2[inlier_index, :]
    index1 = index1[inlier_index].tolist()
    index2 = index2[inlier_index].tolist()
    return pts1, index1, pts2, index2


def match_latch_features(keypiont1: list, descriptor1: np.ndarray, keypoint2: list, descriptor2: np.ndarray,
                         verbose: bool=False) -> tuple[np.ndarray, list, np.ndarray, list]:
    """
    :param keypiont1: list of keypoints
    :param descriptor1:
    :param keypoint2:
    :param descriptor2:
    :param verbose:
    :return: matched 2D points, and matched descriptor index
    """
    assert len(keypiont1) >= 4  # assume homography matching

    # step 1: matching by hamming distance
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptor1, descriptor2)

    # step 2: remove outlier using RANSAC  @todo this code is same (redundant) as in match_sift_features
    N = len(matches)
    pts1, pts2 = np.zeros((N, 2)), np.zeros((N, 2))
    index1 = np.zeros((N), dtype=np.int32)
    index2 = np.zeros((N), dtype=np.int32)
    for i in range(N):
        idx1, idx2 = (
            matches[i].queryIdx,
            matches[i].trainIdx,
        )  # query is from the first image
        index1[i], index2[i] = idx1, idx2
        pts1[i] = keypiont1[idx1].pt
        pts2[i] = keypoint2[idx2].pt

    # inlier index from homography estimation
    inlier_index = homography_ransac(pts1, pts2, 1.0)

    if verbose == True:
        print("%d matches passed the homography ransac" % len(inlier_index))

    pts1, pts2 = pts1[inlier_index, :], pts2[inlier_index, :]
    index1 = index1[inlier_index].tolist()
    index2 = index2[inlier_index].tolist()
    return pts1, index1, pts2, index2


def compute_homography(keypiont1: list, descriptor1: np.ndarray, keypoint2: list, descriptor2: np.ndarray) -> np.ndarray:
    """
    Get the estimated homography matrix from two sets of keypoints with descriptor.
    :param keypiont1:
    :param descriptor1:
    :param keypoint2:
    :param descriptor2:
    :return:
    """
    bf = cv.BFMatcher()
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)  # (query_data, train_data)

    # step 1: apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    N = len(good)
    if N <= 8:
        print("warning: match sift features failed, not enough matching")
        return None, [], None, []

    pts1 = np.zeros((N, 2))
    pts2 = np.zeros((N, 2))
    for i in range(N):
        idx1, idx2 = good[i].queryIdx, good[i].trainIdx  # query is from the first image
        pts1[i] = keypiont1[idx1].pt
        pts2[i] = keypoint2[idx2].pt

    # step 2: apply homography constraint
    # inlier index from homography estimation

    homography, _ = cv.findHomography(
        srcPoints=pts1, dstPoints=pts2, ransacReprojThreshold=1.0, method=cv.FM_RANSAC
    )

    return homography


def detect_harris_corner_grid(gray_img: np.ndarray, row: int, column: int) -> np.ndarray:
    """
    :param gray_img:
    :param row:
    :param column:
    :return: harris corner in shape (n ,2)
    """

    mask = np.zeros_like(gray_img, dtype=np.uint8)

    grid_height = gray_img.shape[0] // row
    grid_width = gray_img.shape[1] // column

    all_harris = np.ndarray([0, 1, 2], dtype=np.float32)

    for i in range(row):
        for j in range(column):
            mask.fill(0)
            grid_y1 = i * grid_height
            grid_x1 = j * grid_width

            if i == row - 1:
                grid_y2 = gray_img.shape[0]
            else:
                grid_y2 = i * grid_height + grid_height

            if j == column - 1:
                grid_x2 = gray_img.shape[1]
            else:
                grid_x2 = j * grid_width + grid_width

            mask[grid_y1:grid_y2, grid_x1:grid_x2] = 1
            grid_harris = cv.goodFeaturesToTrack(
                gray_img,
                maxCorners=20,
                qualityLevel=0.2,
                minDistance=10,
                mask=mask.astype(np.uint8),
            )

            if grid_harris is not None:
                all_harris = np.concatenate([all_harris, grid_harris], axis=0)

    return all_harris.reshape([-1, 2])


def optical_flow_matching(img: np.ndarray, next_img: np.ndarray, points: np.ndarray, ssd_threshold: int=20) -> tuple[list, np.ndarray]:
    """
    :param img:    current image
    :param next_img: next image
    :param points: points on the current image
    :param ssd_threshold: optical flow parameters
    :return: matched index in the points, points in the next image. two lists
    """
    points = points.reshape((-1, 1, 2))  # 2D matrix to 3D matrix
    next_points, status, err = cv.calcOpticalFlowPyrLK(
        img, next_img, points.astype(np.float32), None, winSize=(31, 31)
    )

    h, w = img.shape[0], img.shape[1]
    matched_index = []

    for i in range(len(next_points)):
        x, y = next_points[i, 0, 0], next_points[i, 0, 1]
        if err[i] < ssd_threshold and 0 < x < w and 0 < y < h:
            matched_index.append(i)

    next_points = np.array([next_points[i, 0] for i in matched_index])

    return matched_index, next_points


def homography_ransac(
    points1: np.ndarray, points2: np.ndarray, reprojection_threshold: float=0.5, return_matrix: bool=False
) -> tuple[list, np.ndarray] | list:
    """
    Homography based RANSAC.
    Try to find the homography matrix and RANSAC inliers index.
    :param points1: [N, 2] matched points
    :param points2: [N, 2] matched points
    :param reprojection_threshold:
    :param return_matrix: True for get homography matrix with RANSAC inlier index
    :return: RANSAC inlier index, e.g. matched index in original points, [0, 3, 4...]
            and the homography matrix if return_matrix is True
    """
    # check parameter
    assert points1.shape[0] == points2.shape[0]
    assert points1.shape[0] >= 4
    ransac_mask = np.ndarray([len(points1)])
    homography, ransac_mask = cv.findHomography(
        srcPoints=points1,
        dstPoints=points2,
        ransacReprojThreshold=reprojection_threshold,
        method=cv.FM_RANSAC,
        mask=ransac_mask,
    )

    index = [i for i in range(len(ransac_mask)) if ransac_mask[i] == 1]

    if return_matrix:
        return index, homography
    else:
        return index


def matching_and_ransac(
    img1, img2, img1_keypoints, img1_keypoints_index, visualize=False
):
    """
    matching with sparse optical flow and run ransac to get homography based inliers.
    :param img1: image 1
    :param img2: image 2
    :param img1_keypoints: keypoints in image 1 [n, 2]
    :param img1_keypoints_index: keypoints corresponding global indexes
    :return: inliers in current frame(img2), inliers global indexes, outliers global indexes
    """

    # local_matched_index is matched index in img1_keypoints (or current_keypoints)
    # current_keypoints is matched keypoints in current frame (img2)
    local_matched_index, current_keypoints = optical_flow_matching(
        img1, img2, img1_keypoints
    )

    # current_keypoints_index is matched keypoints indexes in corresponding rays.
    current_keypoints_index = img1_keypoints_index[local_matched_index]

    # previous_matched_keypoints is matched keypoints in previous frame.
    previous_matched_keypoints = img1_keypoints[local_matched_index]

    # run RANSAC, local_inlier_index is index in the input for ransac
    local_inlier_index = homography_ransac(
        previous_matched_keypoints, current_keypoints, reprojection_threshold=0.5
    )

    # inlier_keypoints is keypoints in current frame (img2) after 1: optical flow 2: homography RANSAC
    inlier_keypoints = current_keypoints[local_inlier_index]

    # previous_inlier_keypoints is keypoints in previous frame (img2) after 1: optical flow 2: homography RANSAC
    previous_inlier_keypoints = previous_matched_keypoints[local_inlier_index]

    # an option to show the matching result for each frame
    if visualize is True:
        vis = draw_matches(img1, img2, previous_inlier_keypoints, inlier_keypoints)
        cv.imshow("test", vis)
        cv.waitKey(0)

    # inlier_index is inliers global ray indexes
    inlier_index = current_keypoints_index[local_inlier_index]

    # outlier_index is outliers global ray indexes
    outlier_index = np.delete(current_keypoints_index, local_inlier_index, axis=0)

    return inlier_keypoints, inlier_index, outlier_index


def build_matching_graph(
    images, image_match_mask=None, feature_method="sift", verbose=False
):
    """
    build a graph for a list of images
    The graph is 2D hash map using list index as key
    node: image
    edge: matched key points and a global index (from zero)
    :param images: RGB image or gay Image
    :image_match_mask: optional N * N a list of list [[]], 1 for matched, 0 or can not match
    :feature_method, 'sift', 'orb'
    :param verbose:
    :return: keypoints, points,descriptors, src_pt_index, dst_pt_index, landmark_index (global index), landmark_num
    """
    if image_match_mask is None:
        image_match_mask = []
    assert (
        feature_method == "sift" or feature_method == "orb" or feature_method == "latch"
    )
    N = len(images)
    if verbose:
        print("build a matching graph from %d images." % N)

    if len(image_match_mask) != 0:
        assert len(image_match_mask) == N
        # check image match mask
        for mask in image_match_mask:
            assert len(mask) == N
        if verbose:
            print("image match is used")
    else:
        print("Warning: image match mask is NOT used, may have false positive matches!")

    # step 1: extract key points and descriptors
    keypoints, descriptors = [], []
    for im in images:
        kp, des = None, None
        if feature_method == "sift":
            kp, des = detect_compute_sift(im, 1500)
        elif feature_method == "orb":
            kp, des = detect_compute_orb(im, 6000)
        elif feature_method == "latch":
            kp, des = detect_compute_latch(im, 5000)
        keypoints.append(kp)
        descriptors.append(des)

    # step 2: pair-wise matching between images
    # A temporal class to store local matching result
    class Node:
        def __init__(self, kp, des):
            self.key_points = kp
            self.descriptors = des

            # local matches
            self.dest_image_index = []  # destination
            self.src_kp_index = []  # list of list
            self.dest_kp_index = []  # list of list

    nodes = []  # node in the graph
    for i in range(N):
        node = Node(keypoints[i], descriptors[i])
        nodes.append(node)

    # compute and store local matches
    min_match_num = 20  # 4 * 3
    max_match_num = 200
    for i in range(N):
        kp1, des1 = keypoints[i], descriptors[i]
        for j in range(i + 1, N):
            # skip un-matched frames
            if len(image_match_mask) != 0 and image_match_mask[i][j] == 0:
                continue

            kp2, des2 = keypoints[j], descriptors[j]
            if feature_method == "sift":
                pts1, index1, pts2, index2 = match_sift_features(
                    kp1, des1, kp2, des2
                )
            elif feature_method == "orb":
                pts1, index1, pts2, index2 = match_orb_features(
                    kp1, des1, kp2, des2
                )
            elif feature_method == "latch":
                pts1, index1, pts2, index2 = match_latch_features(
                    kp1, des1, kp2, des2
                )

            # matching is not found
            assert len(index1) == len(index2)
            if len(index1) > min_match_num:
                # randomly remove some matches
                if len(index1) > max_match_num:
                    rand_list = list(range(len(index1)))
                    random.shuffle(rand_list)
                    rand_list = rand_list[0:max_match_num]
                    index1 = [index1[idx] for idx in rand_list]
                    index2 = [index2[idx] for idx in rand_list]

                # match from image 2 to image 1
                nodes[i].dest_image_index.append(j)
                nodes[i].src_kp_index.append(index1)
                nodes[i].dest_kp_index.append(index2)
                if verbose == True:
                    print("%d matches between image: %d and %d" % (len(index1), i, j))
            else:
                if verbose == True:
                    print("no enough matches between image: %d and %d" % (i, j))

    # step 3: matching consistency check @todo

    # step 4: compute global landmark index
    landmark_index_map = dict.fromkeys(range(N))
    for i in range(N):
        landmark_index_map[i] = dict()

    g_index = 0  # global ray index
    for i in range(len(nodes)):
        node = nodes[i]
        for j, src_idx, dest_idx in zip(
            node.dest_image_index, node.src_kp_index, node.dest_kp_index
        ):
            # check each key point index
            for idx1, idx2 in zip(src_idx, dest_idx):
                # update index of landmarks
                if idx1 in landmark_index_map[i] and idx2 in landmark_index_map[j]:
                    if landmark_index_map[i][idx1] != landmark_index_map[j][idx2]:
                        print(
                            "Warning: in-consistent matching result! (%d %d) <--> (%d %d)"
                            % (i, idx1, j, idx2)
                        )
                elif (
                    idx1 in landmark_index_map[i] and idx2 not in landmark_index_map[j]
                ):
                    landmark_index_map[j].update({idx2: landmark_index_map[i][idx1]})
                elif (
                    idx1 not in landmark_index_map[i] and idx2 in landmark_index_map[j]
                ):
                    landmark_index_map[i].update({idx1: landmark_index_map[j][idx2]})
                else:
                    landmark_index_map[i].update({idx1: g_index})
                    landmark_index_map[j].update({idx2: g_index})
                    g_index += 1

    if verbose:
        print("number of landmark is %d" % g_index)
    landmark_num = g_index

    # re-organize keypoint index
    src_pt_index = [[[] for i in range(N)] for i in range(N)]
    dst_pt_index = [[[] for i in range(N)] for i in range(N)]
    landmark_index = [[[] for i in range(N)] for i in range(N)]
    for i in range(len(nodes)):
        node = nodes[i]
        for j, src_idx, dest_idx in zip(
            node.dest_image_index, node.src_kp_index, node.dest_kp_index
        ):
            src_pt_index[i][j] = src_idx
            dst_pt_index[i][j] = dest_idx
            for idx1 in src_idx:
                landmark_index[i][j].append(landmark_index_map[i][idx1])

    # change format of keypoints
    def keypoint_to_matrix(key_points):
        N = len(key_points)
        key_points_mat = np.zeros((N, 2))
        for i in range(len(key_points)):
            key_points_mat[i] = key_points[i].pt
        return key_points_mat

    # a list of N x 2 matrix
    points = [keypoint_to_matrix(keypoints[i]) for i in range(len(keypoints))]

    # step 5: output result to key frames
    return (
        keypoints,
        descriptors,
        points,
        src_pt_index,
        dst_pt_index,
        landmark_index,
        landmark_num,
    )


def visualize_points(img, points, pt_color, rad):
    """draw some colored points in img"""
    for j in range(len(points)):
        cv.circle(
            img,
            (int(points[j][0]), int(points[j][1])),
            color=pt_color,
            radius=rad,
            thickness=2,
        )


def draw_matches(im1, im2, pts1, pts2):
    """
    :param im1: RGB image
    :param im2:
    :param pts1:  N * 2 matrix, points in image 1
    :param pts2:  N * 2 matrix, points in image 2
    :return: lines overlaid on the original image
    """
    # step 1: horizontal concat image
    vis = np.concatenate((im1, im2), axis=1)
    w = im1.shape[1]
    N = pts1.shape[0]
    # step 2:draw lines
    for i in range(N):
        p1, p2 = pts1[i], pts2[i]
        p1 = p1.astype(np.int32)
        p2 = p2.astype(np.int32)
        p2[0] += w
        cv.line(vis, (p1[0], p1[1]), (p2[0], p2[1]), (0, 255, 0), thickness=1)
    return vis


def blur_sub_image(im, x, y, w, h, kernal_size=31):
    """
    @blur an image area
    :param im: image
    :param x: left top corner (x, y)
    :param y:
    :param w: sub image width and height (w, h)
    :param h:
    :param kernal_size: blur kernel size
    :return:
    """
    im[y : y + h, x : x + w] = cv.blur(
        im[y : y + h, x : x + w], (kernal_size, kernal_size)
    )
    return im


def ut_blur_sub_image():
    im = cv.imread("./seq3/00000515.jpg")
    im = blur_sub_image(im, 60, 60, 430, 40)
    cv.imshow("test", im)
    cv.waitKey(0)

    for i in range(333):
        im = cv.imread("./seq3/00000" + str(515 + i) + ".jpg")
        im = blur_sub_image(im, 60, 60, 430, 40)
        print(i)
        cv.imwrite("./seq3_blur/00000" + str(515 + i) + ".jpg", im)


def ut_match_sift_features():
    # im1 = cv.imread('/Users/jimmy/Desktop/ptz_slam_dataset/basketball/images/00084000.jpg', 1)
    # im2 = cv.imread('/Users/jimmy/Desktop/ptz_slam_dataset/basketball/images/00084660.jpg', 1)

    # im1 = cv.imread("./basketball/basketball/images/00084711.jpg")
    # im2 = cv.imread("./basketball/basketball/images/00084734.jpg")

    # im1 = cv.imread("./seq3_blur/00000733.jpg")
    # im2 = cv.imread("./seq3_blur/00000800.jpg")

    im1 = cv.imread("../../dataset/hockey/UBC_2017/images/00048600.jpg")
    im2 = cv.imread("../../dataset/hockey/UBC_2017/images/00048601.jpg")

    kp1, des1 = detect_compute_sift(im1, 2000, True)
    kp2, des2 = detect_compute_sift(im2, 2000, True)

    print(type(des1[0]))
    print(des1[0].shape)

    pt1, index1, pt2, index2 = match_sift_features(kp1, des1, kp2, des2, True)

    im3 = draw_matches(im1, im2, pt1, pt2)
    cv.imshow("matches", im3)
    cv.waitKey(0)
    # print('image shape:', im1.shape)


def ut_build_matching_graph():
    im0 = cv.imread(
        "/Users/jimmy/Desktop/ptz_slam_dataset/basketball/images/00084000.jpg", 1
    )
    im1 = cv.imread(
        "/Users/jimmy/Desktop/ptz_slam_dataset/basketball/images/00084660.jpg", 1
    )
    im2 = cv.imread(
        "/Users/jimmy/Desktop/ptz_slam_dataset/basketball/images/00084700.jpg", 1
    )
    im3 = cv.imread(
        "/Users/jimmy/Desktop/ptz_slam_dataset/basketball/images/00084740.jpg", 1
    )
    im4 = cv.imread(
        "/Users/jimmy/Desktop/ptz_slam_dataset/basketball/images/00084800.jpg", 1
    )

    # cv.imshow('image 0', im0)
    # cv.imshow('image 4', im4)
    # cv.waitKey(0)
    images = [im0, im1, im2, im3, im4]
    (
        keypoints,
        descriptors,
        points,
        src_pt_index,
        dst_pt_index,
        landmark_index,
        landmark_num,
    ) = build_matching_graph(images, [], "orb", True)
    print(type(points[0]))
    print(type(descriptors[0]))


def ut_orb():
    # im1 = cv.imread('/Users/jimmy/Desktop/ptz_slam_dataset/basketball/images/00084000.jpg', 1)
    # im2 = cv.imread('/Users/jimmy/Desktop/ptz_slam_dataset/basketball/images/00084660.jpg', 1)
    im1 = cv.imread("./basketball/basketball/images/00084711.jpg")
    im2 = cv.imread("./basketball/basketball/images/00084734.jpg")

    # im1 = cv.imread("./seq3_blur/00000733.jpg")
    # im2 = cv.imread("./seq3_blur/00000800.jpg")

    # pts1 = detect_orb(im1, 1000)
    # print(pts1.shape)

    kp1, des1 = detect_compute_orb(im1, 6000, True)
    kp2, des2 = detect_compute_orb(im2, 6000, True)

    pt1, index1, pt2, index2 = match_orb_features(kp1, des1, kp2, des2, True)
    im3 = draw_matches(im1, im2, pt1, pt2)
    cv.imshow("matches", im3)
    cv.waitKey(0)

    """"
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)

    matches = sorted(matches, key = lambda x:x.distance)

    vis = cv.drawKeypoints(im1, kp1, None, color=(0, 255, 0), flags=0)
    cv.imshow('orb keypoints', vis)

    vis = cv.drawMatches(im1, kp1, im2, kp2, matches[:100], None, flags=2)
    cv.imshow('org matches, first 100 keypoints', vis)
    cv.waitKey(0)
    """


def ut_latch():
    # im1 = cv.imread("./basketball/basketball/images/00084000.jpg")
    # im2 = cv.imread("./basketball/basketball/images/00084660.jpg")
    im1 = cv.imread("./basketball/basketball/images/00084711.jpg")
    im2 = cv.imread("./basketball/basketball/images/00084734.jpg")

    kp1, des1 = detect_compute_latch(im1, 3000)
    kp2, des2 = detect_compute_latch(im2, 3000)

    pt1, index1, pt2, index2 = match_latch_features(kp1, des1, kp2, des2, True)
    im3 = draw_matches(im1, im2, pt1, pt2)
    cv.imshow("matches", im3)
    cv.waitKey(0)

    # detect_latch(im1)
    #
    # orb = cv.ORB_create(1000)
    # kp1 = orb.detect(im1, None)
    #
    #
    # latch = cv.xfeatures2d.LATCH_create(16)
    # kp_l, des_l = latch.compute(im1, kp1)
    print(len(kp1))

    # kp1, des1 = detect_compute_latch(im1,1000, True)

    # print(len(kp1))

    # vis = cv.drawKeypoints(im1, kp1, None, color=(0, 255, 0), flags=0)
    # cv.imshow('latch keypoints', vis)
    # cv.waitKey(0)


def ut_redundant():
    im = cv.imread("./two_point_calib_dataset/highlights/seq1/0419.jpg", 0)
    print("image shape:", im.shape)

    # unit test
    pts = detect_sift(im, 50)
    print(pts.shape)

    kp, des = detect_compute_sift(im, 50)
    print(len(kp))
    print(len(des))
    print(des[0].shape)

    corners = detect_harris_corner_grid(im, 5, 5)
    print(len(corners))
    print(corners[0].shape)

    im1 = cv.imread("./two_point_calib_dataset/highlights/seq1/0419.jpg", 0)
    im2 = cv.imread("./two_point_calib_dataset/highlights/seq1/0422.jpg", 0)

    pts1 = detect_sift(im1, 50)
    matched_index, next_points = optical_flow_matching(im1, im2, pts1, 20)

    print(len(matched_index), len(next_points))

    cv.imshow("image", im)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    # ut_match_sift_features()
    # ut_build_matching_graph()
    # ut_blur_sub_image()
    # ut_orb()

    """
    im1 = cv.imread("./basketball/basketball/images/00084711.jpg")
    im2 = cv.imread("./basketball/basketball/images/00084734.jpg")

    start = time.time()

    for i in range(10):
        # DoG
        # kp = detect_sift(im1, 50)

        # fast
        # kp = detect_orb(im1, 50)

        # sift
        # kp1, des1 = detect_compute_sift(im1, 1500, False)
        # kp2, des2 = detect_compute_sift(im2, 1500, False)
        #
        # pt1, index1, pt2, index2 = match_sift_features(kp1, des1, kp2, des2, True)

        # orb
        # kp1, des1 = detect_compute_orb(im1, 6000, False)
        # kp2, des2 = detect_compute_orb(im2, 6000, False)
        #
        # pt1, index1, pt2, index2 = match_orb_features(kp1, des1, kp2, des2, True)

        # latch
        kp1, des1 = detect_compute_latch(im1, 5000)
        kp2, des2 = detect_compute_latch(im2, 5000)

        pt1, index1, pt2, index2 = match_latch_features(kp1, des1, kp2, des2, True)

    # ut_latch()

    end = time.time()
    print(end - start)
    """
    ut_match_sift_features()
