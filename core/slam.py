#!/usr/bin/env python

try:
    import cPickle as pickle
except ImportError("No cPickle found. Will import pickle instead."):
    import pickle
import cv2
import gflags
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import utils
import yaml

from grid_map import GridMap
from optimizer import SdfOptimizer
from semantic_map import SemanticMap

FLAGS = gflags.FLAGS
gflags.DEFINE_string("data_path", "../data/robopark_2.pkl",
                     "Path to the data file.")
gflags.DEFINE_string("map_config_path", "../data/maps/robopark_map_config.yaml",
                     "Path to the map config file.")
gflags.DEFINE_string("depth_sensor_path", "../data/sensors/KinectDepth.yaml",
                     "Path to the map config file.")
gflags.DEFINE_string("output_map_path", "./output_sdf.png",
                     "Path to the output sdf map file.")
gflags.DEFINE_string("output_occ_path", "./output_occ.png",
                     "Path to the output occupancy map file.")
gflags.DEFINE_string("semantic_map_path", "../data/maps/colored_map.png",
                     "Path to the colored semantic map figrue file.")


class SLAM(object):
    # Some constants
    kDeltaTime = 1
    kOptMaxIters = 10
    kEpsOfYaw = 1e-3
    kEpsOfTrans = 1e-3
    kHuberThr = 15.0
    kOptStopThr = 0.0015

    def __init__(self, data_path, map_config_path, depth_sensor_path, semantic_map_path):
        if not os.path.exists(data_path):
            raise RuntimeError("File {} not found.".format(data_path))

        # Construct 2D grid map
        self._grid_map = GridMap(map_config_path)
        # Construct 2D semantic map
        self._semantic_map = SemanticMap(map_config_path, semantic_map_path)

        fs = cv2.FileStorage(depth_sensor_path, cv2.FILE_STORAGE_READ)
        T_rc = fs.getNode('Extrinsic').mat()

        # Read scan and pose data
        with open(data_path) as fp:
            data = pickle.load(fp)
            self._scans, self._poses, self._times = data
        self._gt_poses = []

        # Transform ground truth of depth camera
        for p in self._poses:
            # Twc = Twr * Trc
            mat_p = utils.GetSE2FromPose(p)
            self._gt_poses.append(np.dot(mat_p, T_rc))

        # Initialization
        self.Init()

        # Scan angle range
        self._scan_angles = np.arange(self._min_angle,
                                      self._max_angle + self._res_angle,
                                      self._res_angle)
        self._scan_dir_vecs = np.stack(
            (np.cos(self._scan_angles), np.sin(self._scan_angles)))

        # Estimated poses (se2) from SDF tracker
        self._est_poses = []
        # Last tracked pose
        self._last_pose = self._gt_poses[0]

    def Init(self):
        self._GetScanSensorInfo()

    def _GetScanSensorInfo(self):
        if len(self._scans) == 0:
            raise RuntimeError("No scan data is extracted.")
        self._min_angle = self._scans[0][1]
        self._res_angle = self._scans[0][2]
        self._max_angle = self._scans[0][3]
        self._min_range = self._scans[0][4]
        self._max_range = self._scans[0][5]

    def _ProcessScanToLocalCoords(self, scan):
        valid_idxs = np.logical_and((scan > self._min_range),
                                    (scan < self._max_range))
        angles = self._scan_angles
        x = scan * np.cos(angles)
        y = scan * np.sin(angles)
        z = np.ones(x.shape)
        ret = np.stack((x, y))
        return valid_idxs, ret

    def Track(self, valid_idxs, scan):
        # Perturbation xi that we are trying to optimize
        xi = np.array([0, 0, 0], dtype=np.float32)
        it = 0
        # last_pose is a SE2
        last_pose = self._last_pose

        while it < self.kOptMaxIters:
            # World scan coordinates
            scan_w = utils.GetScanWorldCoordsFromSE2(scan, last_pose)
            scan_cs, scan_rs = self._grid_map.FromMeterToCellNoRound(scan_w)
            # Hessian
            H = np.zeros((3, 3), dtype=np.float32)
            g = np.zeros((3, 1), dtype=np.float32)
            err_sum = 0.0

            # Calculate hessian and g term
            opt_num = 0
            invalid_rs = []
            invalid_cs = []
            for i in range(scan_cs.shape[0]):
                if not valid_idxs[i]:
                    continue
                c = scan_cs[i]
                r = scan_rs[i]
                # World x and y
                x_w = scan_w[0, i]
                y_w = scan_w[1, i]
                # Local x and y
                x_l = scan[0, i]
                y_l = scan[1, i]
                if self._grid_map.HasValidGradient(r, c):
                    opt_num += 1
                    # dD / dx
                    J_d_x = self._grid_map.CalcSdfGradient(r, c)
                    # dx / d\xi
                    J_x_xi = np.zeros((2, 3), dtype=np.float32)
                    J_x_xi[0, 0] = J_x_xi[1, 1] = 1
                    J_x_xi[0, 2] = -y_w
                    J_x_xi[1, 2] = x_w
                    # Jacobian J_d_xi of shape (1, 3)
                    J = np.dot(J_d_x, J_x_xi)
                    # Gauss-Newton approximation to Hessian
                    freq = float(self._grid_map.weight_map[int(r), int(c)])
                    wt = 1.0 if freq >= self.kHuberThr else freq / self.kHuberThr
                    sdf_val = self._grid_map.GetSdfValue(r, c)
                    H += np.dot(J.transpose(), J) * wt
                    g += J.transpose() * sdf_val * wt
                    # print self._grid_map.GetSdfValue(r, c)
                    err_sum += sdf_val * sdf_val
                else:
                    invalid_rs.append(int(r))
                    invalid_cs.append(int(c))
            # self._grid_map.VisualizePoints(invalid_rs, invalid_cs)
            logging.info("opt_num: %s", opt_num)
            if opt_num == 0:
                logging.error("opt_num=0!")
                break
            err_metric = err_sum / opt_num
            logging.info("   error term: %s ", err_metric)
            try:
                xi = -np.dot(np.linalg.inv(H), g)
            except np.linalg.LinAlgError as err:
                logging.info("Hessian matrix not invertible.")
                xi = np.zeros((3, 1), dtype=np.float32)

            # Check if xi is too small so that we can stop optimization
            if np.abs(xi[2]) < self.kEpsOfYaw and np.linalg.norm(xi[:2]) < self.kEpsOfTrans or \
               err_metric < self.kOptStopThr:
                break
            last_pose = np.dot(utils.ExpFromSe2(xi), last_pose)
            it += 1
        return last_pose

    def Run(self):
        scan_data = np.array(self._scans[0][0])
        pose_mat = self._last_pose
        scan_valid_idxs, scan_local_xys = self._ProcessScanToLocalCoords(
            scan_data)
        self._grid_map.FuseSdf(
            scan_data, scan_valid_idxs, scan_local_xys, pose_mat, self._min_angle, self._max_angle, self._res_angle,
            self._min_range, self._max_range, self._scan_dir_vecs, use_plane=True, init=True)
        # self._grid_map.VisualizeSdfMap(save_path=FLAGS.output_map_path)
        # self._grid_map.VisualizeOccMap(save_path=FLAGS.output_occ_path)

        t = self.kDeltaTime
        prev_scan_data = scan_data
        while (t < len(self._times) - self.kDeltaTime):
            logging.info("t: %s", t)
            logging.info("Ground truth: %s, %s", self._gt_poses[t][0, 2], self._gt_poses[t][1, 2])
            # Get scan data in local xy coordinate
            scan_data = np.array(self._scans[t][0])
            scan_valid_idxs, scan_local_xys = self._ProcessScanToLocalCoords(scan_data)
            # Track from sdf map and semantic map
            scan_gt_world_xys = utils.GetScanWorldCoordsFromSE2(scan_local_xys, self._gt_poses[t])
            # Get semantic lables of the scan points
            semantic_labels = self._semantic_map.GetLabelsOfOneScan(scan_gt_world_xys)
            # if t % 7 == 0:
            #     self._grid_map.MapOneScanFromSE2WithSemantic(scan_local_xys, self._gt_poses[t], semantic_labels)

            curr_pose = self.Track(scan_valid_idxs, scan_local_xys)
            self._est_poses.append(curr_pose)
            self._last_pose = curr_pose
            # Update the sdf map
            self._grid_map.FuseSdf(
                scan_data, scan_valid_idxs, scan_local_xys, curr_pose, self._min_angle, self._max_angle, self._res_angle,
                self._min_range, self._max_range, self._scan_dir_vecs, use_plane=True)
            logging.info("current pose %s, %s\n", curr_pose[0, 2], curr_pose[1, 2])
            t += self.kDeltaTime
            # exit()
        self.VisualizeOdomAndGt(display=False)
        self._grid_map.VisualizeSdfMap(save_path=FLAGS.output_map_path)
        self._grid_map.VisualizeOccMap(save_path=FLAGS.output_occ_path)

    def VisualizeOdomAndGt(self, display=True):
        xs = []
        ys = []
        gt_xs = []
        gt_ys = []
        for pose in self._est_poses:
            xs.append(pose[0, 2])
            ys.append(pose[1, 2])
        for gt_pose in self._gt_poses:
            gt_xs.append(gt_pose[0, 2])
            gt_ys.append(gt_pose[1, 2])
        plt.plot(xs, ys, '-go')
        plt.plot(gt_xs, gt_ys, '-r+')
        plt.legend()
        if display:
            plt.show(block=True)
        else:
            plt.savefig("odom.png")


def main(argv):
    FLAGS(argv)
    logging.basicConfig(format='%(asctime)s,%(msecs)d [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)
    slam = SLAM(FLAGS.data_path, FLAGS.map_config_path, FLAGS.depth_sensor_path, FLAGS.semantic_map_path)
    slam.Run()


if __name__ == "__main__":
    main(sys.argv)
