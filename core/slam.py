#!/usr/bin/env python

try:
    import cPickle as pickle
except ImportError("No cPickle found. Will import pickle instead."):
    import pickle
import gflags
import logging
import numpy as np
import os
import sys
import utils
import yaml

from grid_map import GridMap
from optimizer import SdfOptimizer

FLAGS = gflags.FLAGS
gflags.DEFINE_string("data_path", "../data/robopark.pkl",
                     "Path to the data file.")
gflags.DEFINE_string("map_config_path", "../data/maps/robopark_map_config.yaml",
                     "Path to the map config file.")


class SLAM(object):
    kDeltaTime = 3
    kOptMaxIters = 20

    def __init__(self, data_path, map_config_path):
        if not os.path.exists(data_path):
            raise RuntimeError("File {} not found.".format(data_path))

        # Construct 2D grid map
        self._grid_map = GridMap(FLAGS.map_config_path)

        # Read scan and pose data
        with open(FLAGS.data_path) as fp:
            data = pickle.load(fp)
            self._scans, self._poses, self._times = data

        # Initialization
        self.Init()

        # Scan angle range
        self._scan_angles = np.arange(self._min_angle,
                                      self._max_angle + self._res_angle,
                                      self._res_angle)

        # Estimated poses (se2) from SDF tracker
        self._est_poses = []
        # Last tracked pose
        self._last_pose = [0, 0, 0]

        # Construct an optimizer
        self._optimizer = SdfOptimizer()

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
        data = scan[valid_idxs]
        angles = self._scan_angles[valid_idxs]
        x = data * np.cos(angles)
        y = data * np.sin(angles)
        z = np.ones(x.shape)
        ret = np.stack((x, y))
        return ret

    def Track(self, scan):
        # scan_local_xy = self._ProcessScanToLocalCoords(scan_data)
        # last_pose_mat = utils.GetSE2FromPose(self._last_pose)
        # Perturbation xi that we are trying to optimize
        xi = np.array([0, 0, 0], dtype=np.float32)
        it = 0
        # last_pose is a SE2
        last_pose = utils.GetSE2FromPose(self._last_pose)

        while it < self.kOptMaxIters:
            init_scan_w = utils.GetScanWorldCoordsFromSE2(scan, last_pose)
            init_scan_cs, init_scan_rs = self._grid_map.FromMeterToCell(init_scan_w)
            # Hessian
            H = np.zeros((3, 3), dtype=np.float32)
            g = np.zeros((3, 1), dtype=np.float32)

            # Calculate hessian and g term
            for i in range(init_scan_cs.shape[0]):
                c = init_scan_cs[i]
                r = init_scan_rs[i]
                x = init_scan_w[0, i]
                y = init_scan_w[1, i]
                if self._grid_map.HasValidGradient(r, c):
                    # dD / dx
                    J_d_x = self._grid_map.CalcSdfGradient(r, c)
                    # dx / d\xi
                    J_x_xi = np.zeros((2, 3), dtype=np.float32)
                    J_x_xi[0, 0] = J_x_xi[1, 1] = 1
                    J_x_xi[0, 2] = -y
                    J_x_xi[1, 2] = x
                    # J: (1, 3)
                    J = np.dot(J_d_x, J_x_xi)
                    # Gauss-Newton approximation to Hessian
                    H += np.dot(J.transpose(), J)
                    g += J.transpose() * self._grid_map.GetSdfValue(r, c)

            xi = -np.dot(np.linalg.inv(H), g)
            last_pose = np.dot(utils.ExpFromSe2(xi), last_pose)
            it += 1

        return last_pose

    def Run(self):
        scan_data = np.array(self._scans[0][0])
        # scan_local_xy = self._ProcessScanToLocalCoords(scan_data)
        # self._grid_map.MapOneScan(scan_local_xy, self._poses[0])
        self._grid_map.FuseSdf(
            scan_data, self._poses[0], self._min_angle, self._max_angle, self._res_angle,
            self._min_range, self._max_range)
        self._est_poses.append([0, 0, 0])
        self._grid_map.VisualizeSdfMap()
        t = self.kDeltaTime
        while (t < len(self._times)):
            # we want is (x, y, yaw), which is not se2
            # yet this pose is a SE2
            scan_data = np.array(self._scans[0][0])
            scan_local_xys = self._ProcessScanToLocalCoords(scan_data)
            pose = self.Track(scan_local_xys)
            # self._est_poses.append(pose)
            t += self.kDeltaTime


def main(argv):
    FLAGS(sys.argv)
    logging.getLogger().setLevel(logging.INFO)
    slam = SLAM(FLAGS.data_path, FLAGS.map_config_path)
    slam.Run()


if __name__ == "__main__":
    main(sys.argv)
