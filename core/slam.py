#!/usr/bin/env python

try:
    import cPickle as pickle
except ImportError("No cPickle found. Will import pickle instead."):
    import pickle
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

FLAGS = gflags.FLAGS
gflags.DEFINE_string("data_path", "../data/robopark.pkl",
                     "Path to the data file.")
gflags.DEFINE_string("map_config_path", "../data/maps/robopark_map_config.yaml",
                     "Path to the map config file.")


class SLAM(object):
    # Some constants
    kDeltaTime = 3
    kOptMaxIters = 20
    kEpsOfYaw = 1e-3
    kEpsOfTrans = 1e-3

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
        self._last_pose = np.identity(3, dtype=np.float32)

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
        # Perturbation xi that we are trying to optimize
        xi = np.array([0, 0, 0], dtype=np.float32)
        it = 0
        # last_pose is a SE2
        last_pose = self._last_pose

        while it < self.kOptMaxIters:
            scan_w = utils.GetScanWorldCoordsFromSE2(scan, last_pose)
            scan_cs, scan_rs = self._grid_map.FromMeterToCellNoRound(scan_w)
            # Hessian
            H = np.zeros((3, 3), dtype=np.float32)
            g = np.zeros((3, 1), dtype=np.float32)
            err_sum = 0.0

            # Calculate hessian and g term
            for i in range(scan_cs.shape[0]):
                c = scan_cs[i]
                r = scan_rs[i]
                x = scan_w[0, i]
                y = scan_w[1, i]
                if self._grid_map.HasValidGradient(r, c):
                    # dD / dx
                    J_d_x = self._grid_map.CalcSdfGradient(r, c)
                    # dx / d\xi
                    J_x_xi = np.zeros((2, 3), dtype=np.float32)
                    J_x_xi[0, 0] = J_x_xi[1, 1] = 1
                    J_x_xi[0, 2] = -y
                    J_x_xi[1, 2] = x
                    # Jacobian J_d_xi of shape (1, 3)
                    J = np.dot(J_d_x, J_x_xi)
                    # Gauss-Newton approximation to Hessian
                    H += np.dot(J.transpose(), J)
                    g += J.transpose() * self._grid_map.GetSdfValue(r, c)
                    # print self._grid_map.GetSdfValue(r, c)
                    err_sum += self._grid_map.GetSdfValue(r, c) * self._grid_map.GetSdfValue(r, c)
            try:
                xi = -np.dot(np.linalg.inv(H), g)
            except np.linalg.LinAlgError as err:
                print "Hessian matrix not invertible."
                xi = np.zeros((3, 1), dtype=np.float32)

            # Check if xi is too small so that we can stop optimization
            if np.abs(xi[2]) < self.kEpsOfYaw and np.linalg.norm(xi[:2]) < self.kEpsOfTrans:
                break
            last_pose = np.dot(utils.ExpFromSe2(xi), last_pose)
            it += 1
            print "   error term: ", err_sum

        return last_pose

    def Run(self):
        scan_data = np.array(self._scans[0][0])
        pose_mat = utils.GetSE2FromPose(self._poses[0])
        self._grid_map.FuseSdf(
            scan_data, pose_mat, self._min_angle, self._max_angle, self._res_angle,
            self._min_range, self._max_range)
        self._grid_map.VisualizeSdfMap()

        t = self.kDeltaTime
        while (t < len(self._times) - self.kDeltaTime):
            print "t: ", t
            scan_data = np.array(self._scans[t][0])
            scan_local_xys = self._ProcessScanToLocalCoords(scan_data)
            curr_pose = self.Track(scan_local_xys)
            self._est_poses.append(curr_pose)
            self._last_pose = curr_pose
            t += self.kDeltaTime
            self._grid_map.FuseSdf(
                scan_data, curr_pose, self._min_angle, self._max_angle, self._res_angle,
                self._min_range, self._max_range)
            print curr_pose
            print "Ground truth: ", self._poses[t]
            # self._grid_map.VisualizeSdfMap()
            # exit()
        self.VisualizeOdomAndGt()
        self._grid_map.VisualizeSdfMap()

    def VisualizeOdomAndGt(self):
        xs = []
        ys = []
        gt_xs = []
        gt_ys = []
        for pose in self._est_poses:
            xs.append(pose[0, 2])
            ys.append(pose[1, 2])
        for gt_pose in self._poses:
            gt_xs.append(gt_pose[0])
            gt_ys.append(gt_pose[1])
        plt.plot(xs, ys, c='g')
        plt.plot(gt_xs, gt_ys, c='r')
        plt.legend()
        plt.show(block=True)

def main(argv):
    FLAGS(sys.argv)
    logging.getLogger().setLevel(logging.INFO)
    slam = SLAM(FLAGS.data_path, FLAGS.map_config_path)
    slam.Run()


if __name__ == "__main__":
    main(sys.argv)
