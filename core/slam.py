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
import yaml

from grid_map import GridMap

FLAGS = gflags.FLAGS
gflags.DEFINE_string("data_path", "../data/robopark.pkl",
                     "Path to the data file.")
gflags.DEFINE_string("map_config_path", "../data/maps/robopark_map_config.yaml",
                     "Path to the map config file.")


class SLAM(object):

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

        # Estimated poses from SDF tracker
        self._est_poses = []

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
        print scan.shape
        valid_idxs = np.logical_and((scan > self._min_range),
                                    (scan < self._max_range))
        data = scan[valid_idxs]
        angles = self._scan_angles[valid_idxs]
        x = data * np.cos(angles)
        y = data * np.sin(angles)
        z = np.ones(x.shape)
        ret = np.stack((x, y, z))
        return ret

    def Run(self):
        scan_data = np.array(self._scans[0][0])
        scan_data = self._ProcessScanToLocalCoords(scan_data)
        # self._grid_map.MapOneScan(scan_data, self._poses[0])
        self._grid_map.FuseSdf(scan_data, self._poses[0], self._min_angle, self._max_angle)


def main(argv):
    FLAGS(sys.argv)
    logging.getLogger().setLevel(logging.INFO)
    slam = SLAM(FLAGS.data_path, FLAGS.map_config_path)
    slam.Run()


if __name__ == "__main__":
    main(sys.argv)
