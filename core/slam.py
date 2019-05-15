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

        self._grid_map = GridMap(FLAGS.map_config_path)

        # Read data
        with open(FLAGS.data_path) as fp:
            data = pickle.load(fp)
            self._scans, self._true_poses, self._times = data

        self._est_poses = []


    def Init(self):
        pass


    def Run(self):
        pass


def main(argv):
    FLAGS(sys.argv)
    logging.getLogger().setLevel(logging.INFO)
    slam = SLAM(FLAGS.data_path, FLAGS.map_config_path)
    slam.Run()


if __name__ == "__main__":
    main(sys.argv)
