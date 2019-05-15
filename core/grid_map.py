#!/usr/bin/env python

import numpy as np
import os
import yaml


class GridMap(object):

    def __init__(self, config_file):
        if not os.path.exists(config_file):
            raise RuntimeError("File {} not found.".format(config_file))

        with open(config_file, 'r') as fp:
            cfg = yaml.load(fp)
            self._map_name = cfp['name']
            self._width = cfg['width']
            self._height = cfg['height']
            self._res = cfg['resolution']

        # Construct sdf map
        self._sdf_map = np.array([self._height, self._width], np.float32)
        # Construct visit frequency map
        self._freq_map = np.array([self._height, self._width], np.float32)


    def MapOneScan(self, scan):
        pass


    def VisSDFMap(self):
        pass
