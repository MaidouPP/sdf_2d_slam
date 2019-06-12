#!/usr/bin/env python

import cv2
import gflags
import logging
import numpy as np
import os
import sys
import yaml


FLAGS = gflags.FLAGS
gflags.DEFINE_string("map_config_path", "../data/maps/robopark_map_config.yaml",
                     "Path to the map config file.")
gflags.DEFINE_string("map_fig_path", "../data/maps/colored_map.png",
                     "Path to the colored semantic map figrue file.")


class SemanticMap(object):
    """
    Groundtruth semantic map from colored map figure.
    """

    def __init__(self, config_file, map_fig_file):
        if not os.path.exists(config_file):
            raise RuntimeError("File {} not found.".format(config_file))
        if not os.path.exists(config_file):
            raise RuntimeError("File {} not found.".format(map_fig_file))

        with open(config_file, 'r') as fp:
            cfg = yaml.load(fp)
            self._map_name = cfg['name']
            # Width, height and resolution in meters
            self._width = cfg['width']
            self._height = cfg['height']
            # Width, height in pixel
            self._size_x = cfg['pixel_width']
            self._size_y = cfg['pixel_height']

        # In meters
        self._res = self._width / self._size_x
        self._mini_x = - float(self._width) / 2
        self._mini_y = - float(self._height) / 2
        self._maxi_x = float(self._width) / 2
        self._maxi_y = float(self._height) / 2

        # Semantic label stored grid
        self._semantic_grid = np.zeros((self._size_y, self._size_x), dtype=np.int8)

        self._ReadSemanticGtMap(map_fig_file)

    def _ReadSemanticGtMap(self, map_fig_file):
        img = cv2.imread(map_fig_file)
        # BGR range for r/g/b colors
        boundaries = [
	    ([0, 0, 100], [50, 56, 255]),  # Red
	    ([0, 100, 0], [30, 255, 30]),  # Green
	    ([100, 0, 0], [255, 88, 50])  # Blue
        ]

        label = 1
        for (lower, upper) in boundaries:
            # Create numpy arrays from the boundaries
            lower = np.array(lower, dtype = "uint8")
            upper = np.array(upper, dtype = "uint8")

            # Find the color mask
            mask = cv2.inRange(img, lower, upper)
            output = cv2.bitwise_and(img, img, mask=mask)
            self._semantic_grid = np.where(mask, label, self._semantic_grid)

            # Test and visualize the masks
            # cv2.imshow("images", np.hstack([img, 50 * self._semantic_grid]))
            # cv2.waitKey(0)

            label = label + 1

    def FromMeterToCell(self, scan):
        """
        Transform **world** scan in meter to world scan in cell coordinates.
        """
        xs = scan[0, :]
        ys = scan[1, :]
        # Convert from meters to cells
        cell_cs = ((xs - self._mini_x) / self._res).astype(np.int16)
        cell_rs = self._size_y - \
            ((ys - self._mini_y) / self._res).astype(np.int16) - 1
        return cell_cs, cell_rs

    def GetLabelsOfOneScan(self, scan):
        cs, rs = self.FromMeterToCell(scan)
        return self._semantic_grid[cs, rs]

def main(argv):
    FLAGS(argv)
    logging.basicConfig(format='%(asctime)s,%(msecs)d [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)
    mp = SemanticMap(FLAGS.map_config_path, FLAGS.map_fig_path)

if __name__ == "__main__":
    main(sys.argv)
