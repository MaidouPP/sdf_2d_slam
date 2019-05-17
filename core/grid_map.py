import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import os
import yaml


class GridMap(object):

    def __init__(self, config_file):
        if not os.path.exists(config_file):
            raise RuntimeError("File {} not found.".format(config_file))

        with open(config_file, 'r') as fp:
            cfg = yaml.load(fp)
            self._map_name = cfg['name']
            self._width = cfg['width']
            self._height = cfg['height']
            self._res = cfg['resolution']

            self._size_x = int(np.ceil(self._width / self._res))
            self._size_y = int(np.ceil(self._height / self._res))
            self._mini_x = - int((self._width - self._res) / 2)
            self._mini_y = - int((self._height - self._res) / 2)
            self._maxi_x = int(self._width / 2)
            self._maxi_y = int(self._height / 2)


        # Construct sdf map
        self._sdf_map = np.array([self._size_y, self._size_x], np.float32)
        # Construct visit frequency map
        self._freq_map = np.array([self._size_y, self._size_x], np.float32)

        # For test
        self._occupancy_map = np.zeros([self._size_y, self._size_x], np.float32)


    def MapOneScan(self, scan, pose):
        x, y, yaw = pose

        # Construct transform matrix
        rot = np.identity(3, dtype=np.float32)
        rot[0, 0] = np.cos(yaw)
        rot[0, 1] = -np.sin(yaw)
        rot[1, 0] = np.sin(yaw)
        rot[1, 1] = np.cos(yaw)
        # Translation vector
        trans = np.array([[x, y, 0]])

        # Transform points in robot frame to world frame
        scan_w = np.dot(rot, scan) + np.tile(trans.transpose(),
                                             (1, scan.shape[1]))
        scan_w = scan[:2, :]

        # Get the cell coordinates of scan hit points
        scan_w_xs, scan_w_ys = self._FromMeterToCell(scan_w)

        self._occupancy_map[scan_w_ys, scan_w_xs] = 1

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(self._occupancy_map.T, origin='lower')
        plt.show(block=True)


    def VisualizeSdfMap(self):
        pass


    def _FromMeterToCell(self, scan):
        """
        Transform world scan in meter to world scan in cell coordinates.
        """
        xs = scan[0, :]
        ys = scan[1, :]
        # Convert from meters to cells
        cell_xs = ((xs - self._mini_x) / self._res).astype(np.int16)
        cell_ys = ((ys - self._mini_y) / self._res).astype(np.int16)
        print cell_xs, cell_ys
        return cell_xs, cell_ys
