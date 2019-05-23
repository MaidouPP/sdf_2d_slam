import yaml
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import utils
plt.ion()


class GridMap(object):
    kEps = 1e-6

    def __init__(self, config_file):
        if not os.path.exists(config_file):
            raise RuntimeError("File {} not found.".format(config_file))

        with open(config_file, 'r') as fp:
            cfg = yaml.load(fp)
            self._map_name = cfg['name']
            # Width, height and resolution in meters
            self._width = cfg['width']
            self._height = cfg['height']
            self._res = cfg['resolution']

        # Always assume the center of the map is at world (0m, 0m)
        # Size of map in cells
        self._size_x = int(np.ceil(self._width / self._res))
        self._size_y = int(np.ceil(self._height / self._res))
        # Grid map corners position in cells
        self._mini_x = - float(self._width) / 2
        self._mini_y = - float(self._height) / 2
        self._maxi_x = float(self._width / 2)
        self._maxi_y = float(self._height / 2)
        # Grid's upper left origin world coordinate in meters (row, col)
        self._grid_ul_coord = np.array(
            [self._mini_x, self._maxi_y], dtype=np.float32)

        # Threshold of front and back truncation (in meters)
        self._truncation = 5 * self._res
        # Construct sdf map
        self._sdf_map = np.full([self._size_y, self._size_x], self._truncation)
        # Construct visit frequency map
        self._freq_map = np.zeros([self._size_y, self._size_x])
        # For test
        self._occupancy_map = np.zeros(
            [self._size_y, self._size_x], np.float32)

    @staticmethod
    def _VisualizeOccupancyGrid(grid):
        """
        grid: binary occupancy map
        """
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(grid)
        plt.colorbar()
        plt.show(block=True)

    @property
    def sdf_map(self):
        return self._sdf_map

    def GetSdfValue(self, r, c):
        return self._sdf_map[r, c]

    def MapOneScan(self, scan, pose):
        """
        input:
          scan - laser point coordinates in meters in robot frame
          pose - (x, y, yaw)
        """
        scan_w = utils.GetScanWorldCoords(scan, pose)

        # Get the cell coordinates of scan hit points
        scan_w_xs, scan_w_ys = self.FromMeterToCell(scan_w)

        self._occupancy_map[scan_w_ys, scan_w_xs] = 1

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(self._occupancy_map)
        plt.show(block=True)

    def FuseSdf(self, scan, pose, min_angle, max_angle, inc_angle, min_range, max_range):
        """
        input:
        - pose: SE2
        """
        # trans = pose[:2, 2].reshape((2, 1))
        N = self._size_x * self._size_y
        ys, xs = np.meshgrid(range(self._size_y),
                             range(self._size_x), indexing='ij')
        # In xy fashion
        grid_coords = np.concatenate(
            (xs.reshape(1, -1), -ys.reshape(1, -1)), axis=0).astype(int)

        # Grid cells coordinates to world coordinates in meters (in xy fashion)
        world_pts = grid_coords.astype(
            float) * self._res + self._grid_ul_coord.reshape(-1, 1) + np.array([self._res/2, self._res/2]).reshape(-1, 1)

        # World coordinates to camera coordinates transform
        T_c_w = np.linalg.inv(pose)

        # (2, N), N: total number of grids, grid point coordinates in robot frame
        grid_local_pts = np.dot(T_c_w[:2, :2], world_pts) + np.tile(
            T_c_w[:2, 2].reshape(2, 1), (1, world_pts.shape[1]))
        # grid_local_dist = np.linalg.norm(
        #     grid_local_pts - np.repeat(trans, grid_local_pts.shape[1], axis=1), axis=0)
        grid_local_dist = np.linalg.norm(grid_local_pts, axis=0)

        grid_local_pts_tan = [(grid_local_pts[1, i] / grid_local_pts[0, i]) if grid_local_pts[0, i]
                              != 0 else (math.pi/2 if grid_local_pts[1, i] > 0 else -math.pi/2)
                              for i in range(grid_local_pts.shape[1])]
        # Angle between scan center and the point
        grid_local_pts_angle = np.arctan(grid_local_pts_tan)

        # Pick the points inside the view frustum
        # This positive grid_local_pts[0] check only works when the scan angle covering less than pi!
        """
            --------------------------------
              \                         /
                  \                /
                       \       /
                           *  (scan optical center)
        """
        # For each local grid coordinates in robot frame, compute the scan hit index in camera
        scan_pts_idxs = ((grid_local_pts_angle - min_angle + inc_angle / 2) /
                         inc_angle).astype(np.int16)
        grid_valid_idxs = np.logical_and(np.logical_and(grid_local_pts[0] > 0,
                                                        np.logical_and(grid_local_dist <= max_range,
                                                                       grid_local_dist >= min_range)),
                                         np.logical_and(grid_local_pts_angle >= min_angle,
                                                        grid_local_pts_angle <= max_angle)
                                         )
        num_scan = np.ceil((max_angle - min_angle) / inc_angle) + 1

        # Calculate depth update for valid voxels
        depth_val = np.zeros(N)
        depth_val[grid_valid_idxs] = scan[scan_pts_idxs[grid_valid_idxs]]
        depth_diff = depth_val - grid_local_dist
        # Truncate
        depth_diff_valid_idxs = np.logical_and(depth_diff > -self._truncation,
                                               depth_diff < self._truncation)
        valid_idxs = np.logical_and(grid_valid_idxs, depth_diff_valid_idxs)
        valid_idxs = np.reshape(valid_idxs, (self._size_y, self._size_x))
        depth_diff = np.reshape(depth_diff, (self._size_y, self._size_x))

        self._UpdateSdfMap(valid_idxs, depth_diff)

    def FuseSdfRayTracing(self, scan, pose, min_angle, max_angle, inc_angle, min_range, max_range):
        pass

    def _UpdateSdfMap(self, idxs, depth_diff):
        new_freq_map = self._freq_map + 1
        sdf_map = np.divide(np.multiply(
            self._sdf_map, self._freq_map) + depth_diff, new_freq_map)
        self._sdf_map[idxs] = sdf_map[idxs]
        self._freq_map[idxs] += 1

    def VisualizeSdfMap(self):
        self._VisualizeOccupancyGrid(self._sdf_map)

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

    def _IsValid(self, r, c):
        # Assume (r, c) is valid coordinate
        if self._sdf_map[r][c] > self._truncation - self.kEps or \
           self._sdf_map[r][c] < -self._truncation + self.kEps or \
           self._freq_map[r][c] < 1:
            return False
        else:
            return True

    def HasValidGradient(self, r, c):
        # Check boundary
        if r < 1 or c < 1 or r >= self._size_y - 1 or c >= self._size_x - 1:
            return False

        if self._IsValid(r-1, c) and self._IsValid(r+1, c) and \
           self._IsValid(r, c-1) and self._IsValid(r, c+1):
            return True
        else:
            return False

    def CalcSdfGradient(self, r, c):
        # User must already check the validity of (r, c)
        g_x = 0.5 * (self._sdf_map[r, c+1] - self._sdf_map[r, c-1]) / self._res
        g_y = 0.5 * (self._sdf_map[r-1, c] - self._sdf_map[r+1, c]) / self._res
        g = np.array([g_x, g_y], dtype=np.float32)
        g = np.reshape(g, [1, 2])
        return g

    def VisualizePoints(self, rows, cols):
        canvas = np.zeros([self._size_y, self._size_x])
        canvas[rows, cols] = 1
        self._VisualizeOccupancyGrid(canvas)
