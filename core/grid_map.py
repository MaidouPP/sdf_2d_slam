import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
plt.ion()


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

            # Always assume the center of the map is at world (0m, 0m)
            self._size_x = int(np.ceil(self._width / self._res))
            self._size_y = int(np.ceil(self._height / self._res))
            # Grid map corners in cells
            self._mini_x = - float(self._width) / 2
            self._mini_y = - float(self._height) / 2
            self._maxi_x = float(self._width / 2)
            self._maxi_y = float(self._height / 2)
            # Grid's upper left origin world coordinate in meters (row, col)
            self._grid_ul_coord = np.array(
                [self._mini_x, self._maxi_y], dtype=np.float32)

        # Construct sdf map
        self._sdf_map = np.array([self._size_y, self._size_x], np.float32)
        # Construct visit frequency map
        self._freq_map = np.array([self._size_y, self._size_x], np.float32)

        # For test
        self._occupancy_map = np.zeros(
            [self._size_y, self._size_x], np.float32)

    def MapOneScan(self, scan, pose):
        scan_w = self.GetScanWorldCoords(scan, pose)

        # Get the cell coordinates of scan hit points
        scan_w_xs, scan_w_ys = self._FromMeterToCell(scan_w)

        self._occupancy_map[scan_w_ys, scan_w_xs] = 1

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(self._occupancy_map, origin='lower')
        plt.show(block=True)

    def FuseSdf(self, scan, pose, min_angle, max_angle):
        x, y, yaw = pose
        # scan_w = self.GetScanWorldCoords(scan, pose)

        # Get voxel grid coordinates
        ys, xs = np.meshgrid(range(self._size_y),
                             range(self._size_x), indexing='ij')
        # Get grid coords from a row-col scheme
        grid_coords = np.concatenate(
            (xs.reshape(1, -1), -ys.reshape(1, -1)), axis=0).astype(int)
        print grid_coords

        # Grid cells coordinates to world coordinates in meters
        world_pts = grid_coords.astype(
            float) * self._res + self._grid_ul_coord.reshape(-1, 1)
        print world_pts

        # World coordinates to camera coordinates
        T_w_c = self._GetSE2FromPose(pose)
        T_c_w = np.linalg.inv(T_w_c)
        cam_pts = np.dot(T_c_w[:2, :2], world_pts) + np.tile(
            T_c_w[:2, 2].reshape(2, 1), (1, world_pts.shape[1]))
        print cam_pts

        # Skip if outside view frustum
        depth_val = np.zeros(cam_pts.shape[1])
        """
            --------------------------------
              \                         /
                  \                /
                       \       /
                           *  (camera optical center)
        """
        valid_coords = np.logical_and(pix_x >= 0,
                                      np.logical_and(pix_x < im_w,
                                                     np.logical_and(pix_y >= 0,
                                                                    pix_y < im_h)))


    def VisualizeSdfMap(self):
        pass

    def _GetSE2FromPose(self, pose):
        x, y, yaw = pose
        # Construct transform matrix
        rot = np.identity(2, dtype=np.float32)
        rot[0, 0] = np.cos(yaw)
        rot[0, 1] = -np.sin(yaw)
        rot[1, 0] = np.sin(yaw)
        rot[1, 1] = np.cos(yaw)
        # Translation vector
        mat = np.identity(3, dtype=np.float32)
        mat[:2, :2] = rot
        mat[2, 0] = x
        mat[2, 1] = y
        return mat

    def GetScanWorldCoords(self, scan, pose):
        """
        input:
          scan - laser point coordinates in meters in robot frame
          pose - (x, y, yaw)
        """
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
        return scan_w

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
