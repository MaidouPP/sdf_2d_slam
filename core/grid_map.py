from sklearn.decomposition import PCA
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import utils
import yaml
plt.ion()


class GridMap(object):
    kEps = 1e-6  # Truncation numerical error
    kNormalWindow = 4  # The left/right neighboring of the beam hit point
    # Range of distance of neighboring points (in meters)
    kNormalDistThr = 0.1
    kTruncationThr = 9
    kIsoMapThr = 0.024

    def __init__(self, config_file, num_semantic_classes=4):
        if not os.path.exists(config_file):
            raise RuntimeError("File {} not found.".format(config_file))

        with open(config_file, 'r') as fp:
            cfg = yaml.load(fp)
            self._map_name = cfg['name']
            # Width, height and resolution in meters
            self._width = cfg['width'] + 1.0
            self._height = cfg['height'] + 1.0
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
        self._truncation = self.kTruncationThr * self._res
        # Construct sdf map
        self._sdf_map = np.full([self._size_y, self._size_x], self._truncation)
        # Construct visit frequency map
        self._freq_map = np.zeros([self._size_y, self._size_x])
        # Only for test
        self._occupancy_map = np.zeros(
            [self._size_y, self._size_x], np.float32)

        # Semantics
        self._num_semantic_classes = num_semantic_classes
        self._sdf_map_semantic = np.full([self._size_y, self._size_x, num_semantic_classes],
                                         self._truncation)
        self._freq_map_semantic = np.full(
            [self._size_y, self._size_x, num_semantic_classes], 0)

    @staticmethod
    def _VisualizeOccupancyGrid(grid, save_path=None):
        """
        grid: binary occupancy map
        """
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(grid)
        if save_path is not None:
            plt.savefig(save_path)

    @property
    def sdf_map(self):
        return self._sdf_map

    @property
    def freq_map(self):
        return self._freq_map

    def GetSdfValue(self, r, c):
        return self.InterpolateSdfValue(r, c)

    def GetSdfValueSemantic(self, r, c, cls):
        return self.InterpolateSdfValueSemantic(r, c, cls)

    def GetFreqSemantic(self, r, c, cls):
        return self._freq_map_semantic[r, c, cls]

    def MapOneScan(self, scan, pose):
        """
        input:
          scan - laser point coordinates in meters in robot frame
          pose - (x, y, yaw)
        """
        scan_w = utils.GetScanWorldCoordsFromPose(scan, pose)

        # Get the cell coordinates of scan hit points
        scan_w_xs, scan_w_ys = self.FromMeterToCell(scan_w)

        self._occupancy_map[scan_w_ys, scan_w_xs] = 1

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(self._occupancy_map)
        plt.show(block=True)

    def MapOneScanFromSE2(self, scan, pose):
        """
        input:
          scan - laser point coordinates in meters in robot frame
          pose - SE2
        """
        occupancy_map = np.zeros([self._size_y, self._size_x], np.float32)
        scan_w = utils.GetScanWorldCoordsFromSE2(scan, pose)

        # Get the cell coordinates of scan hit points
        scan_w_xs, scan_w_ys = self.FromMeterToCell(scan_w)

        occupancy_map[scan_w_ys, scan_w_xs] = 1

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(occupancy_map)
        plt.show(block=True)

    def MapOneScanFromSE2WithSemantic(self, scan, pose, semantic_labels):
        """
        input:
          scan - laser point coordinates in meters in robot frame
          pose - SE2
          semantic_label - semantic labels of the scan points
        """
        occupancy_map = np.zeros([self._size_y, self._size_x], np.float32)
        scan_w = utils.GetScanWorldCoordsFromSE2(scan, pose)

        # Get the cell coordinates of scan hit points
        scan_w_xs, scan_w_ys = self.FromMeterToCell(scan_w)

        occupancy_map[scan_w_ys, scan_w_xs] = semantic_labels

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(occupancy_map)
        plt.show(block=True)

    def InterpolateSdfValue(self, r, c):
        # r and c are float numbers, indicating the index of the cell
        w_sum = 0.0
        sdf_sum = 0.0
        for r_offset in [-1.0, 0.0, 1.0]:
            for c_offset in [-1.0, 0.0, 1.0]:
                r_curr = int(int(r) + r_offset)
                c_curr = int(int(c) + c_offset)
                # Calculate the coordinate distance (in cells)
                r_dist = float(np.fabs(r_curr + 0.5 - r))
                c_dist = float(np.fabs(c_curr + 0.5 - c))
                if r_dist > 1.0 or c_dist > 1.0:
                    continue
                volume = r_dist * c_dist
                if self._freq_map[r_curr, c_curr] > 0:
                    if volume < 1e-7:
                        return self._sdf_map[r_curr, c_curr]
                    w = 1.0 / volume
                    w_sum += w
                    sdf_sum += w * self._sdf_map[r_curr, c_curr]
        return sdf_sum / w_sum

    def InterpolateSdfValueSemantic(self, r, c, cls):
        # r and c are float numbers, indicating the index of the cell
        w_sum = 0.0
        sdf_sum = 0.0
        for r_offset in [-1.0, 0.0, 1.0]:
            for c_offset in [-1.0, 0.0, 1.0]:
                r_curr = int(int(r) + r_offset)
                c_curr = int(int(c) + c_offset)
                # Calculate the coordinate distance (in cells)
                r_dist = float(np.fabs(r_curr + 0.5 - r))
                c_dist = float(np.fabs(c_curr + 0.5 - c))
                if r_dist > 1.0 or c_dist > 1.0:
                    continue
                volume = r_dist * c_dist
                if self._freq_map_semantic[r_curr, c_curr, cls] > 0:
                    if volume < 1e-7:
                        return self._sdf_map_semantic[r_curr, c_curr, cls]
                    w = 1.0 / volume
                    w_sum += w
                    sdf_sum += w * self._sdf_map_semantic[r_curr, c_curr, cls]
        return sdf_sum / w_sum

    def CalcNormalVecOfAScan(self, scan_valid_idxs, scan_local_xys, scan_dir_vecs):
        assert(scan_local_xys.shape[0] == 2 and scan_dir_vecs.shape[0] == 2)
        total_num = scan_local_xys.shape[1]
        normals = np.zeros((total_num, 2), dtype=np.float32)
        pca = PCA(n_components=2)
        for i in range(total_num):
            if not scan_valid_idxs[i]:
                continue
            coord_i = scan_local_xys[:, i].reshape((1, -1))
            pts = coord_i
            for j in range(i-self.kNormalWindow, i+self.kNormalWindow+1):
                if j < 0 or j >= total_num or j == i:
                    continue
                coord_j = scan_local_xys[:, j].reshape((1, -1))
                if np.linalg.norm(coord_i - coord_j) < self.kNormalDistThr:
                    pts = np.concatenate((pts, coord_j), axis=0)

            # If no neigboring points
            if pts.shape[1] <= 2:
                normals[i] = scan_dir_vecs[:, i]
            else:
                pca.fit(pts)
                normals[i] = pca.components_[1]
        return normals

    def FuseSdf(self, scan, scan_valid_idxs, scan_local_xys, pose, min_angle, max_angle, inc_angle,
                min_range, max_range, scan_dir_vecs, semantic_labels=None, use_plane=False, use_semantics=True):
        """
        input:
        - scan: beam depth vector
        - pose: SE2
        """
        N = self._size_x * self._size_y
        ys, xs = np.meshgrid(range(self._size_y),
                             range(self._size_x), indexing='ij')
        # In xy fashion
        grid_coords = np.concatenate(
            (xs.reshape(1, -1), -ys.reshape(1, -1)), axis=0).astype(int)

        # Grid cells coordinates to world coordinates in meters (in xy fashion)
        world_pts = grid_coords.astype(
            float) * self._res + self._grid_ul_coord.reshape(-1, 1) + \
            np.array([self._res/2, -self._res/2]).reshape(-1, 1)

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
        """
            -------------------------------
              \                         /
                  \                /
                       \       /
                           *  (scan optical center)
        """
        # For each local grid coordinates in robot frame, compute the scan hit index in camera
        scan_pts_idxs = ((grid_local_pts_angle - min_angle) /
                         inc_angle + 0.5).astype(np.int16)
        num_scan = np.ceil((max_angle - min_angle) / inc_angle) + 1
        # Get valid grid cell indexes
        # This positive grid_local_pts[0] check only works when the scan angle covering less than pi!
        grid_valid_idxs = np.logical_and(np.logical_and(grid_local_pts[0] > 0,
                                                        np.logical_and(grid_local_dist <= max_range,
                                                                       grid_local_dist >= min_range)),
                                         np.logical_and(
                                             np.logical_and(
                                                 scan_pts_idxs > 0, scan_pts_idxs < num_scan),
                                             np.logical_and(grid_local_pts_angle >= min_angle,
                                                            grid_local_pts_angle <= max_angle)))
        # Prepare for validating the indexing (getting rid of too large or too small indexes)
        scan_pts_idxs[scan_pts_idxs < 0] = 0
        scan_pts_idxs[scan_pts_idxs >= num_scan] = 0

        # Calculate normal vectors of the current scan
        if use_plane:
            normals = self.CalcNormalVecOfAScan(
                scan_valid_idxs, scan_local_xys, scan_dir_vecs)

        # Calculate depth update for valid voxels
        depth_val = np.zeros(N)
        depth_val[grid_valid_idxs] = scan[scan_pts_idxs[grid_valid_idxs]]

        # Calculate point-to-plane or point-to-point distance
        if use_plane:
            p2p_dist = -grid_local_pts + scan_local_xys[:, scan_pts_idxs]
            # Normal direction may be incorrect, we can check x-axis value to see if positive or negative diff
            p2p_dist_sign = np.sign(p2p_dist[0, :])
            # p2p_dist = depth_val - grid_local_dist
            # tmp = scan_dir_vecs[:, scan_pts_idxs] * p2p_dist
            depth_diff = np.multiply(p2p_dist_sign, np.fabs(
                np.dot(normals[scan_pts_idxs], p2p_dist).diagonal()))
        else:
            depth_diff = depth_val - grid_local_dist

        # Truncate
        depth_diff_valid_idxs = np.logical_and(depth_diff > -self._truncation/2 + self.kEps,
                                               depth_diff < self._truncation - self.kEps)
        valid_idxs = np.logical_and(grid_valid_idxs, depth_diff_valid_idxs)
        valid_idxs = np.logical_and(valid_idxs, scan_valid_idxs[scan_pts_idxs])

        valid_idxs = np.reshape(valid_idxs, (self._size_y, self._size_x))
        depth_diff = np.reshape(depth_diff, (self._size_y, self._size_x))

        self._UpdateSdfMap(valid_idxs, depth_diff)

        if use_semantics:
            if semantic_labels is None:
                raise RuntimeError("No semantic label provided.")
            self._UpdateSdfMapWithSemantics(
                valid_idxs, depth_diff, scan_pts_idxs, semantic_labels)

    def _UpdateSdfMap(self, idxs, depth_diff):
        new_freq_map = self._freq_map + 1
        sdf_map = np.divide(np.multiply(
            self._sdf_map, self._freq_map) + depth_diff, new_freq_map)
        self._sdf_map[idxs] = sdf_map[idxs]
        self._freq_map[idxs] += 1

    def _UpdateSdfMapWithSemantics(self, idxs, depth_diff, scan_pts_idxs, semantic_labels):
        voxel_labels = semantic_labels[scan_pts_idxs].reshape((self._size_y, self._size_x))
        # rs, cs = np.ogrid[:self._size_y, :self._size_x]
        # new_freq_map = self._freq_map_semantic[rs, cs, voxel_labels] + 1
        # new_sdf_map = np.divide(np.multiply(
        #     self._sdf_map_semantic[rs, cs, voxel_labels], self._freq_map_semantic[rs, cs, voxel_labels]) + depth_diff, new_freq_map)

        # TODO (shixin): Stupid for-loop, need to substitute with better numpy operation
        for r in range(self._size_y):
            for c in range(self._size_x):
                if idxs[r, c]:
                    self._sdf_map_semantic[r, c, voxel_labels[r, c]] = float(self._freq_map_semantic[r, c, voxel_labels[r, c]] * \
                                                                       self._sdf_map_semantic[r, c, voxel_labels[r, c]] + depth_diff[r, c]) / \
                                                                       (self._freq_map_semantic[r, c, voxel_labels[r, c]] + 1)
                    self._freq_map_semantic[r, c, voxel_labels[r, c]] = self._freq_map_semantic[r, c, voxel_labels[r, c]] + 1

    def VisualizeSdfMap(self, save_path=None):
        self._VisualizeOccupancyGrid(self._sdf_map, save_path=save_path)

    def VisualizeOccMap(self, save_path=None):
        occ_map = np.zeros((self._size_y, self._size_x), dtype=np.float32)
        for i in range(self._size_y):
            for j in range(self._size_x):
                if math.fabs(self._sdf_map[i, j]) < self.kIsoMapThr:
                    occ_map[i, j] = 1.0 * np.argmax(self._freq_map_semantic[i, j]) * 50
        self._VisualizeOccupancyGrid(occ_map, save_path=save_path)

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

    def FromMeterToCellNoRound(self, scan):
        """
        Transform **world** scan in meter to world scan in cell coordinates
        (no rounding to integer).
        """
        xs = scan[0, :]
        ys = scan[1, :]
        # Convert from meters to cells
        cell_cs = ((xs - self._mini_x) / self._res)
        cell_rs = self._size_y - ((ys - self._mini_y) / self._res)
        return cell_cs, cell_rs

    def _IsValid(self, r, c):
        # Assume (r, c) is valid coordinate
        if self._freq_map[r][c] < 1:
            return False
        else:
            return True

    def _IsValidSemantic(self, r, c, cls):
        # Assume (r, c) is valid coordinate
        if self._freq_map_semantic[r, c, cls] < 1:
            return False
        else:
            return True

    def HasValidGradient(self, r, c):
        # Check boundary
        if r-1 < 0 or c-1 < 0 or r+1 > self._size_y - 1 or c+1 > self._size_x - 1:
            return False

        if self._IsValid(int(r-1), int(c)) and self._IsValid(int(r+1), int(c)) and \
           self._IsValid(int(r), int(c-1)) and self._IsValid(int(r), int(c+1)):
            return True
        else:
            return False

    def HasValidGradientSemantic(self, r, c, cls):
        # Check boundary
        if r-1 < 0 or c-1 < 0 or r+1 > self._size_y - 1 or c+1 > self._size_x - 1:
            return False

        if self._IsValidSemantic(int(r-1), int(c), cls) and self._IsValidSemantic(int(r+1), int(c), cls) and \
           self._IsValidSemantic(int(r), int(c-1), cls) and self._IsValidSemantic(int(r), int(c+1), cls):
            return True
        else:
            return False

    def CalcSdfGradient(self, r, c):
        # User must already check the validity of (r, c)
        g_x = 0.5 * (self.InterpolateSdfValue(r, c+1.0) -
                     self.InterpolateSdfValue(r, c-1.0)) / self._res
        g_y = 0.5 * (self.InterpolateSdfValue(r-1.0, c) -
                     self.InterpolateSdfValue(r+1.0, c)) / self._res
        g = np.array([g_x, g_y], dtype=np.float32)
        g = np.reshape(g, [1, 2])
        return g

    def CalcSdfGradientSemantic(self, r, c, cls):
        # User must already check the validity of (r, c)
        g_x = 0.5 * (self.InterpolateSdfValueSemantic(r, c+1.0, cls) -
                     self.InterpolateSdfValueSemantic(r, c-1.0, cls)) / self._res
        g_y = 0.5 * (self.InterpolateSdfValueSemantic(r-1.0, c, cls) -
                     self.InterpolateSdfValueSemantic(r+1.0, c, cls)) / self._res
        g = np.array([g_x, g_y], dtype=np.float32)
        g = np.reshape(g, [1, 2])
        return g

    def VisualizePoints(self, rows, cols):
        canvas = np.zeros([self._size_y, self._size_x])
        canvas[rows, cols] = 1
        self._VisualizeOccupancyGrid(canvas)

    def VisualizeFreqMap(self):
        self._VisualizeOccupancyGrid(self._freq_map)

    def VisualizeSemanticMap(self):
        valid_idxs = self._freq_map > 0
        valid_idxs = valid_idxs.astype(dtype=np.int8)
        semantics = np.argmax(self._freq_map_semantic, axis=2)
        grid = np.multiply(valid_idxs, semantics)
        self._VisualizeOccupancyGrid(grid * 50)
