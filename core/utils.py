import numpy as np

from scipy.linalg import expm


def GetSE2FromPose(pose):
    # This is NOT exponential map for se2
    x, y, yaw = pose
    # Construct transform matrix
    rot = np.identity(2, dtype=np.float32)
    rot[0, 0] = float(np.cos(yaw))
    rot[0, 1] = float(-np.sin(yaw))
    rot[1, 0] = float(np.sin(yaw))
    rot[1, 1] = float(np.cos(yaw))
    # Translation vector
    mat = np.identity(3, dtype=np.float32)
    mat[:2, :2] = rot
    mat[0, 2] = x
    mat[1, 2] = y
    return mat

def LogFromSE2(mat):
    assert(mat.shape[0] == 3 and mat.shape[1] == 3)
    # Yaw
    yaw = np.arctan2(mat[1, 0], mat[0, 0])
    # Translation
    t = mat[:2, 2]
    a = float(np.sin(yaw) / yaw)
    b = float((1 - np.cos(yaw)) / yaw)
    tmp = np.array([[a, b], [-b, a]], dtype=np.float32)
    inv_left_jac = 1.0 / (a * a + b * b) * tmp
    u = np.dot(inv_left_jac, t)
    print u
    return np.hstack(u, yaw)

def ExpFromSe2(xi):
    assert(xi.shape[0] == 3)
    yaw = xi[2]
    tmp = np.zeros((3, 3), dtype=np.float32)
    tmp[0, 1] = -yaw
    tmp[1, 0] = yaw
    tmp[:2, 2] = xi[:2].reshape((2,))
    return expm(tmp)

def GetScanWorldCoordsFromSE2(scan, mat):
    """
    input:
      scan - laser point coordinates in meters in robot frame
      pose - 3 by 3 matrix in SE2
    """
    assert(mat.shape[0] == 3 and mat.shape[1] == 3)
    rot = mat[:2, :2]
    trans = mat[:2, 2].reshape(2, 1)
    # Transform points in robot frame to world frame
    scan_w = np.dot(rot, scan) + np.tile(trans, (1, scan.shape[1]))
    return scan_w

def GetScanWorldCoordsFromPose(scan, pose):
    """
    input:
      scan - laser point coordinates in meters in robot frame
      pose - (x, y, yaw)
    """
    mat = GetSE2FromPose(pose)
    return GetScanWorldCoordsFromSE2(scan, mat)
