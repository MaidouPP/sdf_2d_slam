"""
Iterative Closest Point (ICP) SLAM example
author: Atsushi Sakai (@Atsushi_twi)
"""

import pickle
import math
import matplotlib.pyplot as plt
import numpy as np

#  ICP parameters
EPS = 0.0001
MAXITER = 100

show_animation = True


def ICP_matching(ppoints, cpoints):
    """
    Iterative Closest Point matching
    - input
    ppoints: 2D points in the previous frame
    cpoints: 2D points in the current frame
    - output
    R: Rotation matrix
    T: Translation vector
    """
    H = None  # homogeneous transformation matrix

    dError = 1000.0
    preError = 1000.0
    count = 0

    while dError >= EPS:
        count += 1

        if show_animation:  # pragma: no cover
            plt.cla()
            plt.plot(ppoints[0, :], ppoints[1, :], ".r")
            plt.plot(cpoints[0, :], cpoints[1, :], ".b")
            plt.plot(0.0, 0.0, "xr")
            plt.axis("equal")
            plt.pause(1.0)

        inds, error = nearest_neighbor_assosiation(ppoints, cpoints)
        Rt, Tt = SVD_motion_estimation(ppoints[:, inds], cpoints)

        # update current points
        cpoints = np.dot(Rt, cpoints) + Tt[:, np.newaxis]

        H = update_homogeneous_matrix(H, Rt, Tt)

        dError = abs(preError - error)
        preError = error
        print("Residual:", error)

        if dError <= EPS:
            print("Converge", error, dError, count)
            break
        elif MAXITER <= count:
            print("Not Converge...", error, dError, count)
            break

    R = np.array(H[0:2, 0:2])
    T = np.array(H[0:2, 2])

    return R, T


def update_homogeneous_matrix(Hin, R, T):

    H = np.zeros((3, 3))

    H[0, 0] = R[0, 0]
    H[1, 0] = R[1, 0]
    H[0, 1] = R[0, 1]
    H[1, 1] = R[1, 1]
    H[2, 2] = 1.0

    H[0, 2] = T[0]
    H[1, 2] = T[1]

    if Hin is None:
        return H
    else:
        return np.dot(Hin, H)


def nearest_neighbor_assosiation(ppoints, cpoints):

    # calc the sum of residual errors
    dcpoints = ppoints - cpoints
    d = np.linalg.norm(dcpoints, axis=0)
    error = sum(d)

    # calc index with nearest neighbor assosiation
    inds = []
    for i in range(cpoints.shape[1]):
        minid = -1
        mind = float("inf")
        for ii in range(ppoints.shape[1]):
            d = np.linalg.norm(ppoints[:, ii] - cpoints[:, i])

            if mind >= d:
                mind = d
                minid = ii

        inds.append(minid)

    return inds, error


def SVD_motion_estimation(ppoints, cpoints):

    pm = np.mean(ppoints, axis=1)
    cm = np.mean(cpoints, axis=1)

    pshift = ppoints - pm[:, np.newaxis]
    cshift = cpoints - cm[:, np.newaxis]

    W = np.dot(cshift, pshift.T)
    u, s, vh = np.linalg.svd(W)

    R = (np.dot(u, vh)).T
    t = pm - np.dot(R, cm)

    return R, t

def ProcessScanToLocalCoords(scan, scan_angles):
    angles = scan_angles
    x = scan * np.cos(angles)
    y = scan * np.sin(angles)
    z = np.ones(x.shape)
    ret = np.stack((x, y))
    return ret

def main():
    print(__file__ + " start!!")

    # simulation parameters
    nPoint = 10
    fieldLength = 50.0
    motion = [0.5, 2.0, np.deg2rad(-10.0)]  # movement [x[m],y[m],yaw[deg]]

    nsim = 3  # number of simulation

    # Read scan and pose data
    with open("../data/robopark_2.pkl") as fp:
        data = pickle.load(fp)
        scans, _, _ = data
        min_angle = scans[0][1]
        res_angle = scans[0][2]
        max_angle = scans[0][3]
        scan_angles = np.arange(min_angle,
                                max_angle + res_angle,
                                res_angle)


    # for _ in range(nsim):

    #     # previous points
    #     px = (np.random.rand(nPoint) - 0.5) * fieldLength
    #     py = (np.random.rand(nPoint) - 0.5) * fieldLength
    #     ppoints = np.vstack((px, py))

    #     # current points
    #     cx = [math.cos(motion[2]) * x - math.sin(motion[2]) * y + motion[0]
    #           for (x, y) in zip(px, py)]
    #     cy = [math.sin(motion[2]) * x + math.cos(motion[2]) * y + motion[1]
    #           for (x, y) in zip(px, py)]
    #     cpoints = np.vstack((cx, cy))

    #     R, T = ICP_matching(ppoints, cpoints)
    #     print("R:", R)
    #     print("T:", T)

    # previous points
    ppoints = ProcessScanToLocalCoords(np.array(scans[30][0]), scan_angles)

    # current points
    cpoints = ProcessScanToLocalCoords(np.array(scans[32][0]), scan_angles)

    R, T = ICP_matching(ppoints, cpoints)
    print("R:", R)
    print("T:", T)

if __name__ == '__main__':
    main()
