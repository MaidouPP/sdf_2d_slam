"""Convert a ROS bag of odometry poses and laser scans into a pickle file.

This allows the data to be used from pure Python scripts, without needing ROS.

Author: Matthew Edwards
Date: August 2016
"""

from __future__ import print_function, division

try:
    import cPickle as pickle
except ImportError:
    import pickle
import rosbag
import tf


def _load_bag(bag_filename):
    """Load the laser scans and corresponding odom poses from a bag."""
    bag = rosbag.Bag(bag_filename)
    # Sort by the time the messages were generated rather than the time they
    # made it into the bag, since they sometimes get slightly mixed up
    messages = list(bag.read_messages(['/base_pose_ground_truth', '/scan']))
    messages.sort(key=lambda m: m.message.header.stamp.secs +
                  m.message.header.stamp.nsecs/1.0e9)
    # Extract the most recent odometry message for each scan
    scans = []
    last_true_pose = None
    true_poses = []
    times = []
    for topic, message, _ in messages:
        if topic == '/base_pose_ground_truth':
            last_true_pose = message
        if topic == '/scan':
            if last_true_pose is None:
                continue
            scans.append(message)
            true_poses.append(last_true_pose.pose)
            times.append(message.header.stamp.to_time())
    return scans, true_poses, times


def _convert_scans(scans):
    """Convert ROS scan messages into (ranges, angle_min, angle_increment) tuples."""
    return [(scan.ranges, scan.angle_min, scan.angle_increment, scan.angle_max,
             scan.range_min, scan.range_max) for scan in scans]


def _convert_pose(pose):
    x = pose.position.x
    y = pose.position.y
    quat = pose.orientation
    quat = [quat.x, quat.y, quat.z, quat.w]
    _, _, rotation = tf.transformations.euler_from_quaternion(quat)
    return x, y, rotation


def _convert_poses(poses):
    return [_convert_pose(pose.pose) for pose in poses]


def _save(scans, true_poses, times, pickle_filename):
    """Save a set of scans and poses to a pickle file."""
    with open(pickle_filename, 'wb') as f:
        pickle.dump((scans, true_poses, times), f, pickle.HIGHEST_PROTOCOL)


def bag_to_pickle(in_filename, out_filename, skip=0):
    scans, true_poses, times = _load_bag(in_filename)
    scans = _convert_scans(scans)[skip:]
    true_poses = _convert_poses(true_poses)[skip:]
    times = times[skip:]
    _save(scans, true_poses, times, out_filename)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print('Usage: %s in_filename [out_filename]' % sys.argv[0])
        sys.exit()
    in_filename = sys.argv[1]
    if len(sys.argv) > 2:
        out_filename = sys.argv[2]
    else:
        out_filename = in_filename.replace('.bag', '.pickle')

    bag_to_pickle(in_filename, out_filename)
    print('Converted', in_filename, 'to', out_filename)
