#!/usr/bin/env python
import math
import rospy
from sensor_msgs.msg import LaserScan
import tf
from sklearn.cluster import DBSCAN
import numpy as np


class DistanceCalculator:
    def __init__(self):
        rospy.init_node('distance_calculator', anonymous=True)

        self.lidar_sub_filtered = rospy.Subscriber(
            # '/lds1/scan_raw', LaserScan, self.lidar_callback, queue_size=10)
            '/scan_filtered', LaserScan, self.lidar_callback, queue_size=10)

        self.tf_broadcaster = tf.TransformBroadcaster()
        self.last_object_time = None
        self.transforms = []  

        self.timer = rospy.Timer(rospy.Duration(0.1), self.remove_transform)

        # Define intensity thresholds
        self.intensity_min = 160
        self.intensity_max = 195
        self.proximity_threshold = 0.3  # Adjust DBSCAN epsilon value
        self.min_distance = 0.61
        self.max_distance = 10
        self.offset_length = 0.6
        self.offset_standby = 1.0

    def lidar_callback(self, msg):
        detected_points = self.filter_points(msg)
        detected_points.sort(key=lambda p: p[0])

        grouped_points = self.group_points_with_dbscan(detected_points)

        if len(grouped_points) >= 2:
            x_avg1 = sum(p[0] for p in grouped_points[0]) / len(grouped_points[0])
            y_avg1 = sum(p[1] for p in grouped_points[0]) / len(grouped_points[0])

            x_avg2 = sum(p[0] for p in grouped_points[1]) / len(grouped_points[1])
            y_avg2 = sum(p[1] for p in grouped_points[1]) / len(grouped_points[1])
            distance = self.calculate_distance((x_avg1, y_avg1), (x_avg2, y_avg2))
            
            rospy.loginfo(f"Measured distance: {distance:.2f} meters")
            if self.min_distance <= distance <= self.max_distance:
                x_midpoint = (x_avg1 + x_avg2) / 2
                y_midpoint = (y_avg1 + y_avg2) / 2
                angle = math.atan2(y_midpoint, x_midpoint)
                angle2 = math.atan2(y_avg2 - y_avg1, x_avg2 - x_avg1)
                perpendicular_angle = angle2 + math.pi / 2
                offset_x = self.offset_length * math.cos(perpendicular_angle)
                offset_y = self.offset_length * math.sin(perpendicular_angle)

                x_outer1 = x_avg1 + offset_x
                y_outer1 = y_avg1 + offset_y
                x_outer2 = x_avg2 + offset_x
                y_outer2 = y_avg2 + offset_y

                x_centroid = (x_avg1 + x_avg2 + x_outer1 + x_outer2) / 4
                y_centroid = (y_avg1 + y_avg2 + y_outer1 + y_outer2) / 4

                quaternion = tf.transformations.quaternion_from_euler(0, 0, angle)
                self.broadcast_group_transforms(grouped_points, quaternion)
                self.send_transform3(x_outer1, y_outer1, "outer_corner_1", quaternion)
                self.send_transform3(x_outer2, y_outer2, "outer_corner_2", quaternion)
                self.send_transform3(x_midpoint, y_midpoint, "midpoint", quaternion)
                self.send_transform3(x_centroid, y_centroid, "centroid", quaternion)
                rospy.loginfo(f"angle {math.degrees(angle):.2f} degrees")
                rospy.loginfo(f"midpoint coordinate: ({x_midpoint:.2f}, {y_midpoint:.2f}), {quaternion}")

            self.last_object_time = rospy.Time.now()

    def filter_points(self, msg):
        detected_points = []
        for i in range(len(msg.ranges)):
            if self.intensity_min <= msg.intensities[i] <= self.intensity_max and msg.ranges[i] < float('inf'):
                angle = msg.angle_min + i * msg.angle_increment
                x = msg.ranges[i] * math.cos(angle)
                y = msg.ranges[i] * math.sin(angle)
                detected_points.append((x, y))
        return detected_points

    def group_points_with_dbscan(self, detected_points):
        if not detected_points:
            return []

        points_array = np.array(detected_points)
        clustering = DBSCAN(eps=self.proximity_threshold, min_samples=1).fit(points_array)
        labels = clustering.labels_

        grouped_points = []
        for label in set(labels):
            if label == -1:  # Skip noise points
                continue
            grouped_points.append(points_array[labels == label].tolist())
        return grouped_points

    def broadcast_group_transforms(self, grouped_points, quaternion):
        for i in range(min(2, len(grouped_points))):
            group = grouped_points[i]
            x_avg = sum(p[0] for p in group) / len(group)
            y_avg = sum(p[1] for p in group) / len(group)
            self.send_transform3(x_avg, y_avg, f"group_{i}", quaternion)

    def send_transform3(self, x, y, child_frame_id, quaternion):
        self.tf_broadcaster.sendTransform(
            (x, y, 0),
            quaternion,
            rospy.Time.now(),
            child_frame_id,
            "lidar_merged_link"
        )

    def calculate_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def remove_transform(self, event):
        if self.last_object_time and (rospy.Time.now() - self.last_object_time).to_sec() > 0.5:
            rospy.logwarn("Removing transform for object.")
            self.last_object_time = None


if __name__ == '__main__':
    try:
        DistanceCalculator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
