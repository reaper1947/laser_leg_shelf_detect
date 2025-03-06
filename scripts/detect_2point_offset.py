#!/usr/bin/env python
import math
import rospy
from sensor_msgs.msg import LaserScan
import tf
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker, MarkerArray


class DistanceCalculator:
    def __init__(self):
        rospy.init_node('distance_calculator', anonymous=True)

        self.lidar_sub_filtered = rospy.Subscriber(
            '/scan_filtered', LaserScan, self.lidar_callback, queue_size=10)
        self.marker_publisher = rospy.Publisher('/markers', MarkerArray, queue_size=10)

        self.tf_broadcaster = tf.TransformBroadcaster()
        self.last_object_time = None
        self.transforms = []  

        self.timer = rospy.Timer(rospy.Duration(0.1), self.remove_transform)

        # Define intensity thresholds
        self.intensity_min = 130
        self.intensity_max = 160
    
        self.proximity_threshold = 0.1
        self.min_distance = 0.60
        self.max_distance = 10
        self.offset_length = 0.6
        self.offset_standby = 1.0

    def lidar_callback(self, msg):
        range_msg = min(msg.ranges)
        detected_points = self.filter_points(msg)
        detected_points.sort(key=lambda p: p[0])

        grouped_points = self.group_points(detected_points)

        if len(grouped_points) >= 2:
            x_avg1 = sum(p[0] for p in grouped_points[0]) / len(grouped_points[0])
            y_avg1 = sum(p[1] for p in grouped_points[0]) / len(grouped_points[0])

            x_avg2 = sum(p[0] for p in grouped_points[1]) / len(grouped_points[1])
            y_avg2 = sum(p[1] for p in grouped_points[1]) / len(grouped_points[1])
            distance = self.calculate_distance((x_avg1, y_avg1), (x_avg2, y_avg2))
            marker_array = MarkerArray()

            rospy.loginfo(f"Measured distance: {distance:.2f} meters")
            if self.min_distance <= distance <= self.max_distance:
                x_midpoint = (x_avg1 + x_avg2) / 2
                y_midpoint = (y_avg1 + y_avg2) / 2
                angle = math.atan2(y_midpoint, x_midpoint)
                angle2 = math.atan2(y_avg2 - y_avg1, x_avg2 - x_avg1)
                perpendicular_angle = angle2 + math.pi / 2
                offset_x = self.offset_length * -math.cos(perpendicular_angle)
                offset_y = self.offset_length * -math.sin(perpendicular_angle)

                x_outer1 = x_avg1 + offset_x
                y_outer1 = y_avg1 + offset_y
                x_outer2 = x_avg2 + offset_x
                y_outer2 = y_avg2 + offset_y

                x_centroid = (x_avg1 + x_avg2 + x_outer1 + x_outer2) / 4
                y_centroid = (y_avg1 + y_avg2 + y_outer1 + y_outer2) / 4

                quaternion = tf.transformations.quaternion_from_euler(0, 0, angle)
                self.broadcast_group_transforms(grouped_points, quaternion)
                # self.send_transform3(x_outer1, y_outer1, "outer_corner_1", quaternion)
                self.create_marker(x_outer1, y_outer1, "outer_corner_1", marker_array)
                self.create_marker(x_outer2, y_outer2, "outer_corner_2", marker_array)

                # self.send_transform3(x_outer2, y_outer2, "outer_corner_2", quaternion)
                self.send_transform3(x_midpoint, y_midpoint, "midpoint", quaternion)
                self.send_transform3(x_centroid, y_centroid, "centroid", quaternion)
                rospy.loginfo(f"angle {math.degrees(angle):.2f} degrees")
                rospy.loginfo(f"midpoint coordinate: ({x_midpoint:.2f}, {y_midpoint:.2f}), {quaternion}")

                outer_width = self.calculate_distance((x_avg2, y_avg2), (x_avg1, y_avg1))
                width = self.calculate_distance((x_outer1, y_outer1), (x_outer2, y_outer2))
                outer_length = self.calculate_distance((x_avg2, y_avg2), (x_outer2, y_outer2))
                rospy.loginfo(f"outer_width: {outer_width:.2f}")
                rospy.loginfo(f"width: {width:.2f}")
                rospy.loginfo(f"outer_length: {outer_length:.2f}")
            self.marker_publisher.publish(marker_array)

            self.last_object_time = rospy.Time.now()

    def filter_points(self , msg):
        detected_points = []
        for i in range(len(msg.ranges)):
            if self.intensity_min <= msg.intensities[i] <= self.intensity_max and msg.ranges[i] < float('inf'):
                angle = msg.angle_min + i * msg.angle_increment

                x = msg.ranges[i] * math.cos(angle)
                y = msg.ranges[i] * math.sin(angle)
                detected_points.append((x, y))
        return detected_points

    def group_points(self, detected_points):
        grouped_points = []
        while detected_points:
            point = detected_points.pop(0)
            group = [point]
            for other_point in detected_points[:]:
                if self.calculate_distance(point, other_point) < self.proximity_threshold:
                    group.append(other_point)
                    detected_points.remove(other_point)
            grouped_points.append(group)
        return grouped_points

    def broadcast_group_transforms(self, grouped_points, marker_array):
        marker_array = MarkerArray()  # Properly initialize as MarkerArray

        for i in range(min(2, len(grouped_points))):
            group = grouped_points[i]
            x_avg = sum(p[0] for p in group) / len(group)
            y_avg = sum(p[1] for p in group) / len(group)
            # self.send_transform3(x_avg, y_avg, f"group_{i}", quaternion)
            self.create_marker(x_avg, y_avg, f"group_{i}", marker_array)
            self.marker_publisher.publish(marker_array)

    def send_transform(self, x, y, child_frame_id):
        self.tf_broadcaster.sendTransform(
            (x, y, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            rospy.Time.now(),
            child_frame_id,
            'lidar_merged_link'
        )

    def send_transform3(self, x, y, child_frame_id, quaternion):
        self.tf_broadcaster.sendTransform(
            (x, y, 0),
            quaternion,
            rospy.Time.now(),
            child_frame_id,
            "lidar_merged_link"
        )

    def create_marker(self, x, y, frame_id, marker_array):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = 'lidar_merged_link'
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0 #if frame_id == "group_1" else 0.0
        marker.color.g = 0.0 #if frame_id == "group_1" else 1.0
        marker.color.b = 0.0

        marker.id = len(marker_array.markers)
        marker.ns = frame_id

        marker_array.markers.append(marker)

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

# #!/usr/bin/env python
# import math
# import rospy
# from sensor_msgs.msg import LaserScan
# import tf
# from geometry_msgs.msg import TransformStamped


# class DistanceCalculator:
#     def __init__(self):
#         rospy.init_node('distance_calculator', anonymous=True)

#         self.lidar_sub_filtered = rospy.Subscriber(
#             '/scan_filtered', LaserScan, self.lidar_callback, queue_size=10)

#         self.tf_broadcaster = tf.TransformBroadcaster()
#         self.last_object_time = None
#         self.transforms = []  

#         self.timer = rospy.Timer(rospy.Duration(0.1), self.remove_transform)

#         # Define intensity thresholds
#         self.intensity_min = 135
#         self.intensity_max = 200
#         self.proximity_threshold = 0.03
#         self.min_distance = 0.66
#         self.max_distance = 0.70
#         self.offset_length = 0.64
#         self.offset_standby = 1.0

#     def lidar_callback(self, msg):
#         range_msg = min(msg.ranges)
#         detected_points = self.filter_points(msg)
#         grouped_points = self.group_points(detected_points)

#         if len(grouped_points) >= 2:
#             x_avg1 = sum(p[0] for p in grouped_points[0]) / len(grouped_points[0])
#             y_avg1 = sum(p[1] for p in grouped_points[0]) / len(grouped_points[0])

#             x_avg2 = sum(p[0] for p in grouped_points[1]) / len(grouped_points[1])
#             y_avg2 = sum(p[1] for p in grouped_points[1]) / len(grouped_points[1])
#             distance = self.calculate_distance((x_avg1, y_avg1), (x_avg2, y_avg2))
            
#             rospy.loginfo(f"Measured distance: {distance:.2f} meters")
            
#             x_midpoint = (x_avg1 + x_avg2) / 2
#             y_midpoint = (y_avg1 + y_avg2) / 2
#             angle = math.atan2(y_midpoint, x_midpoint)
#             angle2 = math.atan2(y_avg2 - y_avg1, x_avg2 - x_avg1)
#             perpendicular_angle = angle2 + math.pi / 2
#             offset_x = self.offset_length * math.cos(perpendicular_angle)
#             offset_y = self.offset_length * math.sin(perpendicular_angle)

#             x_outer1 = x_avg1 + offset_x
#             y_outer1 = y_avg1 + offset_y
#             x_outer2 = x_avg2 + offset_x
#             y_outer2 = y_avg2 + offset_y

#             x_centroid = (x_avg1 + x_avg2 + x_outer1 + x_outer2) / 4
#             y_centroid = (y_avg1 + y_avg2 + y_outer1 + y_outer2) / 4

#             quaternion = tf.transformations.quaternion_from_euler(0, 0, angle)
            
#             rospy.loginfo(f"angle {math.degrees(angle):.2f} degrees")
#             rospy.loginfo(f"midpoint coordinate: ({x_midpoint:.2f}, {y_midpoint:.2f}), {quaternion}")

#             outer_width = self.calculate_distance((x_avg2, y_avg2), (x_avg1, y_avg1))
#             width = self.calculate_distance((x_outer1, y_outer1), (x_outer2, y_outer2))
#             outer_length = self.calculate_distance((x_avg2, y_avg2), (x_outer2, y_outer2))
#             rospy.loginfo(f"outer_width: {outer_width:.2f}")
#             rospy.loginfo(f"width: {width:.2f}")
#             rospy.loginfo(f"outer_length: {outer_length:.2f}")
#             if self.min_distance <= distance <= self.max_distance and outer_width < 0.75:
#                 self.broadcast_group_transforms(grouped_points, quaternion)
#                 self.send_transform3(x_outer1, y_outer1, "outer_corner_1", quaternion)
#                 self.send_transform3(x_outer2, y_outer2, "outer_corner_2", quaternion)
#                 self.send_transform3(x_midpoint, y_midpoint, "midpoint", quaternion)
#                 self.send_transform3(x_centroid, y_centroid, "centroid", quaternion)
#         self.last_object_time = rospy.Time.now()

#     def filter_points(self, msg):
#         detected_points = []
#         for i in range(len(msg.ranges)):
#             if self.intensity_min <= msg.intensities[i] <= self.intensity_max and msg.ranges[i] < float('inf'):
#                 angle = msg.angle_min + i * msg.angle_increment
#                 x = msg.ranges[i] * math.cos(angle)
#                 y = msg.ranges[i] * math.sin(angle)
#                 detected_points.append((x, y))
#         return detected_points

#     def group_points(self, detected_points):
#         grouped_points = []
#         while detected_points:
#             point = detected_points.pop(0)
#             group = [point]
#             for other_point in detected_points[:]:
#                 if self.calculate_distance(point, other_point) < self.proximity_threshold:
#                     group.append(other_point)
#                     detected_points.remove(other_point)
#             grouped_points.append(group)
#         return grouped_points

#     def broadcast_group_transforms(self, grouped_points, quaternion):
#         for i in range(min(2, len(grouped_points))):
#             group = grouped_points[i]
#             x_avg = sum(p[0] for p in group) / len(group)
#             y_avg = sum(p[1] for p in group) / len(group)
#             self.send_transform3(x_avg, y_avg, f"group_{i}", quaternion)

#     def send_transform3(self, x, y, child_frame_id, quaternion):
#         self.tf_broadcaster.sendTransform(
#             (x, y, 0),
#             quaternion,
#             rospy.Time.now(),
#             child_frame_id,
#             "lidar_merged2_link"
#         )

#     def calculate_distance(self, point1, point2):
#         return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

#     def remove_transform(self, event):
#         if self.last_object_time and (rospy.Time.now() - self.last_object_time).to_sec() > 0.5:
#             rospy.logwarn("Removing transform for object.")
#             self.last_object_time = None


# if __name__ == '__main__':
#     try:
#         DistanceCalculator()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass