#!/usr/bin/env python3
import math
import rospy
from sensor_msgs.msg import LaserScan
import tf   
from geometry_msgs.msg import TransformStamped


class DistanceCalculator:
    def __init__(self):
        rospy.init_node('distance_calculator', anonymous=True)

        # self.lidar_sub_filtered = rospy.Subscriber(
        #     '/scan_filtered', LaserScan, self.lidar_callback, queue_size=10)
        self.lidar_sub = rospy.Subscriber(
            '/scan_filtered', LaserScan, self.lidar_callback, queue_size=10)

        self.tf_broadcaster = tf.TransformBroadcaster()
        self.last_object_time = None
        self.transforms = []  

        self.timer = rospy.Timer(rospy.Duration(0.1), self.remove_transform)

    def lidar_callback(self, msg):
        # T1 
        min_intensity = 30 #145
        max_intensity = 50 # 200
        self.offset_length = 0.6
        proximity_threshold = 0.1

        # # C1
        # intensity_threshold = 55
        # proximity_threshold = 0.15

        # #SICK NANOSCAN3
        # intensity_threshold = 235
        # proximity_threshold = 0.13

        detected_points = self.filter_points(msg, min_intensity, max_intensity)
        grouped_points = self.group_points(detected_points, proximity_threshold)

        self.broadcast_group_transforms(grouped_points)

        x_avg1 = sum(p[0] for p in grouped_points[0]) / len(grouped_points[0])
        y_avg1 = sum(p[1] for p in grouped_points[0]) / len(grouped_points[0])

        x_avg2 = sum(p[0] for p in grouped_points[1]) / len(grouped_points[1])
        y_avg2 = sum(p[1] for p in grouped_points[1]) / len(grouped_points[1])

        x_avg3 = sum(p[0] for p in grouped_points[2]) / len(grouped_points[2])
        y_avg3 = sum(p[1] for p in grouped_points[2]) / len(grouped_points[2])

        x_avg4 = sum(p[0] for p in grouped_points[3]) / len(grouped_points[3])
        y_avg4 = sum(p[1] for p in grouped_points[3]) / len(grouped_points[3])
        # # For detect only 4 group
        # # Process each detected object (only the first 4) and filter by size
        # valid_objects = 0
        # for idx, obj in enumerate(detected_points[:4]):
        #     if self.process_object(idx, obj, msg, detected_points):
        #         valid_objects += 1
        #         if valid_objects >= 4:  # Limit to 4 valid objects
        #             break
        outer_width = self.calculate_distance((x_avg4, y_avg4), (x_avg1, y_avg1))
        outer_length = self.calculate_distance((x_avg4, y_avg4), (x_avg3, y_avg3))

        if len(grouped_points) >= 2 and 0.6 <=outer_width<= 0.69 and 0.60 <= outer_length <= 0.69:
            x_centroid = (x_avg1 + x_avg2 + x_avg3 + x_avg4) / 4
            y_centroid = (y_avg1 + y_avg2 + y_avg3 + y_avg4) / 4
            angle = math.atan2(y_centroid, x_centroid)
            angle_degrees = math.degrees(angle)
            quaternion = tf.transformations.quaternion_from_euler(0, 0, angle)  
            rospy.loginfo(f"angle {angle_degrees:.2f} c")
            self.send_transform3(x_centroid, y_centroid, "centroid", quaternion)
            rospy.loginfo(f"Centroid coordinate: ({x_centroid}, {y_centroid})")
            rospy.loginfo(f"outer_width: {outer_width:.2f}")
            rospy.loginfo(f"outer_length: {outer_length:.2f}")
        elif len(grouped_points) >= 2 and 0.67 <=outer_width<= 0.69:
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
            self.send_transform3(x_outer1, y_outer1, "outer_corner_1", quaternion)
            self.send_transform3(x_outer2, y_outer2, "outer_corner_2", quaternion)
            rospy.loginfo(f"[     ] ON MIDDLE SHELF NOW [    ]")
            

        self.last_object_time = rospy.Time.now()

    def filter_points(self, msg, min_intensity, max_intensity):
        detected_points = []
        for i in range(len(msg.ranges)):
            if min_intensity <= msg.intensities[i] <= max_intensity and msg.ranges[i] < float('inf'):
                angle = msg.angle_min + i * msg.angle_increment
                x = msg.ranges[i] * math.cos(angle)
                y = msg.ranges[i] * math.sin(angle)
                detected_points.append((x, y))
                
        return detected_points

    def group_points(self, detected_points, proximity_threshold):
        grouped_points = []
        while detected_points:
            point = detected_points.pop(0)
            group = [point]
            for other_point in detected_points[:]:
                if self.calculate_distance(point, other_point) < proximity_threshold:
                    group.append(other_point)
                    detected_points.remove(other_point)
            grouped_points.append(group)
        return grouped_points

    def broadcast_group_transforms(self, grouped_points):
        for i in range(min(4, len(grouped_points))):
            group = grouped_points[i]
            x_avg = sum(p[0] for p in group) / len(group)
            y_avg = sum(p[1] for p in group) / len(group)
            self.send_transform(x_avg, y_avg, f"group_{i}")

    def send_transform(self, x, y, child_frame_id):
        self.tf_broadcaster.sendTransform(
            (x, y, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            rospy.Time.now(),
            child_frame_id,
            'laser'
        )

    def send_transform3(self, x, y, child_frame_id, quaternion):
        self.tf_broadcaster.sendTransform(
            (x, y, 0.0),
            quaternion,
            rospy.Time.now(),
            child_frame_id,
            'laser'
        )

    def calculate_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

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
