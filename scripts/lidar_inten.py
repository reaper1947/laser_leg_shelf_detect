#!/usr/bin/env python3
import rospy
import math
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
import tf
import tf.transformations
from std_msgs.msg import ColorRGBA
from dynamic_reconfigure.server import Server
from lidar_inten_detect.cfg import DistanceCalculatorConfig
from visualization_msgs.msg import Marker, MarkerArray
from sklearn.cluster import DBSCAN
import numpy as np


############################ /// LIDAR_DETECT & CALCULATE /// ##################################
############################### /// MOVEBASE_CONTROL /// ######################################
################################## /// PUB_MARKER /// ##########################################
################################# /// GROUP_OBJECT /// #########################################
################################### /// TF_PUB /// #############################################
############################### /// SETUP_RECONFIG /// #########################################

class DistanceCalculator:
    def __init__(self):
        rospy.init_node('distance_calculator', anonymous=True)
        self.server = Server(DistanceCalculatorConfig, self.reconfigure_callback)

        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        self.marker_pub = rospy.Publisher('/intensity_marker', Marker, queue_size=10)
        self.marker_publisher = rospy.Publisher('/markers', MarkerArray, queue_size=10)

        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

        self.tf_broadcaster = tf.TransformBroadcaster()
        self.last_object_time = None
        self.transforms = []
        self.angle_robot = 0.0  
        self.intensity_threshold = 50
        self.proximity_threshold = 0.04
        self.min_distance = 0.60
        self.max_distance = 0.65
        self.offset_length = 0.64
        self.offset_standby = 1.0
        self.x_min = -3
        self.x_max = 3
        self.y_min = -1.5
        self.y_max = 1.5
        self.min_sample = 3
        self.last_object_time = rospy.Time.now()
        rospy.Timer(rospy.Duration(0.1), self.remove_transform)
        # rospy.Timer(rospy.Duration(1), self.send_transform3)
        # rospy.Rate(1000)

############################### /// LIDAR_DETECT & CALCULATE /// ######################################
    def calculate_angle_to_centroid(self, x_robot, y_robot, x_centroid, y_centroid):
        return math.atan2(y_centroid - y_robot, x_centroid - x_robot)
    
    def calculate_angle_difference(self, angle_robot, angle_centroid):
        angle_diff = angle_robot - angle_centroid
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi  
        return math.degrees(angle_diff)
   
    def lidar_callback(self, msg):
        self.intensity_threshold 
        self.proximity_threshold 
        self.min_distance 
        self.max_distance
        self.offset_length 
        self.offset_standby 
        self.x_min
        self.x_max
        self.y_min
        self.y_max
        self.min_sample
        marker_array = MarkerArray()

        cropped_points = []
        cropped_intensities = []
        for i in range(len(msg.ranges)):
            distance = msg.ranges[i]
            if distance == float('inf') or distance == 0.0:
                continue

            angle = msg.angle_min + i * msg.angle_increment
            x = distance * math.cos(angle)
            y = distance * math.sin(angle)

            if self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max:
                cropped_points.append((x, y))
                if msg.intensities:
                    cropped_intensities.append(msg.intensities[i])

        self.publish_bounding_box(self.x_min, self.x_max, self.y_min, self.y_max)
        # self.publish_intensity_markers(cropped_points, cropped_intensities)

        detected_points = self.filter_points(msg, self.intensity_threshold)
        detected_points.sort(key=lambda p: p[0])

        grouped_points = self.group_points_dbscan(detected_points)
        if len(grouped_points) >= 2:
            x_avg1 = sum(p[0] for p in grouped_points[0]) / len(grouped_points[0])
            y_avg1 = sum(p[1] for p in grouped_points[0]) / len(grouped_points[0])

            x_avg2 = sum(p[0] for p in grouped_points[1]) / len(grouped_points[1])
            y_avg2 = sum(p[1] for p in grouped_points[1]) / len(grouped_points[1])

            # x_avg3 = sum(p[0] for p in grouped_points[2]) / len(grouped_points[2])
            # y_avg3 = sum(p[1] for p in grouped_points[2]) / len(grouped_points[2])

            # x_avg4 = sum(p[0] for p in grouped_points[3]) / len(grouped_points[3])
            # y_avg4 = sum(p[1] for p in grouped_points[3]) / len(grouped_points[3])

            distance = self.calculate_distance((x_avg1, y_avg1), (x_avg2, y_avg2))
            rospy.loginfo(f"Measured distance: {distance:.2f} meters")
            if self.min_distance <= distance <= self.max_distance:
                x_midpoint = (x_avg1 + x_avg2) / 2
                y_midpoint = (y_avg1 + y_avg2) / 2
                angle = math.atan2(y_midpoint, x_midpoint)
                angle2 = math.atan2(y_avg2 - y_avg1, x_avg2 - x_avg1)
                angle_degrees = math.degrees(angle)
                perpendicular_angle = angle2 + math.pi / 2
                offset_x = self.offset_length * math.cos(perpendicular_angle)
                offset_y = self.offset_length * math.sin(perpendicular_angle)
                offset_x_stanby = self.offset_standby * math.cos(perpendicular_angle)
                offset_y_stanby = self.offset_standby * math.sin(perpendicular_angle)

                x_outer1 = x_avg1 + -offset_x
                y_outer1 = y_avg1 + -offset_y
                x_outer2 = x_avg2 + -offset_x
                y_outer2 = y_avg2 + -offset_y

                x_centroid = (x_avg1 + x_avg2 + x_outer1 + x_outer2) / 4
                y_centroid = (y_avg1 + y_avg2 + y_outer1 + y_outer2) / 4

                # x_stanby = x_midpoint + offset_x_stanby
                # y_stanby = y_midpoint + offset_y_stanby

                quaternion = tf.transformations.quaternion_from_euler(0, 0, angle)
                self.broadcast_group_transforms(grouped_points, quaternion)
                # self.send_transform3(x_outer1, y_outer1, "outer_corner_1", quaternion)
                # self.send_transform3(x_outer2, y_outer2, "outer_corner_2", quaternion)
                self.create_line_marker(x_avg1, y_avg1, x_avg2, y_avg2, marker_array)

                self.create_marker(x_outer1, y_outer1, "outer_corner_1", marker_array)
                self.create_marker(x_outer2, y_outer2, "outer_corner_2", marker_array)

                # self.send_transform3(x_stanby, y_stanby, "stanby", quaternion)
                # x_midpoint_adjusted = x_midpoint * math.cos(angle) - y_midpoint * math.sin(angle)
                # y_midpoint_adjusted = x_midpoint * math.sin(angle) + y_midpoint * math.cos(angle)
                # self.send_transform3(x_midpoint, y_midpoint, "midpoint", quaternion)
                # self.send_transform3(x_centroid, y_centroid, "centroid", quaternion)
                # self.create_marker(x_midpoint, y_midpoint, "midpoint", marker_array)
                self.create_marker_centroid(x_centroid, y_centroid, "centroid", marker_array)

                rospy.loginfo(f"midpoint coordinate: ({x_midpoint:.2f}, {y_midpoint:.2f}),{quaternion}")
                # outer_width = self.calculate_distance((x_avg4, y_avg4), (x_avg1, y_avg1))
                # distance2 = self.calculate_distance((x_avg4, y_avg4), (x_avg3, y_avg3))
                outer_width = self.calculate_distance((x_avg2, y_avg2), (x_avg1, y_avg1))
                width = self.calculate_distance((x_outer1, y_outer1), (x_outer2, y_outer2))
                angle_centroid = self.calculate_angle_to_centroid(x_midpoint, y_midpoint, x_centroid, y_centroid)
                angle_diff_degrees = self.calculate_angle_difference(self.angle_robot, angle_centroid)

                rospy.loginfo(f"Angle difference (degrees): {angle_diff_degrees:.2f}")
                outer_length = self.calculate_distance((x_avg2, y_avg2), (x_outer2, y_outer2))
                rospy.loginfo(f"outer_width: {outer_width:.2f}")
                rospy.loginfo(f"width: {width:.2f}")
                rospy.loginfo(f"outer_length: {outer_length:.2f}")
                self.marker_publisher.publish(marker_array)

                # self.publish_goal(x_midpoint, y_midpoint, angle)

                # self.get_logger().info(f"outer_length: {distance2:.2f}")
            self.last_object_time = rospy.Time.now()

############################### /// MOVEBASE_CONTROL /// ######################################
    def publish_goal(self, x, y, angle):
        goal_msg = PoseStamped()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = 'base_link'

        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.position.z = 0.0

        quaternion = tf.transformations.quaternion_from_euler(0, 0, angle)
        goal_msg.pose.orientation.x = quaternion[0]
        goal_msg.pose.orientation.y = quaternion[1]
        goal_msg.pose.orientation.z = quaternion[2]
        goal_msg.pose.orientation.w = quaternion[3]

        self.goal_pub.publish(goal_msg)
        rospy.loginfo(
            f"Published goal: x={x:.2f}, y={y:.2f}, angle={math.degrees(angle):.2f}°")
        
############################### /// PUB_MARKER /// #############################################
    def publish_bounding_box(self, x_min, x_max, y_min, y_max):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = 'laser'
        marker.ns = "bounding_box"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.01
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        points = [
            Point(x=x_min, y=y_min, z=0.0),
            Point(x=x_max, y=y_min, z=0.0),
            Point(x=x_max, y=y_max, z=0.0),
            Point(x=x_min, y=y_max, z=0.0),
            Point(x=x_min, y=y_min, z=0.0)
        ]
        marker.points.extend(points)
        self.marker_pub.publish(marker)

    def publish_intensity_markers(self, points, intensities):
        if not points or not intensities:
            return

        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = 'laser'
        marker.ns = "intensity_points"
        marker.id = 1
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.02
        marker.scale.y = 0.02
        marker.color.a = 1.0

        for i, (x, y) in enumerate(points):
            intensity = intensities[i]
            point = Point(x=x, y=y, z=0.0)
            marker.points.append(point)

            intensity_normalized = max(0.0, min((intensity - 40) / (60 - 40), 1.0))
            marker.colors.append(
                ColorRGBA(
                    r=intensity_normalized,
                    g=0.0,
                    b=0.0 -
                    intensity_normalized,
                    a=1.0))

        self.marker_pub.publish(marker)

    def create_marker(self, x, y, frame_id, marker_array):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = 'laser'
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.a = 1.0
        marker.color.r = 1.0 #if frame_id == "group_1" else 0.0
        marker.color.g = 0.0 #if frame_id == "group_1" else 1.0
        marker.color.b = 0.0

        marker.id = len(marker_array.markers)
        marker.ns = frame_id

        marker_array.markers.append(marker)


    def create_marker_centroid(self, x, y, frame_id, marker_array):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = 'laser'
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.a = 1.0
        marker.color.r = 0.0 #if frame_id == "group_1" else 0.0
        marker.color.g = 0.0 #if frame_id == "group_1" else 1.0
        marker.color.b = 1.0

        marker.id = len(marker_array.markers)
        marker.ns = frame_id

        marker_array.markers.append(marker)


    def create_line_marker(self, x1, y1, x2, y2, marker_array):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = 'laser'
        marker.ns = "line_connection"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.02  # Line thickness
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0  # Green color for the line
        marker.color.b = 0.0

        points = [Point(x=x1, y=y1, z=0.0), Point(x=x2, y=y2, z=0.0)]
        marker.points.extend(points)
        marker_array.markers.append(marker)
        
############################### /// GROUP_OBJECT /// #############################################
    def filter_points(self, msg, intensity_threshold):
        detected_points = []
        for i in range(len(msg.ranges)):
            if msg.intensities[i] > intensity_threshold and msg.ranges[i] < float('inf'):
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
    
    def group_points_dbscan(self, detected_points):
        if not detected_points:
            return []
        detected_points_np = np.array(detected_points)
        db = DBSCAN(eps=self.proximity_threshold, min_samples=self.min_sample).fit(detected_points_np)
        grouped_points = [
            detected_points_np[db.labels_ == label].tolist()
            for label in set(db.labels_) if label != -1
        ]
        return grouped_points
    

############################### /// TF_PUB /// #################################################
    def broadcast_group_transforms(self, grouped_points, marker_array):
        marker_array = MarkerArray()  # Properly initialize as MarkerArray

        for i in range(min(2, len(grouped_points))):
            group = grouped_points[i]
            x_avg = sum(p[0] for p in group) / len(group)
            y_avg = sum(p[1] for p in group) / len(group)
            # self.send_transform3(x_avg, y_avg, f"group_{i}", quaternion)
            self.create_marker(x_avg, y_avg, f"group_{i}", marker_array)
            self.marker_publisher.publish(marker_array)

    def send_transform3(self, x, y, name, quaternion):
        if self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max:
            self.tf_broadcaster.sendTransform(
                (x, y, 0),
                quaternion,
                rospy.Time.now(),
                name,
                "laser"
            )
            rospy.loginfo(f"Published tf for {name}: x={x:.2f}, y={y:.2f}")
        else:
            rospy.loginfo(f"Skipped tf for {name}: x={x:.2f}, y={y:.2f} is out of bounding box")

    def calculate_distance(self, point1, point2):
        return math.dist(point1, point2)
    
    def remove_transform(self, event):
        if self.last_object_time and (rospy.Time.now() - self.last_object_time).to_sec() > 0.5:
            rospy.logwarn("Removing transform for object.")
            self.last_object_time = None   
############################### /// SETUP_RECONFIG /// ######################################
    def reconfigure_callback(self, config, level):
        self.intensity_threshold = config.intensity_threshold
        self.proximity_threshold = config.proximity_threshold
        self.min_distance = config.min_distance
        self.max_distance = config.max_distance
        self.offset_length = config.offset_length
        self.offset_standby = config.offset_standby
        self.x_min = config.x_min
        self.x_max = config.x_max
        self.y_min = config.y_min
        self.y_max = config.y_max
        self.min_sample = config.min_sample

        rospy.loginfo(f"Reconfigure Request: {config}")
        return config
    
if __name__ == '__main__':
    try:
        DistanceCalculator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass