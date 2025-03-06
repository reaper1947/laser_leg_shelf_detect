#!/usr/bin/env python
import rospy
import math
import tf
import numpy as np

from std_msgs.msg import Bool
from dynamic_reconfigure.server import Server
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from sklearn.cluster import DBSCAN
from lidar_inten_detect.cfg import DistanceCalculatorConfig
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus

class DistanceCalculator:
    def __init__(self):
        rospy.init_node('distance_calculator', anonymous=True)
        self.subscriber = rospy.Subscriber('/lds2/scan', LaserScan, self.lidar_callback,
                                           queue_size=1, buff_size=2**24)
        self.marker_publisher = rospy.Publisher('/group_markers', MarkerArray, queue_size=1)
        self.shelf_publisher = rospy.Publisher('/shelf_detected', Bool, queue_size=1)
        self.tick = rospy.Publisher('/tick', Bool, queue_size=1)
        self.tick_subscriber = rospy.Subscriber('/tick', Bool, self.tick_callback, queue_size=1)

        # TF broadcaster for sending transforms (e.g., shelf position)
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.last_tf_time = rospy.Time.now()
        self.tf_timeout = rospy.Duration(0.1)

        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move_base_client.wait_for_server()

        self.srv = Server(DistanceCalculatorConfig, self.reconfigure_callback)

        # Shelf detection parameters (example values)
        self.shelf_width = 1.0
        self.shelf_width_tolerance = 0.30
        self.intensity_threshold = 35
        self.eps = 0.25
        self.min_samples = 10
        self.dist_threshold = 0.8
        self.offset_length  = 1.2

        # Store the last detected shelf pose (x, y, yaw)
        self.last_shelf_pose = None

        # Flag indicating whether the robot is currently navigating
        self.is_navigating = False

    def reconfigure_callback(self, config, level):
        # Update parameters from dynamic reconfigure
        self.shelf_width = config.shelf_width
        self.shelf_width_tolerance = config.shelf_width_tolerance
        self.intensity_threshold = config.intensity_threshold
        self.eps = config.eps
        self.min_samples = config.min_samples
        self.dist_threshold = config.dist_threshold
        self.offset_length = config.offset_length
        rospy.loginfo("Reconfigured parameters:")
        rospy.loginfo(f"Shelf Width: {self.shelf_width}, Shelf Tolerance: {self.shelf_width_tolerance}, "
                      f"Intensity Threshold: {self.intensity_threshold}, EPS: {self.eps}, Min Samples: {self.min_samples}, "
                      f"Dist Threshold: {self.dist_threshold}, Offset Length: {self.offset_length}")
        return config

    def lidar_callback(self, msg):
        # If the robot is currently navigating, skip processing new LiDAR data
        if self.is_navigating:
            return

        # Filter LiDAR points based on intensity threshold and sort by x-coordinate (descending)
        detected_points = self.filter_points(msg, self.intensity_threshold)
        detected_points.sort(key=lambda x: x[0], reverse=True)

        if not detected_points:
            rospy.loginfo("No points detected above threshold. Shelf not found.")
            self.shelf_publisher.publish(False)
            self.clear_transform()
            self.last_shelf_pose = None
            return

        # Convert the list of detected points into a NumPy array
        points = np.array(detected_points)
        # Cluster the points using DBSCAN
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points)
        labels = clustering.labels_
        # Remove noise points (label -1)
        unique_labels = set(labels)
        unique_labels.discard(-1)

        if not unique_labels:
            rospy.loginfo("No valid clusters. Shelf not found.")
            self.shelf_publisher.publish(False)
            self.clear_transform()
            self.last_shelf_pose = None
            return

        self.last_tf_time = rospy.Time.now()
        # Group points by their cluster labels
        clusters = [points[labels == label] for label in unique_labels]

        # Calculate the average distance of points in each cluster
        cluster_distances = []
        for c in clusters:
            d_array = [math.sqrt(p[0]**2 + p[1]**2) for p in c]
            avg_d = np.mean(d_array) if d_array else float('inf')
            cluster_distances.append(avg_d)
        min_dist = min(cluster_distances) if cluster_distances else float('inf')

        # Decide which shelf detection logic to use based on the closest cluster distance
        if min_dist < self.dist_threshold:
            self.handle_shelf_4legs(clusters, msg)
        else:
            self.handle_shelf_2legs_plus_offset(clusters, msg)

    def handle_shelf_4legs(self, clusters, msg):
        if len(clusters) < 4:
            rospy.loginfo("Less than 4 clusters found in near range. Shelf not found.")
            self.shelf_publisher.publish(False)
            self.clear_transform()
            self.last_shelf_pose = None
            return

        x_avg_list, y_avg_list, intensity_avg_list = [], [], []
        marker_array = MarkerArray()

        for i, cluster in enumerate(clusters[:4]):
            x_avg = np.mean(cluster[:, 0])
            y_avg = np.mean(cluster[:, 1])
            intensity_avg = np.mean([
                msg.intensities[np.argmin([abs(r - self.calculate_range(p)) for r in msg.ranges])]
                for p in cluster
            ])
            x_avg_list.append(x_avg)
            y_avg_list.append(y_avg)
            intensity_avg_list.append(intensity_avg)
            marker_array.markers.append(self.create_marker(x_avg, y_avg, i + 1))

        self.marker_publisher.publish(marker_array)

        # Calculate distances between front pair and back pair of legs
        dist_front = self.calculate_distance((x_avg_list[0], y_avg_list[0]),
                                             (x_avg_list[1], y_avg_list[1]))
        dist_back = self.calculate_distance((x_avg_list[2], y_avg_list[2]),
                                            (x_avg_list[3], y_avg_list[3]))

        if (abs(dist_front - self.shelf_width) <= self.shelf_width_tolerance and
            abs(dist_back - self.shelf_width) <= self.shelf_width_tolerance):

            rospy.loginfo("==== Shelf Detected (4 legs)! ====")
            self.shelf_publisher.publish(True)

            # Compute the midpoint of the front legs (assumed to be indices 0 and 1)
            front_center_x = (x_avg_list[0] + x_avg_list[1]) / 2.0
            front_center_y = (y_avg_list[0] + y_avg_list[1]) / 2.0

            # Compute the midpoint of the back legs (assumed to be indices 2 and 3)
            back_center_x = (x_avg_list[2] + x_avg_list[3]) / 2.0
            back_center_y = (y_avg_list[2] + y_avg_list[3]) / 2.0

            # Calculate yaw as the angle from the back center to the front center
            yaw = math.atan2(front_center_y - back_center_y, front_center_x - back_center_x)

            # Compute the overall shelf centroid (average of front and back centers)
            x_centroid = (front_center_x + back_center_x) / 2.0
            y_centroid = (front_center_y + back_center_y) / 2.0

            # Convert the yaw angle to a quaternion
            quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)

            # Now, send the transform with the computed centroid and orientation.
            self.send_transform(x_centroid, y_centroid, "shelf", quaternion)

            x_midpoint = (x_avg_list[0] + x_avg_list[1]) / 2
            y_midpoint = (y_avg_list[0] + y_avg_list[1]) / 2
            x_centroid_4 = sum(x_avg_list) / 4
            y_centroid_4 = sum(y_avg_list) / 4
            self.send_transform(x_midpoint, y_midpoint, "midpoint", quaternion)
            self.send_transform(x_centroid_4, y_centroid_4, "centroid", quaternion)

            # Store the last detected shelf pose for navigation
            self.last_shelf_pose = (x_centroid, y_centroid, yaw)
        else:
            rospy.loginfo("Distances do not match shelf_width ± tolerance (4 legs). Shelf not found.")
            self.shelf_publisher.publish(False)
            self.clear_transform()
            self.last_shelf_pose = None

    def handle_shelf_2legs_plus_offset(self, clusters, msg):
        if len(clusters) < 2:
            rospy.loginfo("Less than 2 clusters found in far range. Shelf not found.")
            self.shelf_publisher.publish(False)
            self.clear_transform()
            self.last_shelf_pose = None
            return

        rospy.loginfo("Detected 2 legs. Creating back legs with offset logic...")

        marker_array = MarkerArray()

        # Get the centroids of the two detected clusters (assumed to be the front legs)
        front_cluster_1 = clusters[0]
        front_cluster_2 = clusters[1]
        x1 = np.mean(front_cluster_1[:, 0])
        y1 = np.mean(front_cluster_1[:, 1])
        x2 = np.mean(front_cluster_2[:, 0])
        y2 = np.mean(front_cluster_2[:, 1])

        # Compute the midpoint of the front legs
        front_center_x = (x1 + x2) / 2.0
        front_center_y = (y1 + y2) / 2.0

        # Calculate a perpendicular vector to the line connecting the two front legs
        dx = x2 - x1
        dy = y2 - y1
        perp_x = -dy
        perp_y = dx
        norm = math.sqrt(perp_x**2 + perp_y**2) + 1e-9
        perp_x /= norm
        perp_y /= norm

        # Generate virtual back leg positions using the specified offset_length
        back1_x = x1 + self.offset_length * perp_x
        back1_y = y1 + self.offset_length * perp_y
        back2_x = x2 + self.offset_length * perp_x
        back2_y = y2 + self.offset_length * perp_y

        # Compute the midpoint of the back legs
        back_center_x = (back1_x + back2_x) / 2.0
        back_center_y = (back1_y + back2_y) / 2.0

        # Compute the overall shelf centroid as the average of the front and back centers
        x_centroid_shelf = (front_center_x + back_center_x) / 2.0
        y_centroid_shelf = (front_center_y + back_center_y) / 2.0

        # Calculate yaw as the angle from the back center to the front center
        yaw = math.atan2(front_center_y - back_center_y, front_center_x - back_center_x)

        # Create markers for visualization: front legs, back legs, etc.
        x_list = [x1, x2, back1_x, back2_x]
        y_list = [y1, y2, back1_y, back2_y]
        for i, (xx, yy) in enumerate(zip(x_list, y_list)):
            marker_array.markers.append(self.create_marker(xx, yy, i+1))
        self.marker_publisher.publish(marker_array)

        # Convert yaw to a quaternion
        quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
        # Send the shelf transform with the computed centroid and orientation
        self.send_transform(x_centroid_shelf, y_centroid_shelf, "shelf", quaternion)
        rospy.loginfo(f"==== Shelf Detected (2 legs + offset)! Centroid=({x_centroid_shelf:.2f}, {y_centroid_shelf:.2f}) ====")
        self.shelf_publisher.publish(True)

        # Additionally, send a transform for the midpoint of the front legs (if needed)
        self.send_transform(front_center_x, front_center_y, "midpoint_offset", quaternion)

        # Store the detected shelf pose for navigation
        self.last_shelf_pose = (x_centroid_shelf, y_centroid_shelf, yaw)


    def tick_callback(self, msg):
        if msg.data and self.last_shelf_pose is not None:
            x, y, yaw = self.last_shelf_pose
            rospy.loginfo("Tick received. Navigating to Shelf at ({:.2f}, {:.2f}) with yaw {:.2f}".format(x, y, yaw))
            self.navigate_to_shelf(x, y, yaw)

            self.tick.publish(False)
        else:
            rospy.loginfo("Tick received but no valid shelf detected or tick is false.")

    def navigate_to_shelf(self, x, y, yaw):
        # Set the navigation flag to True to pause LiDAR processing during navigation
        self.is_navigating = True

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "lidar_2_link"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x 
        goal.target_pose.pose.position.y = y 
        quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
        goal.target_pose.pose.orientation.x = quaternion[0]
        goal.target_pose.pose.orientation.y = quaternion[1]
        goal.target_pose.pose.orientation.z = quaternion[2]
        goal.target_pose.pose.orientation.w = quaternion[3]

        rospy.loginfo("Sending robot to centroid position")
        self.move_base_client.send_goal(goal)

        while not rospy.is_shutdown():
            state = self.move_base_client.get_state()

            if state == GoalStatus.PENDING:
                rospy.loginfo("MoveBase: Goal pending...")
            elif state == GoalStatus.ACTIVE:
                rospy.loginfo("MoveBase: Goal is being processed...")
            elif state == GoalStatus.PREEMPTED:
                rospy.logwarn("MoveBase: Goal was preempted.")
                break
            elif state == GoalStatus.SUCCEEDED:
                rospy.loginfo("MoveBase: Goal reached successfully!")
                break
            elif state == GoalStatus.ABORTED:
                rospy.logerr("MoveBase: Goal was aborted. The robot could not reach the target.")
                break
            elif state == GoalStatus.REJECTED:
                rospy.logerr("MoveBase: Goal was rejected. Check if move_base is running properly.")
                break
            elif state == GoalStatus.LOST:
                rospy.logerr("MoveBase: Goal lost. Something went wrong.")
                break

            rospy.sleep(1)

        # Reset the navigation flag to resume LiDAR processing
        self.is_navigating = False

    def clear_transform(self):
        if rospy.Time.now() - self.last_tf_time > self.tf_timeout:
            self.tf_broadcaster.sendTransform(
                (0, 0, 0),
                tf.transformations.quaternion_from_euler(0, 0, 0),
                rospy.Time.now(),
                "shelf",
                "lidar_2_link"
            )

    def filter_points(self, msg, intensity_threshold):
        detected_points = []
        for i in range(len(msg.ranges)):
            if msg.intensities[i] > intensity_threshold and msg.ranges[i] < float('inf'):
                angle = msg.angle_min + i * msg.angle_increment
                x = msg.ranges[i] * math.cos(angle)
                y = msg.ranges[i] * math.sin(angle)
                detected_points.append((x, y))
        return detected_points

    def create_marker(self, x, y, group_id):
        marker = Marker()
        marker.header.frame_id = "lidar_2_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "group_markers"
        marker.id = group_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.1
        marker.color.a, marker.color.b = 1.0, 1.0
        return marker

    def send_transform(self, x, y, child_frame_id, quaternion):
        self.tf_broadcaster.sendTransform(
            (x, y, 0),
            quaternion,
            rospy.Time.now(),
            child_frame_id,
            "lidar_2_link"
        )

    def calculate_distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def calculate_range(self, point):
        # Compute Euclidean norm (distance) of a point from the origin
        return np.linalg.norm(point)


if __name__ == '__main__':
    DistanceCalculator()
    rospy.spin()
