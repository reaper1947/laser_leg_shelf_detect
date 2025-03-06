import math
import rospy
from sensor_msgs.msg import LaserScan
import tf
import geometry_msgs.msg
import tf
import tf.transformations

class DistanceCalculator:

    def __init__(self):
        rospy.init_node('distance_calculator', anonymous=True)

        # self.laser_sub = rospy.Subscriber(
        #     '/lds1/scan_detect',
        #     LaserScan,
        #     self.lidar_callback,
        #     queue_size=10
        # )

        self.laser_sub = rospy.Subscriber(
            '/scan',
            LaserScan,
            self.lidar_callback,
            queue_size=10
        )

        self.tf_broadcaster = tf.TransformBroadcaster()
        self.last_object_time = None
        self.timer = rospy.Timer(rospy.Duration(0.1), self.remove_transform)
        self.transforms = []  # Store transforms instead of broadcasting immediately

    def lidar_callback(self, msg):
        intensity_threshold = 140
        proximity_threshold = 0.14

        self.transforms = []  # Clear old transforms
        detected_points = []

        # Filter points by intensity and calculate transform for each point
        for i in range(len(msg.ranges)):
            if msg.intensities[i] > intensity_threshold and msg.ranges[i] < float('inf'):
                angle = msg.angle_min + i * msg.angle_increment
                x = msg.ranges[i] * math.cos(angle)
                y = msg.ranges[i] * math.sin(angle)
                detected_points.append((x, y))
                self.transforms.append((x, y))

        # Group points based on proximity
        grouped_points = []

        while detected_points:
            point = detected_points.pop(0)
            group = [point]
            for other_point in detected_points[:]:
                if self.calculate_distance(point, other_point) < proximity_threshold:
                    group.append(other_point)
                    detected_points.remove(other_point)
            grouped_points.append(group)
        quaternion = tf.transformations.quaternion_from_euler(0, 0, angle)  # (roll, pitch, yaw)
        # Broadcast averaged positions for Group 1 and Group 2
        if len(grouped_points) > 0:
            group_1 = grouped_points[0]
            x_avg1 = sum(p[0] for p in group_1) / len(group_1)
            y_avg1 = sum(p[1] for p in group_1) / len(group_1)
            self.send_transform3(x_avg1, y_avg1, "group_0", quaternion)

        if len(grouped_points) > 1:
            group_2 = grouped_points[1]
            x_avg2 = sum(p[0] for p in group_2) / len(group_2)
            y_avg2 = sum(p[1] for p in group_2) / len(group_2)
            self.send_transform3(x_avg2, y_avg2, "group_1", quaternion)

        if len(grouped_points) > 2:
            group_3 = grouped_points[2]
            x_avg3 = sum(p[0] for p in group_3) / len(group_3)
            y_avg3 = sum(p[1] for p in group_3) / len(group_3)
            self.send_transform3(x_avg3, y_avg3, "group_2", quaternion)

        if len(grouped_points) > 3:
            group_4 = grouped_points[3]
            x_avg4 = sum(p[0] for p in group_4) / len(group_4)
            y_avg4 = sum(p[1] for p in group_4) / len(group_4)
            self.send_transform3(x_avg4, y_avg4, "group_3", quaternion)

        if len(grouped_points) >= 4:
            x_avg1 = sum(p[0] for p in grouped_points[0]) / len(grouped_points[0])
            y_avg1 = sum(p[1] for p in grouped_points[0]) / len(grouped_points[0])

            x_avg2 = sum(p[0] for p in grouped_points[1]) / len(grouped_points[1])
            y_avg2 = sum(p[1] for p in grouped_points[1]) / len(grouped_points[1])

            x_avg3 = sum(p[0] for p in grouped_points[2]) / len(grouped_points[2])
            y_avg3 = sum(p[1] for p in grouped_points[2]) / len(grouped_points[2])

            x_avg4 = sum(p[0] for p in grouped_points[3]) / len(grouped_points[3])
            y_avg4 = sum(p[1] for p in grouped_points[3]) / len(grouped_points[3])

            x_centroid = (x_avg1 + x_avg2 + x_avg3 + x_avg4) / 4
            y_centroid = (y_avg1 + y_avg2 + y_avg3 + y_avg4) / 4
            angle = math.atan2(y_centroid, x_centroid)
            angle_degrees = math.degrees(angle)

            self.send_transform3(x_centroid, y_centroid, "centroid", quaternion)
            rospy.loginfo(f"Centroid coordinate: ({x_centroid}, {y_centroid})")
            distance = self.calculate_distance((x_avg4, y_avg4), (x_avg1, y_avg1))
            distance2 = self.calculate_distance((x_avg4, y_avg4), (x_avg3, y_avg3))
            r_mid = distance / 2
            if 0.60 <= distance <= 0.70:
                rospy.loginfo(f"outer_width: {distance:.2f} meters")
                rospy.loginfo(f"outer_length: {distance2:.2f} meters")
                rospy.loginfo(f"r mid {r_mid:.2f} meters")

                # Calculate midpoint and broadcast as transform
                y_midpoint = (y_avg4 + y_avg1) / 2
                x_midpoint = (x_avg4 + x_avg1) / 2
                self.send_transform3(x_midpoint, y_midpoint, "midpoint", quaternion)

                # Calculate angles A, B, C
                a = math.sqrt(((x_avg4 - 0)**2) + ((y_avg4 - 0)**2))
                b = math.sqrt(((x_avg1 - 0)**2) + ((y_avg1 - 0)**2))
                c = distance
                angle_A_deg, angle_B_deg, angle_C_deg = self.calculate_angles(a, b, c)

        self.last_object_time = rospy.get_time()

    def calculate_angles(self, a, b, c):
        # Calculate angles A, B, C in radians
        cos_value_A = (b**2 + c**2 - a**2) / (2 * b * c)
        cos_value_B = (a**2 + c**2 - b**2) / (2 * a * c)
        cos_value_C = (a**2 + b**2 - c**2) / (2 * a * b)

        # Safe acos
        angle_A_rad = self.safe_acos(cos_value_A)
        angle_B_rad = self.safe_acos(cos_value_B)
        angle_C_rad = self.safe_acos(cos_value_C)

        # Convert radians to degrees
        return math.degrees(angle_A_rad), math.degrees(angle_B_rad), math.degrees(angle_C_rad)

    def safe_acos(self, value):
        return math.acos(max(-1, min(1, value)))

    # def send_transform(self, x, y, child_frame_id):
    #     self.tf_broadcaster.sendTransform(
    #         (x, y, 0),
    #         quaternion,
    #         rospy.Time.now(),
    #         name,
    #         "lidar_merged_link"
    #     )

    def send_transform3(self, x, y, child_frame_id, quaternion):
        transform = geometry_msgs.msg.TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = 'lidar_merged_link'
        transform.child_frame_id = child_frame_id
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = 0.0

        transform.transform.rotation.x = quaternion[0]
        transform.transform.rotation.y = quaternion[1]
        transform.transform.rotation.z = quaternion[2]
        transform.transform.rotation.w = quaternion[3]

        # Use tf.TransformBroadcaster for ROS1
        self.tf_broadcaster.sendTransform(
            (transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z),
            (transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w),
            rospy.Time.now(),
            transform.child_frame_id,
            transform.header.frame_id
        )
    def calculate_distance(self, point1, point2):
        return math.dist(point1, point2)

    def remove_transform(self, event):
        if self.last_object_time and (rospy.get_time() - self.last_object_time) > 0.5:
            rospy.logwarn("Removing transform for object.")
            self.last_object_time = None


def main():
    distance_calculator = DistanceCalculator()
    rospy.spin()

if __name__ == '__main__':
    main()
