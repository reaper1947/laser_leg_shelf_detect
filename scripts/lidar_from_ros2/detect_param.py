import rospy
from sensor_msgs.msg import LaserScan
import numpy as np
import math

class ObjectDetector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)
        self.subscription = rospy.Subscriber(
            'scan', 
            LaserScan, 
            self.scan_callback
        )
        rospy.spin()

    def scan_callback(self, msg):
        print("NUMBER ", len(msg.ranges))
        print("Min ", min(msg.ranges))
        filtered_ranges = [distance for distance in msg.ranges if distance < float('inf')]
        print("Max ", max(filtered_ranges) if filtered_ranges else "No valid ranges")
        print("min_angle", msg.angle_min)
        print("max_angle", msg.angle_max)
        print("angle_increment", msg.angle_increment)
        print("time_increment", msg.time_increment)
        print("intensities", max(msg.intensities))
        self.calculate_sq(msg)
        print(" ")

    def calculate_sq(self, msg):
        filtered_ranges = [distance for distance in msg.ranges if distance < float('inf')]

        if len(filtered_ranges) < 4:
            self.last_object_time = None
            return

        sorted_filtered_ranges = sorted(filtered_ranges)[:4]  # Get the closest 4 points
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        # Find the index for each distance in the original ranges to get angles
        min_indices = [msg.ranges.index(dist) for dist in sorted_filtered_ranges]

        num_elements = len(filtered_ranges)  # number of index
        rospy.loginfo(f"The number of elements in Index: {num_elements}")

        # Calculate x, y coordinates for each of the closest points
        points = []
        for i, dist in zip(min_indices, sorted_filtered_ranges):
            angle = angle_min + i * angle_increment
            x = dist * math.cos(angle)
            y = dist * math.sin(angle)
            points.append((x, y))

        # Calculate width and length using the closest two distances
        w = sorted_filtered_ranges[0]
        l = sorted_filtered_ranges[1]

        # Example log for the points
        rospy.loginfo(f"x1, y1: {points[0]}")
        rospy.loginfo(f"x2, y2: {points[1]}")
        rospy.loginfo(f"x3, y3: {points[2]}")
        rospy.loginfo(f"x4, y4: {points[3]}")

def main():
    object_detector = ObjectDetector()
    # Keep the node running
    rospy.spin()

if __name__ == '__main__':
    main()
