import rospy
from sensor_msgs.msg import LaserScan

class ObjectDetection:
    
    def __init__(self):
        rospy.init_node('laser_obstacle', anonymous=True)
        self.laser_sub = rospy.Subscriber(
            '/scan',
            LaserScan,
            self.laser_callback,
            queue_size=10
        )
        self.scan_msg = None

    def laser_callback(self, msg):
        self.scan_msg = msg
        self.process_msg(self.scan_msg)

    def process_msg(self, msg):
        range_msg = min(msg.ranges)  # Adjust for indexing from -30 to 30 degrees

        # If the distance to the obstacle is less than 2.0 meters, it’s detected.
        if range_msg < 1.0:
            rospy.loginfo("Obstacle detected")


def main():
    object_detection = ObjectDetection()
    # Keep the node running
    rospy.spin()

if __name__ == '__main__':
    main()
