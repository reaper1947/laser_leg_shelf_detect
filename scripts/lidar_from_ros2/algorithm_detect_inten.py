import math
import time


class DistanceCalculator:
    def __init__(self):
        self.last_object_time = None
        self.results = None
        self.detected_points = None

        self.grouped_points = []

    def get_grouped_points(self):
        return self.grouped_points

    def lidar_callback(self, msg):
        intensity_threshold = 140
        proximity_threshold = 0.03
        min_distance = 0.60
        max_distance = 0.69
        offset_length = 0.62
        offset_standby = 1.0

        detected_points = self.filter_points(msg, intensity_threshold)
        grouped_points = self.group_points(detected_points, proximity_threshold)

        if len(grouped_points) >= 2:
            x_avg1 = sum(p[0] for p in grouped_points[0]) / len(grouped_points[0])
            y_avg1 = sum(p[1] for p in grouped_points[0]) / len(grouped_points[0])

            x_avg2 = sum(p[0] for p in grouped_points[1]) / len(grouped_points[1])
            y_avg2 = sum(p[1] for p in grouped_points[1]) / len(grouped_points[1])

            distance = self.calculate_distance((x_avg1, y_avg1), (x_avg2, y_avg2))
            if min_distance <= distance <= max_distance:
                x_midpoint = (x_avg1 + x_avg2) / 2
                y_midpoint = (y_avg1 + y_avg2) / 2
                angle = math.atan2(y_midpoint, x_midpoint)
                angle2 = math.atan2(y_avg2 - y_avg1, x_avg2 - x_avg1)
                angle_degrees = math.degrees(angle)
                perpendicular_angle = angle2 + math.pi / 2
                offset_x = offset_length * math.cos(perpendicular_angle)
                offset_y = offset_length * math.sin(perpendicular_angle)

                x_outer1 = x_avg1 + offset_x
                y_outer1 = y_avg1 + offset_y
                x_outer2 = x_avg2 + offset_x
                y_outer2 = y_avg2 + offset_y

                x_centroid = (x_avg1 + x_avg2 + x_outer1 + x_outer2) / 4
                y_centroid = (y_avg1 + y_avg2 + y_outer1 + y_outer2) / 4
                outer_width = self.calculate_distance((x_avg2, y_avg2), (x_avg1, y_avg1))
                width = self.calculate_distance((x_outer1, y_outer1), (x_outer2, y_outer2))
                self.results = (angle, (x_midpoint, y_midpoint),
                                (x_centroid, y_centroid), outer_width, width, (x_outer1, y_outer1), (x_outer2, y_outer2))
                self.detected_points = detected_points
                self.grouped_points = grouped_points
        else:
            self.results = None
            self.detected_points = None
            self.grouped_points = None

    def get_calculated_results(self):
        return self.results

    def get_calculated_detected_points(self):
        return self.detected_points

    def get_calculated_grouped_points(self):
        return self.grouped_points

    def filter_points(self, msg, intensity_threshold):
        detected_points = []
        for i in range(len(msg['ranges'])):
            if msg['intensities'][i] > intensity_threshold and msg['ranges'][i] < float('inf'):
                angle = msg['angle_min'] + i * msg['angle_increment']
                x = msg['ranges'][i] * math.cos(angle)
                y = msg['ranges'][i] * math.sin(angle)
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

    def calculate_distance(self, point1, point2):
        return math.dist(point1, point2)
