from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np


class ClusteringModule:
    def __init__(self, eps=0.05, min_samples=3, proximity_threshold=0.03):
        """
        โมดูลสำหรับจัดกลุ่มข้อมูลด้วย DBSCAN และ KNN

        Args:
            eps (float): ค่ารัศมีสำหรับ DBSCAN
            min_samples (int): จำนวนตัวอย่างขั้นต่ำสำหรับ DBSCAN
            proximity_threshold (float): ค่ารัศมีสำหรับการค้นหาเพื่อนบ้านด้วย KNN
        """
        self.eps = eps
        self.min_samples = min_samples
        self.proximity_threshold = proximity_threshold

    def dbscan_clustering(self, data):
        """
        จัดกลุ่มข้อมูลโดยใช้ DBSCAN

        Args:
            data (list of tuples): ข้อมูลจุด เช่น [(x1, y1), (x2, y2), ...]

        Returns:
            list of list of tuples: กลุ่มของจุดข้อมูลที่จัดกลุ่มแล้ว
        """
        if not data:
            return []
        data_np = np.array(data)
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(data_np)
        groups = [
            data_np[db.labels_ == label].tolist()
            for label in set(db.labels_) if label != -1
        ]
        return groups

    def knn_clustering(self, data):
        """
        จัดกลุ่มข้อมูลโดยใช้ KNN

        Args:
            data (list of tuples): ข้อมูลจุด เช่น [(x1, y1), (x2, y2), ...]

        Returns:
            list of list of tuples: กลุ่มของจุดข้อมูลที่จัดกลุ่มแล้ว
        """
        if not data:
            return []
        data_np = np.array(data)
        nbrs = NearestNeighbors(radius=self.proximity_threshold).fit(data_np)
        distances, indices = nbrs.radius_neighbors(data_np)

        groups = []
        visited = set()

        for idx, neighbors in enumerate(indices):
            if idx not in visited:
                group = [tuple(data_np[n]) for n in neighbors]
                groups.append(group)
                visited.update(neighbors)

        return groups


# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    points = [(1, 2), (1.1, 2.1), (10, 10), (10.1, 10.2), (20, 20)]

    clustering = ClusteringModule(eps=0.5, min_samples=2, proximity_threshold=0.3)

    print("DBSCAN Results:")
    dbscan_result = clustering.dbscan_clustering(points)
    for i, group in enumerate(dbscan_result):
        print(f"Group {i + 1}: {group}")

    print("\nKNN Results:")
    knn_result = clustering.knn_clustering(points)
    for i, group in enumerate(knn_result):
        print(f"Group {i + 1}: {group}")
