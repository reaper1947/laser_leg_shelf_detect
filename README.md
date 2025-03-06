# laser_leg_shelf_detect
use Lidar2D for detect shelf by intensity and use movebase to move your robot to centroid of shelf and true precision angle by calculate.

## HOW TO USE
```bash
ros launch laser_leg_shelf_detect lidar_inten.launch
```

## algorithms
- DBSCAN
- KNN (optional)

## features
- detect 4 leg, pub Marker each leg and pub TF centroid with quaternion.
- detect 2 leg, detect front leg of shelf then pub offset leg with length param then pub Marker and pub TF centroid same detect 4 leg.
- parameter can set on rqt_reconfigure and change or save on code lidar_best.py
- moveBase can use when centroid has pub and you can send state true on /tick  by topic pub.

## lidar
- RPLIDAR LPX-T1
