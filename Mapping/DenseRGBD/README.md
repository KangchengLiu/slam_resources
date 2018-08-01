# Dense RGBD
-----
## 文件说明
﻿color: 按顺序存放RGB图像
depth: 按顺序存放对应D深度图像
pose.txt: 序列图像的相机位姿Twc/格式为平移向量加旋转四元数[x,y,z,qx,qy,qz,qw]

## 查看地图
* Octomap: `octovis octomap.bt`
* PCD: `pcl_viewer map.pcd`
