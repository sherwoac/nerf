import os
import copy
import numpy as np
import open3d as o3d


class PointCloudDistance(object):
    def __init__(self, point_cloud_filename, distance_factor=1., transformation_function=None):
        assert os.path.exists(point_cloud_filename), f'point cloud file not found at: {point_cloud_filename}'
        self.object_point_cloud = o3d.io.read_point_cloud(point_cloud_filename)
        self.distance_factor = distance_factor
        self.transformation_function = transformation_function

    @staticmethod
    def get_distance_between_objects(source, target):
        return np.linalg.norm(np.asarray(source.points) - np.asarray(target.points), axis=1)

    def get_distance_to_transformation(self, gt_transformation, inferred_transformation):
        if self.transformation_function is not None:
            gt_transformation = self.transformation_function(gt_transformation)
            inferred_transformation = self.transformation_function(inferred_transformation)

        object_point_cloud1 = copy.deepcopy(self.object_point_cloud)
        object_point_cloud1 = object_point_cloud1.transform(gt_transformation)
        object_point_cloud2 = copy.deepcopy(self.object_point_cloud)
        object_point_cloud2 = object_point_cloud2.transform(inferred_transformation)
        distances = PointCloudDistance.get_distance_between_objects(object_point_cloud1, object_point_cloud2) * self.distance_factor
        return distances.mean()
