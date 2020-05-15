import os
import json
import numpy as np
import imageio as iio

linemod_camera_intrinsics = np.array([[572.4114, 0., 325.2611], [0., 573.57043, 242.04899], [0., 0., 1.]])


def camera_angle_x_to_focal(camera_angle_x, W):
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    return focal


def focal_to_camera_angle_x(focal_x, W):
    return 2. * np.arctan2(0.5 * W, focal_x)


class DatasetItem(object):
    def __init__(self, image_filename, pose:np.ndarray, camera_angle_x):
        self.image_filename = image_filename
        self.pose = pose
        self.camera_angle_x = camera_angle_x

    def get_record(self):
        return {"file_path": f"{self.image_filename.replace('.jpg', '')}", "transform_matrix":self.pose.tolist()}

def parse_linemod_pose(path_tra, path_rot):
    t = np.loadtxt(path_tra, skiprows=1)
    r = np.loadtxt(path_rot, skiprows=1)
    assert t.size == 3
    assert r.shape == (3,3)
    p = np.eye(4)
    p[:3,:3] = r
    p[:3,3] = 0.01 * t
    return p


def get_records(list_of_ids: list, image_dir, label_dir):
    def get_camera_angle_x():
        example_image_filename = os.path.join(image_dir, f'color{list_of_ids[0]}.jpg')
        example_image = iio.imread(example_image_filename)
        H, W = example_image.shape[:2]
        camera_angle_x = linemod_camera_intrinsics[0, 0] / W
        return camera_angle_x

    camera_angle_x = get_camera_angle_x()
    dataset_records = []
    for id in list_of_ids:
        image_filename = os.path.join(image_dir, f'color{id}.jpg')
        translation_filename = os.path.join(label_dir, f'tra{id}.tra')
        rotation_filename = os.path.join(label_dir, f'rot{id}.rot')
        pose = parse_linemod_pose(translation_filename, rotation_filename)
        dataset_records.append(DatasetItem(image_filename, pose, camera_angle_x))

    return dataset_records, camera_angle_x


def collect_train_val_test_info(linemod_dir, cls_name):
    with open(os.path.join(linemod_dir,cls_name,'test.txt'),'r') as f:
        test_fns=[int(line.strip().split('/')[-1].replace('.jpg', '')) for line in f.readlines()]

    with open(os.path.join(linemod_dir,cls_name,'train.txt'),'r') as f:
        train_fns=[int(line.strip().split('/')[-1].replace('.jpg', '')) for line in f.readlines()]

    return test_fns, train_fns


def get_datasets(linemod_dataset_dir: str, class_name: str):
    datasets = {}
    test_fns, train_fns = collect_train_val_test_info(linemod_dataset_dir, class_name)
    datasets['test'] = test_fns
    datasets['train'] = train_fns
    return datasets


if __name__ == '__main__':
    linemod_orig_dir = '/home/adam/shared/LINEMOD_ORIG'
    linemod_dataset_dir = '/home/adam/shared/LINEMOD'
    output_directory = os.path.join(linemod_dataset_dir, 'nerf')
    linemod_objects = ['driller']
    dataset_names = ['train', 'test', 'val']
    for object_name in linemod_objects:
        obj_output_directory = os.path.join(output_directory, object_name)
        dataset_ids = get_datasets(linemod_dataset_dir, object_name)
        for test_train_dataset_name in dataset_names:
            image_dir = os.path.join(linemod_orig_dir, object_name, 'data')
            label_dir = image_dir
            if test_train_dataset_name in ['test', 'train']:
                ids = dataset_ids[test_train_dataset_name]
            elif test_train_dataset_name == 'val':
                ids = dataset_ids['test'][::50]
            dataset_records, camera_angle_x = get_records(ids, image_dir, label_dir)
            json_dataset = {}
            json_dataset['camera_angle_x'] = camera_angle_x
            json_dataset['frames'] = [dataset_record.get_record() for dataset_record in dataset_records]
            if not os.path.isdir(obj_output_directory):
                os.mkdir(obj_output_directory)

            with open(os.path.join(obj_output_directory, f'transforms_{test_train_dataset_name}.json'), 'w') as outfile:
                json.dump(json_dataset, outfile)
