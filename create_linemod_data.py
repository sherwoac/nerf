import os
import json
import numpy as np
import imageio as iio
import tensorflow as tf

linemod_camera_intrinsics = np.array([[572.4114, 0., 325.2611], [0., 573.57043, 242.04899], [0., 0., 1.]])


def camera_angle_x_to_focal(camera_angle_x, W):
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    return focal


def focal_to_camera_angle_x(focal_x, W):
    return 2. * np.arctan2(0.5 * W, focal_x)


def transform_pose(pose: np.ndarray):
    ret_pose = np.linalg.inv(pose)
    ret_pose[:3, 3] *= -1.
    return ret_pose


class DatasetItem(object):
    def __init__(self, image_filename, pose:np.ndarray, camera_angle_x, depth_filename):
        self.image_filename = image_filename
        self.pose = np.linalg.inv(rub_to_rdf(pose))
        self.camera_angle_x = camera_angle_x
        self.depth_filename = depth_filename

    def get_record(self):
        return {"file_path": f"{self.image_filename.replace('.jpg', '')}",
                "transform_matrix": self.pose.tolist(),
                "depth_path": f'{self.depth_filename}'}


def rub_to_rdf(T):
    rub_to_rdf_transform = np.eye(4)
    rub_to_rdf_transform[1, 1] = -1
    rub_to_rdf_transform[2, 2] = -1
    return np.dot(rub_to_rdf_transform, T)


def parse_linemod_pose(path_tra, path_rot):
    t = np.loadtxt(path_tra, skiprows=1)
    r = np.loadtxt(path_rot, skiprows=1)
    assert t.size == 3
    assert r.shape == (3, 3)
    p = np.eye(4)
    p[:3,:3] = r
    p[:3,3] = 0.01 * t
    return p


def get_records(list_of_ids: list, image_dir, label_dir):
    def get_camera_angle_x():
        example_image_filename = os.path.join(image_dir, f'color{list_of_ids[0]}.jpg')
        example_image = iio.imread(example_image_filename)
        H, W = example_image.shape[:2]
        # =2 * ATAN(W / (2 * f))
        camera_angle_x = 2.*np.arctan2(W, 2.*linemod_camera_intrinsics[0, 0])
        return camera_angle_x

    camera_angle_x = get_camera_angle_x()
    dataset_records = []
    for id in list_of_ids:
        image_filename = os.path.join(image_dir, f'color{id}.jpg')
        translation_filename = os.path.join(label_dir, f'tra{id}.tra')
        rotation_filename = os.path.join(label_dir, f'rot{id}.rot')
        depth_filename = os.path.join(label_dir, f'depth{id}.dpt')
        T = parse_linemod_pose(translation_filename, rotation_filename)
        dataset_records.append(DatasetItem(image_filename, T, camera_angle_x, depth_filename))

    return dataset_records, camera_angle_x, linemod_camera_intrinsics[0, 0]


def collect_train_val_test_info(linemod_dir, cls_name):
    with open(os.path.join(linemod_dir, cls_name, 'test.txt'), 'r') as f:
        test_fns = [int(line.strip().split('/')[-1].replace('.jpg', '')) for line in f.readlines()]

    with open(os.path.join(linemod_dir, cls_name, 'train.txt'), 'r') as f:
        train_fns = [int(line.strip().split('/')[-1].replace('.jpg', '')) for line in f.readlines()]

    return test_fns, train_fns


def get_datasets(linemod_dataset_dir: str, class_name: str):
    datasets = {}
    test_fns, train_fns = collect_train_val_test_info(linemod_dataset_dir, class_name)
    datasets['test'] = test_fns
    datasets['train'] = train_fns
    return datasets


def get_datasets_ape_on_vice(linemod_dataset_dir: str, class_name: str):
    datasets = {}
    test_fns, train_fns = collect_train_val_test_info(linemod_dataset_dir, class_name)
    datasets['test'] = [29]
    datasets['train'] = list(sorted(train_fns + test_fns))[:29]
    return datasets


def get_datasets_ape_in_corner(linemod_dataset_dir: str, class_name: str):
    datasets = {}
    test_fns, train_fns = collect_train_val_test_info(linemod_dataset_dir, class_name)
    datasets['test'] = [98]
    datasets['train'] = list(sorted(train_fns + test_fns))[32:98]
    return datasets


class LinemodLoader(object):
    linemod_orig_dir = '/home/adam/shared/LINEMOD_ORIG'
    linemod_dataset_dir = '/home/adam/shared/LINEMOD'
    linemod_objects = ['driller']
    dataset_names = ['train', 'test', 'val', 'all']
    down_sample_mini = 5
    def __init__(self):
        self.output_directory = os.path.join(LinemodLoader.linemod_dataset_dir, 'nerf')

    def make_jsons(self):
        for object_name in LinemodLoader.linemod_objects:
            obj_output_directory = os.path.join(self.output_directory, object_name)
            dataset_ids = get_datasets(LinemodLoader.linemod_dataset_dir, object_name)
            for test_train_dataset_name in LinemodLoader.dataset_names:
                image_dir = os.path.join(LinemodLoader.linemod_orig_dir, object_name, 'data')
                label_dir = image_dir
                if test_train_dataset_name in ['test', 'train']:
                    ids = dataset_ids[test_train_dataset_name]
                elif test_train_dataset_name == 'val':
                    ids = dataset_ids['test']
                elif test_train_dataset_name == 'all':
                    ids = list(sorted([item for sublist in dataset_ids.values() for item in sublist]))

                dataset_records, camera_angle_x, _ = get_records(ids, image_dir, label_dir)
                json_dataset = {}
                json_dataset['camera_angle_x'] = camera_angle_x
                json_dataset['frames'] = [dataset_record.get_record() for dataset_record in dataset_records]
                if not os.path.isdir(obj_output_directory):
                    os.mkdir(obj_output_directory)

                output_json_filename = os.path.join(obj_output_directory, f'transforms_{test_train_dataset_name}.json')
                with open(output_json_filename, 'w') as outfile:
                    json.dump(json_dataset, outfile)

                print(f'written: {output_json_filename}')

    def make_mini(self, number_of_records):
        for object_name in LinemodLoader.linemod_objects:
            obj_output_directory = os.path.join(self.output_directory, object_name)
            dataset_ids = get_datasets(LinemodLoader.linemod_dataset_dir, object_name)
            for test_train_dataset_name in LinemodLoader.dataset_names:
                image_dir = os.path.join(LinemodLoader.linemod_orig_dir, object_name, 'data')
                label_dir = image_dir
                if test_train_dataset_name in ['test', 'train']:
                    ids = dataset_ids[test_train_dataset_name]
                elif test_train_dataset_name == 'val':
                    ids = dataset_ids['test']
                elif test_train_dataset_name == 'all':
                    ids = list(sorted([item for sublist in dataset_ids.values() for item in sublist]))

                dataset_records, camera_angle_x, focal_length = get_records(ids, image_dir, label_dir)

                # images = data['images']
                # poses = data['poses']
                # focal = data['focal']
                images = []
                poses = []
                for dataset_record in dataset_records[:number_of_records]:
                    images.append(iio.imread(dataset_record.image_filename))
                    poses.append(dataset_record.pose)

            images = np.stack(images, axis=0)
            H, W = images.shape[1:3]
            resized_images = tf.image.resize_images(images, [H // self.down_sample_mini, W // self.down_sample_mini]) / 255.
            output_mini_filename = os.path.join(obj_output_directory, f'mini_{object_name}.npz')
            np.savez(output_mini_filename, **{'images': resized_images.numpy().astype(np.float32), 'poses': np.stack(poses, axis=0).astype(np.float32), 'focal': focal_length / self.down_sample_mini})
            print(f'written: {output_mini_filename} len: {len(images)}')

    @staticmethod
    def load_jsons(obj_output_directory):
        test_train_val_dict = {}
        for test_train_dataset_name in LinemodLoader.dataset_names:
            json_filename = os.path.join(obj_output_directory, f'transforms_{test_train_dataset_name}.json')
            assert os.path.isfile(json_filename), f'file not found at: {json_filename}'
            with open(json_filename, 'r') as input_file:
                test_train_val_dict[test_train_dataset_name] = json.load(input_file)

        return test_train_val_dict

    def load_jsons_all_objects(self):
        object_dictionary = {}
        for object_name in LinemodLoader.linemod_objects:
            obj_output_directory = os.path.join(self.output_directory, object_name)
            test_train_val_dict = LinemodLoader.load_jsons(obj_output_directory)
            object_dictionary[object_name] = test_train_val_dict

        return object_dictionary


if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    lml = LinemodLoader()
    lml.make_jsons()
    lml.make_mini(102)
