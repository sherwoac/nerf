import os
import re
import pickle
import numpy as np

import run_nerf as rn
import run_nerf_helpers as rnh
import load_llff # load_llff_data
import load_blender


if __name__ == '__main__':
    # load_nerf given config
    parser = rn.config_parser()
    args = parser.parse_args()
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_tests = load_llff.load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        masks = rn.load_masks(args.datadir, args.mask_directory)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split, extras = load_blender.load_blender_data(
            args.datadir,
            args.half_res,
            args.testskip,
            args.image_extn,
            mask_directory=args.mask_directory,
            get_depths=args.get_depth_maps,
            image_field=args.image_fieldname,
            image_dir_override=args.image_dir_override)

        i_train, i_val, i_tests = i_split

pcds = []
# test examples x batches x (loss+pcd)

results = np.zeros(shape=(len(i_tests)), dtype=np.float32)
for i, i_test in enumerate(i_tests):
    filename = extras['filenames'][2][i]
    frame_number = int(re.findall(r'\d+', filename)[-1])
    output_filename = os.path.join(args.output_directory, f'results_dict_{str(frame_number).zfill(4)}.pkl')
    if not os.path.isfile(output_filename):
        print(f'missing results at: {output_filename}')
        # exit(1)
        break

    input = open(output_filename, 'rb')
    overall_results = pickle.load(input)
    # results[i, :, 0] = overall_results[filename]['img_loss_min']
    results[i] = overall_results[filename]['pcd'][-1]
    input.close()

# pcds = np.stack(pcds)
print(f'ADD measure: {np.sum(results < 0.1 * 0.259425) / results.shape[0]}')
