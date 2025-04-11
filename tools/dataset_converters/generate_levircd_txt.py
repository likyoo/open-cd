# Copyright (c) Open-CD. All rights reserved.
import argparse
import os
import os.path as osp


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate .txt file for LEVIR-CD dataset')
    parser.add_argument('dataset_path', help='path of LEVIR-CD dataset')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def generate_txt_from_dir(src_dir, dst_dir, split):
    """Generate .txt file for LEVIR-CD dataset.

    Args:
        src_dir (str): path of the source dataset.
        dst_dir (str): Path to save .txt file.
        split (str): sub_dirs. 'train', 'val' or 'test'

    """
    src_dir = osp.join(src_dir, split)
    sub_dir_1 = osp.join(src_dir, 'A')
    sub_dir_2 = osp.join(src_dir, 'B')
    ann_dir = osp.join(src_dir, 'label')

    file_list = []
    for img_name in sorted(os.listdir(ann_dir)):
        assert osp.exists(osp.join(sub_dir_1, img_name)) and \
               osp.exists(osp.join(sub_dir_2, img_name)), \
            f'{img_name} is not in {sub_dir_1} or {sub_dir_2}'

        file_list.append([
            os.path.splitext(img_name)[0]
        ])

    with open('{}.txt'.format(osp.join(dst_dir, split)), 'w') as f:
        for item in file_list:
            f.write(' '.join(item) + '\n')


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join('data', 'levir_cd')
    else:
        out_dir = args.out_dir

    print('Making .txt files ...')
    generate_txt_from_dir(dataset_path, out_dir, 'train')
    generate_txt_from_dir(dataset_path, out_dir, 'val')
    generate_txt_from_dir(dataset_path, out_dir, 'test')

    print('Done!')


if __name__ == '__main__':
    main()


