# Copyright (c) Open-CD. All rights reserved.
# TODO: correctness verification
import argparse
import os
import json
import os.path as osp

import cv2
import numpy as np
from tqdm import tqdm
import pycocotools.mask as mutils


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert segmentation mask to coco instance seg result.')
    parser.add_argument('-i', '--in_dir', help='path of segmemtation mask dir')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def seg2instance(pred, img_id, instance_res):
    """convert segmentation mask to coco instance seg result. 
    Args:
        pred (_type_)
        img_id (int)
        instance_res (list)

    Returns:
        _type_: _description_
    """
    mask = pred.round().astype(np.uint8)
    nc, label = cv2.connectedComponents(mask, connectivity=8)
    for c in range(nc):
        if np.all(mask[label == c] == 0):
            continue
        else:
            ann = np.asfortranarray((label == c).astype(np.uint8))
            if np.sum(ann == 1) < 8:
                continue
            rle = mutils.encode(ann)
            bbox = [int(_) for _ in mutils.toBbox(rle)]
            area = int(mutils.area(rle))
            score = float(pred[label == c].mean())
            instance_res.append({
                "segmentation": {
                    "size": [int(_) for _ in rle["size"]], 
                    "counts": rle["counts"].decode()},
                "bbox": [int(_) for _ in bbox], "area": int(area), "iscrowd": 0, "category_id": 1,
                "image_id": int(img_id), "id": len(instance_res),
                "score": float(score)
            })
    # return instance_res


def main():
    args = parse_args()

    print('Converting files ...')
    os.makedirs(args.out_dir, exist_ok=True)
    instance_res = []
    
    for img_idx, img_name in enumerate(tqdm(os.listdir(args.in_dir))):
        img_id = int(os.path.basename(img_name).split(".")[0])
        img_path = osp.join(args.in_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) // 255
        seg2instance(img, img_id, instance_res)

    with open(osp.join(args.out_dir, "test.segm.json"), "w") as f:
        json.dump(instance_res, f, indent=2)
    os.system(f"zip -9 -r {args.out_dir}.zip {args.out_dir}")

    print('Done!')


if __name__ == '__main__':
    main()
