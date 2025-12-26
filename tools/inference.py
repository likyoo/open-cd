import os
import os.path as osp
import shutil
import time
import cv2
import numpy as np
from opencd.apis import OpenCDInferencer

def get_input_size(checkpoint_path):
    input_size_option = [256, 512]
    for input_size in input_size_option:
        if str(input_size) in checkpoint_path:
            return input_size
    return 512


def get_all_checkpoints(checkpoints_dir='checkpoints'):
    """
    获取checkpoints目录下所有的checkpoint文件
    
    Args:
        checkpoints_dir (str): checkpoints目录路径，默认为'checkpoints'
    
    Returns:
        list: checkpoint文件路径列表
    """
    if not osp.exists(checkpoints_dir):
        return []
    
    checkpoints = []
    for file in os.listdir(checkpoints_dir):
        if file.endswith('.pth'):
            checkpoints.append(osp.join(checkpoints_dir, file))
    
    return sorted(checkpoints)


def get_config_from_checkpoint(checkpoint_path, configs_dir='configs'):
    """
    根据checkpoint文件名获取对应的config文件路径
    
    Args:
        checkpoint_path (str): checkpoint文件路径，例如 'checkpoints/lightcdnet_l_256x256_40k_levircd.pth'
        configs_dir (str): configs目录路径，默认为'configs'
    
    Returns:
        str: config文件路径，如果找不到则返回None
    """
    # 提取checkpoint文件名（去掉路径和扩展名）
    checkpoint_name = osp.basename(checkpoint_path)
    base_name = osp.splitext(checkpoint_name)[0]  # 去掉.pth扩展名
    
    # 提取模型类型（第一个下划线前的部分）
    if '_' in base_name:
        model_type = base_name.split('_')[0]
    else:
        model_type = base_name
    
    # 首先在对应的模型目录下查找
    model_config_dir = osp.join(configs_dir, model_type)
    config_path = osp.join(model_config_dir, f"{base_name}.py")
    
    if osp.exists(config_path):
        return config_path
    
    # 如果找不到，在整个configs目录下递归搜索
    for root, dirs, files in os.walk(configs_dir):
        for file in files:
            if file == f"{base_name}.py":   
                return osp.join(root, file)
    
    return None

def main():

    image_pair_list = [
        ['datas/1_1.png', 'datas/1_2.png'],
        ['datas/1_2.png', 'datas/1_1.png'],
        ['datas/2_1.png', 'datas/2_2.png'],
        ['datas/2_2.png', 'datas/2_1.png'],
    ]
    output_path = 'output'
       
    checkpoints = get_all_checkpoints()
    for checkpoint in checkpoints:
        input_size = get_input_size(checkpoint)
        config_path = get_config_from_checkpoint(checkpoint)
        if config_path:
            print(f"Checkpoint: {checkpoint}, Input Size: {input_size}, Config: {config_path}")
        else:
            print(f"Checkpoint: {checkpoint}, Input Size: {input_size}, Config: Not found")
            continue
        try:
            inferencer = OpenCDInferencer(model=config_path, weights=checkpoint, classes=('unchanged', 'changed'), palette=[[0, 0, 0], [255, 255, 255]])
        except Exception as e:
            print(f"Checkpoint: {checkpoint}, Input Size: {input_size}, Config: {config_path}, Error: {e}")
            continue
        inferencer(image_pair_list, show=False, out_dir=output_path)

        # 将ouput_path 移动 到 output_path 的子目录中
        output_sub_path = osp.join(output_path, osp.basename(config_path))
        os.makedirs(output_sub_path, exist_ok=True)
        shutil.move(output_path + '/vis', output_sub_path)

        for image_pair in image_pair_list:
            out_put_name = osp.basename(image_pair[1])
            img1 = cv2.imread(image_pair[0])
            img2 = cv2.imread(image_pair[1])
            mask = cv2.imread(osp.join(output_sub_path + '/vis', out_put_name))

            img2_res = img2.copy()
            img2_res[mask != 255] = (img2_res[mask != 255] * 0.3).astype(np.uint8)

            result = cv2.hconcat([img1, img2_res, img2])
            result = cv2.resize(result, (int(result.shape[1] * 0.5), int(result.shape[0] * 0.5)))

            result_name = osp.splitext(out_put_name)[0] + '_result.jpg'
            cv2.imwrite(osp.join(output_sub_path, result_name), result)

if __name__ == '__main__':
    main()
