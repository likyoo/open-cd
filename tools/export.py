# Copyright (c) Open-CD. All rights reserved.
import argparse
import logging

import torch
from mmengine import Config
from mmengine.registry import MODELS, init_default_scope

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('opencd')


def main(args):
    # must be called before using opencd
    init_default_scope('opencd')

    config_path = args.config
    checkpoint_path = args.checkpoint
    inputs = args.inputs
    model_name = args.model_name

    config = Config.fromfile(config_path, import_custom_modules=True)
    model = MODELS.build(config.model)

    ckpt = torch.load(checkpoint_path)
    state_dict = ckpt['state_dict']
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    input_shape0 = tuple(map(int, inputs[0].split(',')))
    input_shape1 = tuple(map(int, inputs[1].split(',')))
    input0 = torch.rand(input_shape0)
    input1 = torch.rand(input_shape1)
    images = torch.concat((input0, input1), dim=1)
    torch.onnx.export(
        model,
        (images),
        model_name,
        input_names=['images'],
        output_names=['output'],
        verbose=False,
        opset_version=11,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='')
    parser.add_argument('--checkpoint', '-m', type=str, default='')
    parser.add_argument(
        '--inputs',
        '-i',
        type=str,
        nargs='+',
        default=['1,3,512,512', '1,3,512,512'])
    parser.add_argument('--model-name', '-mn', type=str, default='model.onnx')
    args = parser.parse_args()
    logger.info(args)
    main(args)
