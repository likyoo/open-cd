# Copyright (c) Open-CD. All rights reserved.
import os.path as osp
from typing import List, Optional, Union

import mmcv
import mmengine
import numpy as np

from mmcv.transforms import Compose

from mmseg.utils import ConfigType
from mmseg.apis import MMSegInferencer

class OpenCDInferencer(MMSegInferencer):
    """Change Detection inferencer, provides inference and visualization
    interfaces. Note: MMEngine >= 0.5.0 is required.

    Args:
        classes (list, optional): Input classes for result rendering, as the
            prediction of segmentation model is a segment map with label
            indices, `classes` is a list which includes items responding to the
            label indices. If classes is not defined, visualizer will take
            `cityscapes` classes by default. Defaults to None.
        palette (list, optional): Input palette for result rendering, which is
            a list of color palette responding to the classes. If palette is
            not defined, visualizer will take `cityscapes` palette by default.
            Defaults to None.
        dataset_name (str, optional): `Dataset name or alias.
            visulizer will use the meta information of the dataset i.e. classes
            and palette, but the `classes` and `palette` have higher priority.
            Defaults to None.
        scope (str, optional): The scope of the model. Defaults to 'opencd'.
    """ # noqa

    def __init__(self,
                 classes: Optional[Union[str, List]] = None,
                 palette: Optional[Union[str, List]] = None,
                 dataset_name: Optional[str] = None,
                 scope: Optional[str] = 'opencd',
                 **kwargs) -> None:
        super().__init__(scope=scope, **kwargs)

        classes = classes if classes else self.model.dataset_meta.classes
        palette = palette if palette else self.model.dataset_meta.palette
        self.visualizer.set_dataset_meta(classes, palette, dataset_name)

    def _inputs_to_list(self, inputs: Union[str, np.ndarray]) -> list:
        """Preprocess the inputs to a list.

        Preprocess inputs to a list according to its type:

        - list or tuple: return inputs
        - str:
            - Directory path: return all files in the directory
            - other cases: return a list containing the string. The string
              could be a path to file, a url or other types of string according
              to the task.

        Args:
            inputs (InputsType): Inputs for the inferencer.

        Returns:
            list: List of input for the :meth:`preprocess`.
        """
        return list(inputs)

    def visualize(self,
                  inputs: list,
                  preds: List[dict],
                  return_vis: bool = False,
                  show: bool = False,
                  wait_time: int = 0,
                  img_out_dir: str = '',
                  opacity: float = 1.0) -> List[np.ndarray]:
        """Visualize predictions.

        Args:
            inputs (list): Inputs preprocessed by :meth:`_inputs_to_list`.
            preds (Any): Predictions of the model.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            img_out_dir (str): Output directory of rendering prediction i.e.
                color segmentation mask. Defaults: ''
            opacity (int, float): The transparency of segmentation mask.
                Defaults to 0.8.

        Returns:
            List[np.ndarray]: Visualization results.
        """
        if not show and img_out_dir == '' and not return_vis:
            return None
        if self.visualizer is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None.')

        self.visualizer.alpha = opacity

        results = []

        for single_inputs, pred in zip(inputs, preds):
            img_from_to = []
            for single_input in single_inputs:
                if isinstance(single_input, str):
                    img_bytes = mmengine.fileio.get(single_input)
                    img = mmcv.imfrombytes(img_bytes)
                    img = img[:, :, ::-1]
                    img_name = osp.basename(single_input)
                elif isinstance(single_input, np.ndarray):
                    img = single_input.copy()
                    img_num = str(self.num_visualized_imgs).zfill(8) + '_vis'
                    img_name = f'{img_num}.jpg'
                else:
                    raise ValueError('Unsupported input type:'
                                    f'{type(single_input)}')
                img_shape = img.shape
                img_from_to.append(img)

            out_file = osp.join(img_out_dir, img_name) if img_out_dir != ''\
                else None

            img_zero_board = np.zeros(img_shape)
            self.visualizer.add_datasample(
                img_name,
                img_zero_board,
                img_from_to,
                pred,
                show=show,
                wait_time=wait_time,
                draw_gt=False,
                draw_pred=True,
                out_file=out_file)
            if return_vis:
                results.append(self.visualizer.get_image())
            self.num_visualized_imgs += 1

        return results if return_vis else None

    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        """Initialize the test pipeline.

        Return a pipeline to handle various input data, such as ``str``,
        ``np.ndarray``. It is an abstract method in BaseInferencer, and should
        be implemented in subclasses.

        The returned pipeline will be used to process a single data.
        It will be used in :meth:`preprocess` like this:

        .. code-block:: python
            def preprocess(self, inputs, batch_size, **kwargs):
                ...
                dataset = map(self.pipeline, dataset)
                ...
        """
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline
        # Loading annotations is also not applicable
        for transform in ('MultiImgLoadAnnotations', 'MultiImgLoadDepthAnnotation'):
            idx = self._get_transform_idx(pipeline_cfg, transform)
            if idx != -1:
                del pipeline_cfg[idx]

        load_img_idx = self._get_transform_idx(pipeline_cfg,
                                               'MultiImgLoadImageFromFile')
        if load_img_idx == -1:
            raise ValueError(
                'MultiImgLoadImageFromFile is not found in the test pipeline')
        pipeline_cfg[load_img_idx]['type'] = 'MultiImgLoadInferencerLoader'
        return Compose(pipeline_cfg)
