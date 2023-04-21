import os
import os.path as osp

import cv2
import numpy as np
from mmengine.registry import VISBACKENDS
from mmengine.visualization.vis_backend import LocalVisBackend, force_init_env


@VISBACKENDS.register_module()
class CDLocalVisBackend(LocalVisBackend):

    @force_init_env
    def add_image(self,
                  name: str,
                  image: np.array,
                  image_from: np.array = None,
                  image_to: np.array = None,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image to disk.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        assert image.dtype == np.uint8
        drawn_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        os.makedirs(self._img_save_dir, exist_ok=True)
        save_file_name = f'{name}.png'

        if image_from is not None and image_to is not None:
            assert image_from.dtype == np.uint8 and image_to.dtype == np.uint8
            drawn_image_from = cv2.cvtColor(image_from, cv2.COLOR_RGB2BGR)
            drawn_image_to = cv2.cvtColor(image_to, cv2.COLOR_RGB2BGR)
            for sub_dir in ['binary', 'from', 'to']:
                os.makedirs(osp.join(self._img_save_dir, sub_dir), exist_ok=True)

            cv2.imwrite(osp.join(self._img_save_dir, 'binary', save_file_name), drawn_image)
            cv2.imwrite(osp.join(self._img_save_dir, 'from', save_file_name), drawn_image_from)
            cv2.imwrite(osp.join(self._img_save_dir, 'to', save_file_name), drawn_image_to)
        else:       
            cv2.imwrite(osp.join(self._img_save_dir, save_file_name), drawn_image)