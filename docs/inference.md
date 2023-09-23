# Inference with existing models

Open-CD provides pre-trained models for change detection in the corresponding presentation files, and supports multiple standard datasets, including LEVIR-CD, S2Looking, etc.
This note will show how to use existing models to inference on given images.

Open-CD provides several interfaces for users to easily use pre-trained models for inference.

- [Inference with existing models](#inference-with-existing-models)
  - [Inferencer](#inferencer)
    - [Basic Usage](#basic-usage)
    - [Initialization](#initialization)
    - [Visualize prediction](#visualize-prediction)
    - [List model](#list-model)

## Inferencer

We provide the most **convenient** way to use the model in Open-CD `OpenCDInferencer`. You can get change mask for an image with only 3 lines of code.

### Basic Usage

The following example shows how to use `OpenCDInferencer` to perform inference on a single image pair.

```
>>> from opencd.apis import OpenCDInferencer
>>> # Load models into memory
>>> inferencer = OpenCDInferencer(model='changer_ex_r18_512x512_40k_levircd.py', weights='ChangerEx_r18-512x512_40k_levircd_20221223_120511.pth', classes=('unchanged', 'changed'), palette=[[0, 0, 0], [255, 255, 255]])
>>> # Inference
>>> inferencer([['demo_A.png', 'demo_B.png']], show=False, out_dir='OUTPUT_PATH')
```

Moreover, you can use `OpenCDInferencer` to process a list of images:

```
# Input a list of images
>>> images = [[image1_A, image1_B], [image2_A, image2_B], ...] # image1_A can be a file path or a np.ndarray
>>> inferencer(images, show=True, wait_time=0.5) # wait_time is delay time, and 0 means forever

# Save visualized rendering color maps and predicted results
# out_dir is the directory to save the output results, img_out_dir and pred_out_dir are subdirectories of out_dir
# to save visualized rendering color maps and predicted results
>>> inferencer(images, out_dir='outputs', img_out_dir='vis', pred_out_dir='pred')
```

