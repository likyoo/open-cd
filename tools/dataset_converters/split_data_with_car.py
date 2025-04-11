# Copyright (c) Open-CD. All rights reserved.
"""
Split the data according to Change Area Ratio.

JL1-CD: A New Benchmark for Remote Sensing Change Detection and a Robust Multi-Teacher Knowledge Distillation Framework.
arXiv: https://arxiv.org/pdf/2502.13407
"""
import os
from PIL import Image
import shutil
import numpy as np


def calculate_change_ratio(label_image_path):
    # Open label image and convert to grayscale
    label_image = Image.open(label_image_path).convert('L')
    
    # Convert image to numpy array
    label_array = np.array(label_image)
    
    # Calculate change ratio (255=changed, 0=unchanged)
    total_pixels = label_array.size
    change_pixels = np.sum(label_array == 255)  # Count pixels with value 255
    change_ratio = change_pixels / total_pixels  # Calculate change ratio
    
    return change_ratio

def create_output_directories(base_dir):
    # Create A, B, and label subdirectories in the base directory
    os.makedirs(os.path.join(base_dir, 'A'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'B'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'label'), exist_ok=True)

def copy_files_to_directory(image_name, source_dir, target_dir):
    # Copy image from A folder
    shutil.copy(os.path.join(source_dir, 'A', image_name), os.path.join(target_dir, 'A', image_name))
    # Copy image from B folder
    shutil.copy(os.path.join(source_dir, 'B', image_name), os.path.join(target_dir, 'B', image_name))
    # Copy image from label folder
    shutil.copy(os.path.join(source_dir, 'label', image_name), os.path.join(target_dir, 'label', image_name))

def partition_images(source_dir, s_dir, m_dir, l_dir, threshold1=0.05, threshold2=0.2):
    # Get all image filenames from the label directory
    label_images = sorted(os.listdir(os.path.join(source_dir, 'label')))
    
    for image_name in label_images:
        label_image_path = os.path.join(source_dir, 'label', image_name)
        
        # Calculate change ratio for the label image
        change_ratio = calculate_change_ratio(label_image_path)
        
        # Partition images into s, m, l based on change ratio
        if change_ratio <= threshold1:
            target_dir = s_dir
        elif threshold1 < change_ratio <= threshold2:
            target_dir = m_dir
        else:
            target_dir = l_dir
        
        # Copy A, B, label files to target directory
        copy_files_to_directory(image_name, source_dir, target_dir)

def process_dataset(base_dir, dataset_type):
    # Define paths for the dataset type (train/val/test)
    source_dir = os.path.join(base_dir, dataset_type)
    s_dir = os.path.join(base_dir, f'{dataset_type}_s')
    m_dir = os.path.join(base_dir, f'{dataset_type}_m')
    l_dir = os.path.join(base_dir, f'{dataset_type}_l')
    
    # Create output directories
    create_output_directories(s_dir)
    create_output_directories(m_dir)
    create_output_directories(l_dir)

    # Partition images based on change ratio
    partition_images(source_dir, s_dir, m_dir, l_dir)

def main():
    # Base directory path
    base_dir = 'data/JL1-CD'
    
    # Process train, val, and test sets
    for dataset_type in ['train', 'val', 'test']:
        process_dataset(base_dir, dataset_type)

if __name__ == '__main__':
    main()