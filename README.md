# Brain Tumor Detection

A deep learning project for 3D brain tumor segmentation using MRI scans from the BraTS (Brain Tumor Segmentation) dataset.

## Overview

This project implements a convolutional neural network for detecting and segmenting brain tumors from medical imaging data. The model processes 3D MRI scans to identify tumor regions, enabling automated analysis of brain tumor characteristics.

## Features

- **3D Brain Tumor Segmentation**: Automated detection and segmentation of brain tumors from MRI scans
- **Web Interface**: Flask-based application for easy interaction and visualization
- **Multiple MRI Modalities**: Support for FLAIR, T1, T1ce, and T2 imaging sequences
- **Real-time Predictions**: Upload MRI scans and receive instant segmentation results
- **Visualization**: Generate visual outputs showing detected tumor regions

## Dataset

The project uses the **BraTS (Brain Tumor Segmentation) 2020** dataset, which includes:
- Multi-institutional pre-operative MRI scans
- Multiple imaging modalities (FLAIR, T1w, T1Gd, T2w)
- Expert-annotated tumor segmentations
- Training and validation sets

## Project Structure

```
Brain_Tumor_Detection/
├── app.py                              # Flask web application
├── brats-3d-brain-tumour-segmentation.ipynb  # Training notebook
├── brain_tumor_model.h5                # Trained model (HDF5 format)
├── brain_tumor_model.keras             # Trained model (Keras format)
├── model.png                           # Model architecture visualization
├── uploads/                            # Directory for uploaded files
├── output/                             # Directory for prediction results
├── BraTS20_Validation_005_mask.nii     # Sample mask file
└── test_gif_BraTS20_Training_001_flair.nii  # Sample test file
```

## Requirements

```
tensorflow
keras
numpy
nibabel
flask
matplotlib
scikit-learn
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Shreya-2829/Brain_Tumor_Detection.git
cd Brain_Tumor_Detection
```

2. Install required dependencies:
```bash
pip install tensorflow keras numpy nibabel flask matplotlib scikit-learn
```

3. Download the BraTS dataset (if training from scratch)

## Usage

### Running the Web Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Upload an MRI scan (.nii format) through the web interface

4. View the segmentation results

### Training the Model

Open and run the Jupyter notebook:
```bash
jupyter notebook brats-3d-brain-tumour-segmentation.ipynb
```

The notebook contains the complete pipeline for:
- Data preprocessing
- Model architecture definition
- Training process
- Evaluation metrics
- Visualization of results

## Model Architecture

The model uses a 3D U-Net architecture optimized for medical image segmentation:
- Encoder-decoder structure with skip connections
- 3D convolutional layers for volumetric data processing
- Batch normalization for training stability
- Dropout layers for regularization

See `model.png` for a detailed visualization of the architecture.

## Input Format

The model accepts NIfTI (.nii) format files, which are standard for medical imaging data. Each input should be a 3D MRI scan with shape compatible with the trained model.

## Output

The model outputs:
- Segmented tumor masks highlighting different tumor regions
- Visualization of predictions overlaid on original scans
- Results saved in the `output/` directory

## Performance Metrics

The model is evaluated using:
- Dice Coefficient
- IoU (Intersection over Union)
- Precision and Recall
- Hausdorff Distance

## File Formats

- **Input**: NIfTI (.nii) files containing 3D MRI scans
- **Output**: Segmentation masks in NIfTI format and visualization images

## Future Improvements

- [ ] Add support for additional MRI modalities
- [ ] Implement ensemble methods for improved accuracy
- [ ] Optimize model for real-time inference
- [ ] Add more comprehensive evaluation metrics
- [ ] Improve web interface with better visualization tools
- [ ] Add support for batch processing

