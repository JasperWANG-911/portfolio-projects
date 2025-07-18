# High-Resolution Aerial Image Object Detection with YOLO

## Project Overview

This project implements an efficient object detection pipeline for high-resolution aerial imagery (PDF orthomosaics) using YOLO v8 and v11. The key challenge addressed is processing extremely large aerial images that exceed typical memory constraints by implementing a tile-based detection approach.

### Key Features
- **PDF to Image Conversion**: Handles high-DPI orthomosaic PDFs (1200 DPI)
- **Data Augmentation Pipeline**: 5 different augmentation strategies for robust training
- **Model Comparison**: Benchmarks YOLOv8 vs YOLOv11 performance
- **Instance Segmentation**: Uses YOLO segment models for precise object boundaries

## Technologies Used
- **Data Annotation**: Roboflow
- **Deep Learning**: YOLOv8, YOLOv11 (Ultralytics)
- **Image Processing**: OpenCV, Pillow, PyMuPDF (fitz)
- **Data Augmentation**: Albumentations
