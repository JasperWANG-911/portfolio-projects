from ultralytics import YOLO
import os
import numpy as np

# Create folders for results
output_folder_v11 = "output_folder_v11"
os.makedirs(output_folder_v11, exist_ok=True)


model_v11 = YOLO('train/weights/best.pt')

file_count = len(os.listdir('test images'))

# objective detection with YOLOv11
for i in range(1, file_count):

    # objective detection
    file_path = os.path.join('test images', f'image{i}.png')
    result_v11 = model_v11(file_path)

    # save results to output_folder
    for j, pred in enumerate(result_v11):

        boxes = pred.boxes
        scores = boxes.conf.cpu().numpy()
        valid_indices = np.where(scores > 0.5)[0]  # Only keep those with confidence > 0.5(this number can be modified)

        if len(valid_indices) > 0:
            save_path = os.path.join(output_folder_v11, f"result_v11_{i}.png")
            pred.plot()
            pred.save(save_path)