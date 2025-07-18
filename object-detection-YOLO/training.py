from ultralytics import YOLO
import os

# Create folders for results
output_folder_v8, output_folder_v11 = "output_results_v8", "output_folder_v11"
os.makedirs(output_folder_v8, exist_ok=True)
os.makedirs(output_folder_v11, exist_ok=True)

# Load YOLOv8 and YOLOv11
model_v8 = YOLO('v8.1/detect/train/weights/best.pt')
model_v11 = YOLO('v11.1/weights/best.pt')

file_count = len(os.listdir('test_images'))

# objective detection with YOLOv8
for i in range(1, file_count):

    file_path = os.path.join('test_images', f'image{i}.png')
    result_v8 = model_v8(file_path)

    # save results to output_folder
    for j, pred in enumerate(result_v8):
        save_path = os.path.join(output_folder_v8, f"result_v8_{i}.png")
        pred.plot()
        pred.save(save_path)
        print(f"Saved YOLOv8 prediction to {save_path}")

# objective detection with YOLOv11
for i in range(1, file_count):

    # objective detection
    file_path = os.path.join('test_images', f'image{i}.png')
    result_v11 = model_v11(file_path)

    # save results to output_folder
    for j, pred in enumerate(result_v11):
        save_path = os.path.join(output_folder_v11, f"result_v11_{i}.png")
        pred.plot()
        pred.save(save_path)
        print(f"Saved YOLOv11 prediction to {save_path}")
