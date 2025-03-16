import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import os
import requests

# Use AOD API to detect roofs, and return bbox in json
def call_AOD(image_path):
    AOD_url = "https://api.va.landing.ai/v1/tools/agentic-object-detection"
    headers = {"Authorization": "Basic ejRkbG43a2RsaGxndnF5ZWdpbGp1Om55S1ZoWVMydlJ6QkJxSGp5Z2plQ3ZkeG42a1RmNVhi"}
    
    with open(image_path, "rb") as image_file:
        files = {"image": image_file}
        data = {"prompts": "roof", "model": "agentic"}
        response = requests.post(AOD_url, files=files, data=data, headers=headers)
    
    return response.json()

# Load image
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Load SAM2
def load_sam_model(checkpoint_path, model_type='vit_h'):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device='cuda' if torch.cuda.is_available() else 'cpu')
    return sam

# Extract and save roof region (black background + original roof color)
def extract_and_save_roof(image, mask, output_path):
    # Create a copy of the image
    masked_image = image.copy()
    
    # Convert mask to boolean
    bool_mask = mask.astype(bool)
    
    # Set non-roof pixels to black (0)
    masked_image[~bool_mask] = 0
    
    # Convert back to BGR for saving
    masked_image_bgr = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
    
    # Save the image
    cv2.imwrite(output_path, masked_image_bgr)
    print(f"Saved roof region (color) to {output_path}")

# Save a black-white (binary) mask (white=255 for roof, black=0 for background)
def save_binary_mask(mask, output_path):
    # Sam's mask might be float/ bool in [0,1]. We convert it to 0/255 uint8
    mask_255 = (mask.astype(np.uint8) * 255)
    cv2.imwrite(output_path, mask_255)
    print(f"Saved binary mask to {output_path}")

# Segment and save roof area
def segment_and_save_roof(image_path, checkpoint_path, input_coord, bbox_id, output_dir="output"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename without extension
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Load image
    image = load_image(image_path)
    
    # Initialize SAM model
    sam = load_sam_model(checkpoint_path, model_type='vit_h')
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    
    # Set input point for segmentation
    input_point = np.array([input_coord])
    input_label = np.array([1])  # 1 indicates a foreground point
    
    # Generate mask predictions
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )
    
    # Select the best mask (highest score)
    best_mask_idx = np.argmax(scores)
    best_mask = masks[best_mask_idx]
    
    # save an image with roof + black surrounding
    region_output_path = os.path.join(output_dir, f"{base_filename}_roof_{bbox_id}.png")
    extract_and_save_roof(image, best_mask, region_output_path)
    
    # save corresponding mask
    mask_output_path = os.path.join(output_dir, f"{base_filename}_mask_{bbox_id}.png")
    save_binary_mask(best_mask, mask_output_path)
    
    return region_output_path, mask_output_path

# Get center coordinate for bbox
def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) // 2, (y1 + y2) // 2]

# Process multiple bboxes in json
def process_roof_json(json_data, image_path, checkpoint_path, output_dir="output"):
    # Get all detections
    detections = json_data['data'][0]
    
    # Process each detection
    for idx, detection in enumerate(detections):
        # Get bounding box
        bbox = detection['bounding_box']
        
        # Calculate center point
        center = get_center(bbox)
        
        # Use index as bbox ID
        bbox_id = str(idx)
        
        print(f"Processing roof #{idx} at {center}")
        
        # Segment and save the roof
        segment_and_save_roof(
            image_path, 
            checkpoint_path, 
            center, 
            bbox_id, 
            output_dir
        )

# Process all images in a folder
def process_folder(folder_path, checkpoint_path, output_dir="output"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files from the folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG']
    image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1] in image_extensions]
    
    print(f"Found {len(image_files)} images in folder: {folder_path}")
    
    # Process each image
    for i, img_file in enumerate(image_files):
        # Get full path to image
        img_path = os.path.join(folder_path, img_file)
        print(f"Processing image {i+1}/{len(image_files)}: {img_file}")
        
        try:
            # Call AOD API to detect roofs
            roof_json = call_AOD(img_path)
            
            # Process detected roofs
            process_roof_json(roof_json, img_path, checkpoint_path, output_dir)
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    print(f"Completed processing all images. Results saved to {output_dir}")

if __name__ == "__main__":
    checkpoint_path = "../ROOF_RANKING/sam_vit_h_4b8939.pth"

    # Process all images in folder "raw_data"
    folder_path = "../ROOF_RANKING/raw_data"
    process_folder(folder_path, checkpoint_path, "output_roof_masks")
