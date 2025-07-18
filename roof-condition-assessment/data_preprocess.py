import os
import cv2
import numpy as np
import torch
from PIL import Image
import glob
import logging
from typing import List, Tuple, Dict
import json
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageSplitter:
    """Splits large drone images into smaller grid patches"""
    
    def __init__(self, grid_size: Tuple[int, int] = (8, 6), skip_borders: bool = True):
        self.grid_size = grid_size
        self.skip_borders = skip_borders
    
    def split_image(self, image_path: str, output_dir: str) -> List[str]:
        """
        Split a single image into grid patches
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save patches
            
        Returns:
            List of saved patch file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_patches = []
        
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                slice_width = width // self.grid_size[0]
                slice_height = height // self.grid_size[1]
                
                patch_idx = 0
                rows = self.grid_size[1] - 1 if self.skip_borders else self.grid_size[1]
                cols = self.grid_size[0] - 1 if self.skip_borders else self.grid_size[0]
                
                for y in range(rows):
                    for x in range(cols):
                        left = x * slice_width
                        upper = y * slice_height
                        right = left + slice_width
                        lower = upper + slice_height
                        
                        patch = img.crop((left, upper, right, lower))
                        
                        base_name = os.path.splitext(os.path.basename(image_path))[0]
                        patch_path = os.path.join(output_dir, f"{base_name}_patch_{patch_idx:03d}.jpg")
                        patch.save(patch_path, "JPEG", quality=95)
                        saved_patches.append(patch_path)
                        patch_idx += 1
                
                logger.info(f"Split {image_path} into {len(saved_patches)} patches")
                
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            
        return saved_patches
    
    def process_folder(self, input_folder: str, output_folder: str) -> Dict[str, List[str]]:
        """Process all images in a folder"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG']
        all_images = []
        
        for ext in image_extensions:
            all_images.extend(glob.glob(os.path.join(input_folder, ext)))
        
        if not all_images:
            logger.warning(f"No images found in {input_folder}")
            return {}
        
        logger.info(f"Found {len(all_images)} images to process")
        
        results = {}
        for img_path in all_images:
            patches = self.split_image(img_path, output_folder)
            results[img_path] = patches
            
        return results


class RoofDetector:
    """Detects and segments roof regions using SAM and AOD API"""
    
    def __init__(self, sam_checkpoint_path: str, api_key: str = None):
        self.sam_checkpoint_path = sam_checkpoint_path
        self.api_key = api_key or os.getenv('AOD_API_KEY')
        self.api_url = os.getenv('AOD_API_URL', 'https://api.va.landing.ai/v1/tools/agentic-object-detection')
        
        if not self.api_key:
            logger.warning("AOD API key not found. Object detection will be disabled.")
    
    def detect_roofs(self, image_path: str) -> Dict:
        """Detect roofs using AOD API"""
        if not self.api_key:
            logger.error("API key not configured")
            return {"data": [[]]}
        
        headers = {"Authorization": f"Basic {self.api_key}"}
        
        try:
            with open(image_path, "rb") as image_file:
                files = {"image": image_file}
                data = {"prompts": "roof", "model": "agentic"}
                response = requests.post(self.api_url, files=files, data=data, headers=headers)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error calling AOD API: {e}")
            return {"data": [[]]}
    
    def segment_roof(self, image_path: str, center_point: List[int], 
                     output_dir: str, roof_id: str) -> Tuple[str, str]:
        """
        Segment roof region using SAM
        
        Returns:
            Tuple of (colored_roof_path, binary_mask_path)
        """
        try:
            # Import SAM components
            from segment_anything import sam_model_registry, SamPredictor
            
            # Load image
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Initialize SAM
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            sam = sam_model_registry['vit_h'](checkpoint=self.sam_checkpoint_path)
            sam.to(device=device)
            predictor = SamPredictor(sam)
            predictor.set_image(image_rgb)
            
            # Generate mask
            input_point = np.array([center_point])
            input_label = np.array([1])
            
            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )
            
            # Select best mask
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            
            # Save colored roof
            masked_image = image_rgb.copy()
            masked_image[~best_mask] = 0
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Save colored roof image
            roof_path = os.path.join(output_dir, f"{base_name}_roof_{roof_id}.png")
            cv2.imwrite(roof_path, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
            
            # Save binary mask
            mask_path = os.path.join(output_dir, f"{base_name}_mask_{roof_id}.png")
            cv2.imwrite(mask_path, (best_mask.astype(np.uint8) * 255))
            
            logger.info(f"Segmented roof {roof_id} from {image_path}")
            
            return roof_path, mask_path
            
        except Exception as e:
            logger.error(f"Error segmenting roof: {e}")
            return None, None
    
    def process_image(self, image_path: str, output_dir: str) -> List[Tuple[str, str]]:
        """Process single image: detect and segment all roofs"""
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        # Detect roofs
        detection_results = self.detect_roofs(image_path)
        detections = detection_results.get('data', [[]])[0]
        
        # Segment each detected roof
        for idx, detection in enumerate(detections):
            bbox = detection['bounding_box']
            center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
            
            roof_path, mask_path = self.segment_roof(
                image_path, center, output_dir, str(idx)
            )
            
            if roof_path and mask_path:
                results.append((roof_path, mask_path))
        
        return results


def main():
    """Main preprocessing pipeline"""
    # Configuration
    config = {
        "input_folder": "./data/uncropped_images",
        "output_folder": "./data/processed",
        "sam_checkpoint": "./models/sam_vit_h_4b8939.pth",
        "grid_size": (8, 6),
        "skip_borders": True
    }
    
    # Create output directories
    patches_dir = os.path.join(config["output_folder"], "patches")
    roofs_dir = os.path.join(config["output_folder"], "roofs")
    masks_dir = os.path.join(config["output_folder"], "masks")
    
    for dir_path in [patches_dir, roofs_dir, masks_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Step 1: Split images into patches
    logger.info("Step 1: Splitting images into patches...")
    splitter = ImageSplitter(
        grid_size=config["grid_size"],
        skip_borders=config["skip_borders"]
    )
    patch_results = splitter.process_folder(config["input_folder"], patches_dir)
    
    # Step 2: Detect and segment roofs in patches
    logger.info("Step 2: Detecting and segmenting roofs...")
    detector = RoofDetector(config["sam_checkpoint"])
    
    all_patches = []
    for patches in patch_results.values():
        all_patches.extend(patches)
    
    roof_results = []
    for patch_path in all_patches:
        results = detector.process_image(patch_path, roofs_dir)
        roof_results.extend(results)
    
    logger.info(f"Processing complete! Generated {len(all_patches)} patches and {len(roof_results)} roof segments.")
    
    # Save processing summary
    summary = {
        "total_images": len(patch_results),
        "total_patches": len(all_patches),
        "total_roofs": len(roof_results),
        "config": config
    }
    
    with open(os.path.join(config["output_folder"], "processing_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()