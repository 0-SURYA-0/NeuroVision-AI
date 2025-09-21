"""
3D Post-processing Pipeline for Brain Tumor Segmentation

Advanced post-processing techniques for refining segmentation results including
morphological operations, connected component analysis, and uncertainty-based refinement.

Key Features:
- Morphological operations (opening, closing, hole filling)
- Connected component analysis and filtering
- Surface smoothing and regularization
- Uncertainty-based post-processing
- Multi-class segmentation refinement
- Volume-based filtering
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from scipy import ndimage
from scipy.ndimage import gaussian_filter, median_filter
from skimage import measure, morphology
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.filters import gaussian
import warnings
from nibabel.nifti1 import Nifti1Image
from nibabel.loadsave import save


@dataclass
class PostProcessConfig:
    """Configuration for post-processing operations"""
    min_component_size: int = 100
    hole_fill_area: int = 50
    gaussian_sigma: float = 0.5
    median_kernel_size: int = 3
    morphology_kernel_size: int = 3
    confidence_threshold: float = 0.5
    uncertainty_threshold: float = 0.3


class PostProcessor3D:
    """
    3D post-processing pipeline for segmentation refinement
    
    Args:
        config: Post-processing configuration
        spacing: Voxel spacing in mm (x, y, z)
    """
    
    def __init__(self, config: Optional[PostProcessConfig] = None, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.config = config or PostProcessConfig()
        self.spacing = spacing
        
    def remove_small_components(self, mask: np.ndarray, min_size: Optional[int] = None) -> np.ndarray:
        """
        Remove small connected components
        
        Args:
            mask: Binary segmentation mask
            min_size: Minimum component size (voxels)
            
        Returns:
            Filtered mask
        """
        min_size = min_size or self.config.min_component_size
        
        # Label connected components
        labeled = measure.label(mask.astype(bool))
        
        # Remove small objects
        cleaned = remove_small_objects(labeled, min_size=min_size)
        
        return (cleaned > 0).astype(mask.dtype)
    
    def fill_holes(self, mask: np.ndarray, max_hole_size: Optional[int] = None) -> np.ndarray:
        """
        Fill holes in segmentation masks
        
        Args:
            mask: Binary segmentation mask
            max_hole_size: Maximum hole size to fill (voxels)
            
        Returns:
            Hole-filled mask
        """
        max_hole_size = max_hole_size or self.config.hole_fill_area
        
        # Fill holes slice by slice for memory efficiency
        filled = np.copy(mask)
        
        for i in range(mask.shape[0]):
            slice_2d = mask[i]
            if np.any(slice_2d):
                filled_slice = remove_small_holes(slice_2d.astype(bool), area_threshold=max_hole_size)
                filled[i] = filled_slice.astype(mask.dtype)
        
        return filled
    
    def morphological_operations(self, mask: np.ndarray, operation: str = 'closing') -> np.ndarray:
        """
        Apply morphological operations
        
        Args:
            mask: Binary segmentation mask
            operation: Type of operation ('opening', 'closing', 'dilation', 'erosion')
            
        Returns:
            Processed mask
        """
        kernel_size = self.config.morphology_kernel_size
        kernel = morphology.ball(kernel_size)
        
        if operation == 'opening':
            result = morphology.opening(mask, kernel)
        elif operation == 'closing':
            result = morphology.closing(mask, kernel)
        elif operation == 'dilation':
            result = morphology.dilation(mask, kernel)
        elif operation == 'erosion':
            result = morphology.erosion(mask, kernel)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        return result.astype(mask.dtype)
    
    def smooth_surface(self, mask: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        """
        Smooth segmentation surface using Gaussian filtering
        
        Args:
            mask: Binary segmentation mask
            sigma: Gaussian kernel standard deviation
            
        Returns:
            Smoothed mask
        """
        sigma = sigma or self.config.gaussian_sigma
        
        # Convert to float for smoothing
        mask_float = mask.astype(np.float32)
        
        # Apply Gaussian smoothing
        smoothed = gaussian_filter(mask_float, sigma=sigma)
        
        # Ensure 'smoothed' is compatible with the operator
        smoothed_array = np.asarray(smoothed, dtype=np.float32)
        return (smoothed_array > 0.5).astype(np.uint8)  # Ensure binary output
    
    def largest_component_only(self, mask: np.ndarray) -> np.ndarray:
        """
        Keep only the largest connected component
        
        Args:
            mask: Binary segmentation mask
            
        Returns:
            Mask with only largest component
        """
        labeled = measure.label(mask.astype(bool))
        
        if labeled.max() == 0:
            return mask
        
        # Find largest component
        component_sizes = np.bincount(labeled.ravel())
        component_sizes[0] = 0  # Ignore background
        largest_component = np.argmax(component_sizes)
        
        # Keep only largest component
        result = (labeled == largest_component).astype(mask.dtype)
        
        return result
    
    def uncertainty_guided_refinement(self, prediction: np.ndarray, uncertainty: np.ndarray, 
                                      confidence_threshold: Optional[float] = None, 
                                      uncertainty_threshold: Optional[float] = None) -> np.ndarray:
        """
        Refine predictions using uncertainty information
        
        Args:
            prediction: Raw prediction probabilities
            uncertainty: Uncertainty map
            confidence_threshold: Threshold for high-confidence predictions
            uncertainty_threshold: Threshold for low-uncertainty regions
            
        Returns:
            Refined binary mask
        """
        conf_thresh = confidence_threshold or self.config.confidence_threshold
        unc_thresh = uncertainty_threshold or self.config.uncertainty_threshold
        
        # High confidence predictions
        high_conf = prediction > conf_thresh
        
        # Low uncertainty regions
        low_unc = uncertainty < unc_thresh
        
        # Combine conditions
        refined = high_conf & low_unc
        
        return refined.astype(np.uint8)
    
    def multi_class_refinement(self, predictions: np.ndarray, class_hierarchy: Optional[List[int]] = None) -> np.ndarray:
        """
        Refine multi-class segmentation ensuring proper hierarchy
        
        Args:
            predictions: Multi-class predictions (C, D, H, W)
            class_hierarchy: Class hierarchy order (e.g., [0, 1, 2, 4] for BraTS)
            
        Returns:
            Refined multi-class mask
        """
        if class_hierarchy is None:
            class_hierarchy = list(range(predictions.shape[0]))
        
        refined = np.zeros_like(predictions[0], dtype=np.uint8)
        
        # Process classes in hierarchy order
        for i, class_id in enumerate(class_hierarchy):
            if i == 0:  # Background
                continue
                
            class_mask = predictions[i] > 0.5
            
            # Apply post-processing to each class
            class_mask = self.remove_small_components(class_mask)
            class_mask = self.fill_holes(class_mask)
            class_mask = self.morphological_operations(class_mask, 'closing')
            
            # Assign class label
            refined[class_mask] = class_id
        
        return refined
    
    def pipeline_standard(self, mask: np.ndarray) -> np.ndarray:
        """
        Standard post-processing pipeline
        
        Args:
            mask: Binary segmentation mask
            
        Returns:
            Post-processed mask
        """
        # Step 1: Remove small components
        processed = self.remove_small_components(mask)
        
        # Step 2: Fill holes
        processed = self.fill_holes(processed)
        
        # Step 3: Morphological closing
        processed = self.morphological_operations(processed, 'closing')
        
        # Step 4: Keep largest component only
        processed = self.largest_component_only(processed)
        
        # Step 5: Surface smoothing
        processed = self.smooth_surface(processed)
        
        return processed
    
    def pipeline_conservative(self, mask: np.ndarray) -> np.ndarray:
        """
        Conservative post-processing (minimal changes)
        
        Args:
            mask: Binary segmentation mask
            
        Returns:
            Post-processed mask
        """
        # Only remove very small components and fill small holes
        processed = self.remove_small_components(mask, min_size=50)
        processed = self.fill_holes(processed, max_hole_size=25)
        
        return processed
    
    def pipeline_aggressive(self, mask: np.ndarray) -> np.ndarray:
        """
        Aggressive post-processing (extensive refinement)
        
        Args:
            mask: Binary segmentation mask
            
        Returns:
            Post-processed mask
        """
        # Step 1: Morphological opening (remove noise)
        processed = self.morphological_operations(mask, 'opening')
        
        # Step 2: Remove small components
        processed = self.remove_small_components(processed, min_size=200)
        
        # Step 3: Fill holes
        processed = self.fill_holes(processed)
        
        # Step 4: Morphological closing
        processed = self.morphological_operations(processed, 'closing')
        
        # Step 5: Keep largest component
        processed = self.largest_component_only(processed)
        
        # Step 6: Surface smoothing
        processed = self.smooth_surface(processed, sigma=1.0)
        
        return processed
    
    def crop_and_save_tumor_region(self, mask: np.ndarray, original_scan: np.ndarray, affine: np.ndarray, output_path: str):
        """
        Crop the tumor region from the original scan based on the segmentation mask and save as .nii.gz.

        Args:
            mask: Binary segmentation mask.
            original_scan: Original 3D scan.
            affine: Affine transformation matrix of the scan.
            output_path: Path to save the cropped .nii.gz file.
        """
        # Find bounding box of the tumor region
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            raise ValueError("No tumor region found in the mask.")

        bbox_min = [np.min(coords[i]) for i in range(3)]
        bbox_max = [np.max(coords[i]) for i in range(3)]

        # Crop the mask and the original scan
        cropped_mask = mask[bbox_min[0]:bbox_max[0]+1, bbox_min[1]:bbox_max[1]+1, bbox_min[2]:bbox_max[2]+1]
        cropped_scan = original_scan[bbox_min[0]:bbox_max[0]+1, bbox_min[1]:bbox_max[1]+1, bbox_min[2]:bbox_max[2]+1]

        # Adjust affine for cropping
        new_affine = affine.copy()
        new_affine[:3, 3] += np.dot(affine[:3, :3], bbox_min)

        # Save the cropped region as .nii.gz with updated affine
        save(Nifti1Image(cropped_scan, new_affine), output_path)
        print(f"Cropped tumor region saved to {output_path}")

        # Save the cropped mask as .nii.gz
        cropped_mask_path = output_path.replace('.nii.gz', '_mask.nii.gz')
        save(Nifti1Image(cropped_mask, new_affine), cropped_mask_path)
        print(f"Cropped tumor mask saved to {cropped_mask_path}")


def post_process_batch(predictions: torch.Tensor, method: str = 'standard', 
                      config: Optional[PostProcessConfig] = None) -> torch.Tensor:
    """
    Post-process a batch of predictions
    
    Args:
        predictions: Batch of predictions (B, C, D, H, W)
        method: Post-processing method ('standard', 'conservative', 'aggressive')
        config: Post-processing configuration
        
    Returns:
        Post-processed predictions
    """
    processor = PostProcessor3D(config or PostProcessConfig())
    
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions_np = torch.sigmoid(predictions).cpu().numpy()
        return_tensor = True
    else:
        predictions_np = predictions
        return_tensor = False
    
    batch_size = predictions_np.shape[0]
    processed_batch = np.zeros_like(predictions_np)
    
    for i in range(batch_size):
        for c in range(predictions_np.shape[1]):
            mask = (predictions_np[i, c] > 0.5).astype(np.uint8)
            
            if method == 'standard':
                processed = processor.pipeline_standard(mask)
            elif method == 'conservative':
                processed = processor.pipeline_conservative(mask)
            elif method == 'aggressive':
                processed = processor.pipeline_aggressive(mask)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            processed_batch[i, c] = processed
    
    # Ensure processed_batch is returned as a tensor when required
    if return_tensor:
        return torch.from_numpy(processed_batch).float()
    else:
        return torch.tensor(processed_batch)


def calculate_volume_stats(mask: np.ndarray, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict[str, Union[float, List[float]]]:
    """
    Calculate volume statistics for a segmentation mask
    
    Args:
        mask: Binary segmentation mask
        spacing: Voxel spacing in mm
        
    Returns:
        Dictionary with volume statistics
    """
    voxel_volume = np.prod(spacing)  # mm³ per voxel
    
    total_voxels = np.sum(mask > 0)
    total_volume_mm3 = total_voxels * voxel_volume
    
    # Find bounding box
    coords = np.where(mask > 0)
    if len(coords[0]) > 0:
        bbox_min = [np.min(coords[i]) for i in range(3)]
        bbox_max = [np.max(coords[i]) for i in range(3)]
        bbox_size = [(bbox_max[i] - bbox_min[i] + 1) * spacing[i] for i in range(3)]
    else:
        bbox_size = [0, 0, 0]
    
    bbox_size = [float(size) for size in bbox_size]  # Ensure bbox_size is List[float]
    return {
        'volume_voxels': int(total_voxels),
        'volume_mm3': float(total_volume_mm3),
        'volume_cm3': float(total_volume_mm3 / 1000.0),
        'bbox_size_mm': bbox_size,
        'bbox_volume_mm3': float(np.prod(bbox_size))
    }