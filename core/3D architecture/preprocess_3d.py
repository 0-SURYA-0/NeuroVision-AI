"""
Advanced 3D Preprocessing for Brain Tumor Segmentation

Comprehensive preprocessing pipeline including skull stripping, bias field correction,
intensity normalization, resampling, and volume preparation for deep learning models.

Key Features:
- Multi-modal intensity normalization
- Skull stripping and brain extraction
- Bias field correction
- Isotropic resampling
- Robust cropping and padding
- Quality assessment
- Volume statistics
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
from scipy import ndimage
from scipy.ndimage import zoom, gaussian_filter, binary_erosion, binary_dilation
from scipy.ndimage import binary_closing, binary_opening, label
from skimage import measure, morphology
from skimage.filters import threshold_otsu
from skimage.restoration import denoise_nl_means
from skimage.transform import resize

import torch
import torch.nn.functional as F

from utils_3d import Config, load_nifti, save_nifti, timer, format_bytes


@dataclass
class VolumeStats:
    """Statistics for a 3D volume"""
    shape: Tuple[int, int, int]
    voxel_spacing: Tuple[float, float, float]
    volume_mm3: float
    intensity_range: Tuple[float, float]
    mean_intensity: float
    std_intensity: float
    percentiles: Dict[int, float]
    non_zero_voxels: int
    tumor_volume_mm3: Optional[float] = None
    tumor_percentage: Optional[float] = None


class VolumePreprocessor:
    """
    Advanced 3D volume preprocessing pipeline
    
    Args:
        config: Configuration object
        verbose: Enable verbose logging
    """
    
    def __init__(self, config: Optional[Config] = None, verbose: bool = True):
        self.config = config or Config()
        self.verbose = verbose
        
    def preprocess_case(
        self,
        modalities: Dict[str, Union[str, Path, np.ndarray]],
        segmentation: Optional[Union[str, Path, np.ndarray]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        case_id: str = "case"
    ) -> Dict[str, Any]:
        """
        Preprocess a complete BraTS case
        
        Args:
            modalities: Dictionary of modality names to file paths or arrays
            segmentation: Segmentation file path or array
            output_dir: Directory to save processed files
            case_id: Case identifier
            
        Returns:
            Dictionary with processed data and statistics
        """
        if self.verbose:
            print(f"Preprocessing case: {case_id}")
        
        result = {
            'case_id': case_id,
            'modalities': {},
            'segmentation': None,
            'stats': {},
            'success': False
        }
        
        try:
            # Load all modalities
            loaded_modalities = {}
            reference_img = None
            
            for mod_name, mod_path in modalities.items():
                if isinstance(mod_path, (str, Path)):
                    data, nii_img = load_nifti(mod_path)
                    if reference_img is None:
                        reference_img = nii_img
                else:
                    data = mod_path
                
                loaded_modalities[mod_name] = data
                if self.verbose:
                    print(f"  Loaded {mod_name}: {data.shape}")
            
            # Load segmentation if provided
            seg_data = None
            if segmentation is not None:
                if isinstance(segmentation, (str, Path)):
                    seg_data, _ = load_nifti(segmentation)
                else:
                    seg_data = segmentation
                
                if self.verbose:
                    print(f"  Loaded segmentation: {seg_data.shape}")
            
            # Get reference spacing
            if reference_img is not None:
                voxel_spacing = self._get_voxel_spacing(reference_img)
            else:
                voxel_spacing = (1.0, 1.0, 1.0)
            
            # Preprocess each modality
            processed_modalities = {}
            
            for mod_name, data in loaded_modalities.items():
                processed_data = self._preprocess_modality(
                    data, mod_name, voxel_spacing
                )
                processed_modalities[mod_name] = processed_data
                
                # Calculate statistics
                stats = self._calculate_volume_stats(
                    processed_data, voxel_spacing
                )
                result['stats'][mod_name] = stats
            
            # Preprocess segmentation
            if seg_data is not None:
                processed_seg = self._preprocess_segmentation(
                    seg_data, voxel_spacing
                )
                result['segmentation'] = processed_seg
                
                # Calculate tumor statistics
                tumor_stats = self._calculate_tumor_stats(
                    processed_seg, voxel_spacing
                )
                result['stats']['tumor'] = tumor_stats
            
            # Ensure consistent shapes and alignment
            processed_modalities, processed_seg = self._ensure_consistency(
                processed_modalities, seg_data
            )
            
            result['modalities'] = processed_modalities
            result['segmentation'] = processed_seg
            
            # Save processed data if output directory provided
            if output_dir:
                self._save_processed_case(
                    result, output_dir, reference_img
                )
            
            result['success'] = True
            
            if self.verbose:
                print(f"  Successfully preprocessed case: {case_id}")
                
        except Exception as e:
            if self.verbose:
                print(f"  Error preprocessing case {case_id}: {e}")
            result['error'] = str(e)
        
        return result
    
    def _preprocess_modality(
        self,
        data: np.ndarray,
        modality_name: str,
        voxel_spacing: Tuple[float, float, float],
        skip_cropping: bool = False
    ) -> np.ndarray:
        """
        Preprocess a single modality
        
        Args:
            data: Input volume data
            modality_name: Name of the modality (t1, t1ce, t2, flair)
            voxel_spacing: Voxel spacing (mm)
            
        Returns:
            Preprocessed volume
        """
        if self.verbose:
            print(f"    Processing {modality_name}...")
        
        # 1. Remove NaN and infinite values
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. Bias field correction (simplified)
        if modality_name in ['t1', 't1ce']:
            data = self._bias_field_correction(data)
        
        # 3. Skull stripping (create brain mask)
        brain_mask = self._create_brain_mask(data)
        
        # 4. Apply brain mask
        data = data * brain_mask
        
        # 5. Intensity normalization
        data = self._normalize_intensity(data, modality_name)
        
        # 6. Resampling to isotropic spacing
        if self.config.resample_spacing:
            data = self._resample_volume(
                data, voxel_spacing, self.config.resample_spacing
            )
        
        # 7. Crop to brain region (skip during inference to keep coordinates)
        if not skip_cropping:
            data = self._crop_to_brain(data, brain_mask)
        
        # 8. Resize to target shape
        data = self._resize_volume(data, self.config.input_size)
        
        # 9. Final cleanup
        data = self._final_cleanup(data)
        
        return data.astype(np.float32)
    
    def _preprocess_segmentation(
        self,
        seg_data: np.ndarray,
        voxel_spacing: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        Preprocess segmentation mask
        
        Args:
            seg_data: Input segmentation data
            voxel_spacing: Voxel spacing (mm)
            
        Returns:
            Preprocessed segmentation
        """
        if self.verbose:
            print("    Processing segmentation...")
        
        # Convert multi-class to binary (any tumor vs background)
        binary_seg = (seg_data > 0).astype(np.float32)
        
        # Remove small isolated components
        binary_seg = self._remove_small_components(binary_seg, min_size=100)
        
        # Morphological operations to clean up mask
        binary_seg = self._morphological_cleanup(binary_seg)
        
        # Resampling if needed
        if self.config.resample_spacing:
            binary_seg = self._resample_volume(
                binary_seg, voxel_spacing, self.config.resample_spacing, order=0
            )
        
        # Resize to target shape
        binary_seg = self._resize_volume(
            binary_seg, self.config.input_size, order=0
        )
        
        # Ensure binary
        binary_seg = (binary_seg > 0.5).astype(np.float32)
        
        return binary_seg
    
    def _bias_field_correction(self, data: np.ndarray) -> np.ndarray:
        """
        Simple bias field correction using morphological operations
        
        Args:
            data: Input volume
            
        Returns:
            Bias-corrected volume
        """
        # Create a smooth version representing the bias field
        smooth_data = gaussian_filter(data, sigma=3)
        
        # Avoid division by zero
        smooth_data[smooth_data == 0] = 1
        
        # Correct bias
        corrected = data / smooth_data
        
        # Rescale to original intensity range
        if np.max(data) > 0:
            corrected = corrected * (np.max(data) / np.max(corrected))
        
        return corrected
    
    def _create_brain_mask(self, data: np.ndarray) -> np.ndarray:
        """
        Create brain mask using Otsu thresholding and morphological operations
        
        Args:
            data: Input volume
            
        Returns:
            Binary brain mask
        """
        # Otsu thresholding on non-zero voxels
        non_zero_data = data[data > 0]
        if len(non_zero_data) == 0:
            return np.ones_like(data, dtype=np.float32)
        
        try:
            threshold = threshold_otsu(non_zero_data)
        except Exception:
            threshold = np.percentile(non_zero_data, 25)
        
        # Create initial mask
        brain_mask = data > threshold
        
        # Fill holes
        brain_mask = ndimage.binary_fill_holes(brain_mask)
        
        # Remove small components
        from scipy.ndimage import generate_binary_structure
        structure = generate_binary_structure(3, 1)  
        labeled_mask, num_features = label(brain_mask, structure=structure)
        if num_features > 1:
            indices = np.arange(1, num_features + 1)
            component_sizes = ndimage.sum(
                brain_mask, labeled_mask, index=indices
            )
            largest_component = indices[component_sizes.argmax()]
            brain_mask = labeled_mask == largest_component
        
        # Morphological operations
        brain_mask = binary_closing(brain_mask, structure=np.ones((3, 3, 3)))
        brain_mask = binary_opening(brain_mask, structure=np.ones((2, 2, 2)))
        
        return brain_mask.astype(np.float32)
    
    def _normalize_intensity(self, data: np.ndarray, modality_name: str) -> np.ndarray:
        """
        Normalize intensity values
        
        Args:
            data: Input volume
            modality_name: Modality name for specific normalization
            
        Returns:
            Normalized volume
        """
        # Get non-zero brain voxels
        brain_voxels = data[data > 0]
        
        if len(brain_voxels) == 0:
            return data
        
        if self.config.normalize_method == "zscore":
            mean_val = np.mean(brain_voxels)
            std_val = np.std(brain_voxels)
            
            if std_val > 0:
                data = (data - mean_val) / std_val
            else:
                data = data - mean_val
                
        elif self.config.normalize_method == "minmax":
            min_val = np.min(brain_voxels)
            max_val = np.max(brain_voxels)
            
            if max_val > min_val:
                data = (data - min_val) / (max_val - min_val)
                
        elif self.config.normalize_method == "percentile":
            low, high = self.config.clip_percentiles
            p_low, p_high = np.percentile(brain_voxels, [low, high])
            
            data = np.clip(data, p_low, p_high)
            if p_high > p_low:
                data = (data - p_low) / (p_high - p_low)
        
        elif self.config.normalize_method == "nyul":
            # Nyul histogram standardization (simplified)
            data = self._nyul_normalization(data, modality_name)
        
        return data
    
    def _nyul_normalization(self, data: np.ndarray, modality_name: str) -> np.ndarray:
        """
        Simplified Nyul histogram standardization
        
        Args:
            data: Input volume
            modality_name: Modality name
            
        Returns:
            Normalized volume
        """
        brain_voxels = data[data > 0]
        
        if len(brain_voxels) == 0:
            return data
        
        # Define standard landmarks (percentiles)
        landmarks = [1, 10, 25, 50, 75, 90, 99]
        
        # Calculate current landmarks
        current_landmarks = np.percentile(brain_voxels, landmarks)
        
        # Define target landmarks (modality-specific)
        target_landmarks = {
            't1': [50, 200, 400, 600, 800, 1000, 1200],
            't1ce': [50, 250, 500, 750, 1000, 1250, 1500],
            't2': [100, 300, 600, 900, 1200, 1500, 1800],
            'flair': [50, 200, 450, 700, 950, 1200, 1450]
        }
        
        target = target_landmarks.get(modality_name, target_landmarks['t1'])
        
        # Piecewise linear interpolation
        normalized_data = np.interp(data, current_landmarks, target)
        
        return normalized_data
    
    def _resample_volume(
        self,
        data: np.ndarray,
        current_spacing: Tuple[float, float, float],
        target_spacing: Tuple[float, float, float],
        order: int = 1
    ) -> np.ndarray:
        """
        Resample volume to target spacing
        
        Args:
            data: Input volume
            current_spacing: Current voxel spacing
            target_spacing: Target voxel spacing
            order: Interpolation order
            
        Returns:
            Resampled volume
        """
        # Calculate zoom factors
        zoom_factors = [
            current_spacing[i] / target_spacing[i] for i in range(3)
        ]
        
        # Only resample if significant difference
        if any(abs(zf - 1.0) > 0.05 for zf in zoom_factors):
            data = zoom(data, zoom_factors, order=order, cval=0)
            
            if self.verbose:
                print(f"      Resampled from {current_spacing} to {target_spacing}")
        
        return data
    
    def _crop_to_brain(self, data: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
        """
        Crop volume to brain region with padding
        
        Args:
            data: Input volume
            brain_mask: Brain mask
            
        Returns:
            Cropped volume
        """
        # Find bounding box of brain
        coords = np.where(brain_mask > 0)
        
        if len(coords[0]) == 0:
            return data
        
        # Get bounding box
        min_coords = [np.min(coord) for coord in coords]
        max_coords = [np.max(coord) for coord in coords]
        
        # Add padding
        padding = 10
        min_coords = [max(0, mc - padding) for mc in min_coords]
        max_coords = [
            min(data.shape[i], mc + padding)
            for i, mc in enumerate(max_coords)
        ]
        
        # Crop
        cropped = data[
            min_coords[0]:max_coords[0],
            min_coords[1]:max_coords[1],
            min_coords[2]:max_coords[2]
        ]
        
        return cropped
    
    def _resize_volume(
        self,
        data: np.ndarray,
        target_shape: Tuple[int, int, int],
        order: int = 1
    ) -> np.ndarray:
        """
        Resize volume to target shape
        
        Args:
            data: Input volume
            target_shape: Target shape
            order: Interpolation order
            
        Returns:
            Resized volume
        """
        if data.shape == target_shape:
            return data
        
        # Use skimage resize with anti-aliasing
        resized = resize(
            data,
            target_shape,
            order=order,
            preserve_range=True,
            anti_aliasing=order > 0
        )
        
        return resized.astype(data.dtype)
    
    def _remove_small_components(
        self,
        binary_mask: np.ndarray,
        min_size: int = 100
    ) -> np.ndarray:
        """
        Remove small connected components
        
        Args:
            binary_mask: Binary mask
            min_size: Minimum component size
            
        Returns:
            Cleaned binary mask
        """
        cleaned = morphology.remove_small_objects(
            binary_mask.astype(bool),
            min_size=min_size
        )
        
        return cleaned.astype(np.float32)
    
    def _morphological_cleanup(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up mask
        
        Args:
            binary_mask: Input binary mask
            
        Returns:
            Cleaned mask
        """
        # Convert to boolean
        mask = binary_mask.astype(bool)
        
        # Closing to fill small holes
        mask = binary_closing(mask, structure=np.ones((3, 3, 3)))
        
        # Opening to remove small protrusions
        mask = binary_opening(mask, structure=np.ones((2, 2, 2)))
        
        # Fill holes
        mask = ndimage.binary_fill_holes(mask)
        
        if mask is None:
            return np.zeros_like(binary_mask, dtype=np.float32)
        return mask.astype(np.float32)
    
    def _final_cleanup(self, data: np.ndarray) -> np.ndarray:
        """
        Final cleanup of preprocessed volume
        
        Args:
            data: Input volume
            
        Returns:
            Cleaned volume
        """
        # Remove NaN and infinite values
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Optional denoising
        if hasattr(self.config, 'denoise') and self.config.denoise:
            # Apply mild denoising
            data = gaussian_filter(data, sigma=0.5)
        
        return data
    
    def _ensure_consistency(
        self,
        modalities: Dict[str, np.ndarray],
        segmentation: Optional[np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
        """
        Ensure all volumes have consistent shapes
        
        Args:
            modalities: Dictionary of modality volumes
            segmentation: Segmentation volume
            
        Returns:
            Consistent modalities and segmentation
        """
        target_shape = self.config.input_size
        
        # Resize all modalities
        for mod_name, data in modalities.items():
            if data.shape != target_shape:
                modalities[mod_name] = self._resize_volume(data, target_shape)
        
        # Resize segmentation
        if segmentation is not None and segmentation.shape != target_shape:
            segmentation = self._resize_volume(segmentation, target_shape, order=0)
            segmentation = (segmentation > 0.5).astype(np.float32)
        
        return modalities, segmentation
    
    def _calculate_volume_stats(
        self,
        data: np.ndarray,
        voxel_spacing: Tuple[float, float, float]
    ) -> VolumeStats:
        """
        Calculate volume statistics
        
        Args:
            data: Volume data
            voxel_spacing: Voxel spacing
            
        Returns:
            Volume statistics
        """
        voxel_volume = np.prod(voxel_spacing)
        non_zero_mask = data != 0
        non_zero_data = data[non_zero_mask]
        
        percentiles = {}
        if len(non_zero_data) > 0:
            for p in [1, 5, 25, 50, 75, 95, 99]:
                percentiles[p] = float(np.percentile(non_zero_data, p))
        
        return VolumeStats(
            shape=(int(data.shape[0]), int(data.shape[1]), int(data.shape[2])),
            voxel_spacing=voxel_spacing,
            volume_mm3=float(np.sum(non_zero_mask) * voxel_volume),
            intensity_range=(float(np.min(data)), float(np.max(data))),
            mean_intensity=float(np.mean(non_zero_data)) if len(non_zero_data) > 0 else 0.0,
            std_intensity=float(np.std(non_zero_data)) if len(non_zero_data) > 0 else 0.0,
            percentiles=percentiles,
            non_zero_voxels=int(np.sum(non_zero_mask))
        )
    
    def _calculate_tumor_stats(
        self,
        seg_data: np.ndarray,
        voxel_spacing: Tuple[float, float, float]
    ) -> VolumeStats:
        """
        Calculate tumor-specific statistics
        
        Args:
            seg_data: Segmentation data
            voxel_spacing: Voxel spacing
            
        Returns:
            Tumor statistics
        """
        voxel_volume = np.prod(voxel_spacing)
        tumor_voxels = np.sum(seg_data > 0)
        tumor_volume = tumor_voxels * voxel_volume
        total_volume = np.prod(seg_data.shape) * voxel_volume
        
        return VolumeStats(
            shape=(int(seg_data.shape[0]), int(seg_data.shape[1]), int(seg_data.shape[2])),
            voxel_spacing=voxel_spacing,
            volume_mm3=float(total_volume),
            intensity_range=(0.0, 1.0),
            mean_intensity=float(np.mean(seg_data)),
            std_intensity=float(np.std(seg_data)),
            percentiles={50: 0.0 if tumor_voxels == 0 else 1.0},
            non_zero_voxels=int(tumor_voxels),
            tumor_volume_mm3=float(tumor_volume),
            tumor_percentage=float(tumor_voxels / np.prod(seg_data.shape) * 100)
        )
    
    def _get_voxel_spacing(self, nifti_img: Nifti1Image) -> Tuple[float, float, float]:
        """
        Extract voxel spacing from NIfTI header
        
        Args:
            nifti_img: NIfTI image object
            
        Returns:
            Voxel spacing (mm)
        """
        try:
            header = nifti_img.header
            spacing = header.get_zooms()
            spacing = tuple(list(spacing) + [1.0] * (3 - len(spacing)))
            return (float(spacing[0]), float(spacing[1]), float(spacing[2]))
        except AttributeError as e:
            raise ValueError("Invalid NIfTI image provided") from e
    
    def _save_processed_case(
        self,
        result: Dict[str, Any],
        output_dir: Union[str, Path],
        reference_img: Optional[Nifti1Image]
    ):
        """
        Save processed case to disk
        
        Args:
            result: Processing result dictionary
            output_dir: Output directory
            reference_img: Reference NIfTI image
        """
        output_dir = Path(output_dir)
        case_dir = output_dir / result['case_id']
        case_dir.mkdir(parents=True, exist_ok=True)
        
        # Save modalities
        for mod_name, data in result['modalities'].items():
            save_path = case_dir / f"{result['case_id']}_{mod_name}_preprocessed.nii.gz"
            save_nifti(data, save_path, reference_img=reference_img)
        
        # Save segmentation
        if result['segmentation'] is not None:
            seg_path = case_dir / f"{result['case_id']}_seg_preprocessed.nii.gz"
            save_nifti(result['segmentation'], seg_path, reference_img=reference_img)
        
        # Save statistics
        stats_path = case_dir / f"{result['case_id']}_stats.json"
        with open(stats_path, 'w') as f:
            import json
            # Convert numpy types to native Python types for JSON serialization
            stats_dict = {}
            for key, stats in result['stats'].items():
                if isinstance(stats, VolumeStats):
                    stats_dict[key] = {
                        'shape': stats.shape,
                        'voxel_spacing': stats.voxel_spacing,
                        'volume_mm3': stats.volume_mm3,
                        'intensity_range': stats.intensity_range,
                        'mean_intensity': stats.mean_intensity,
                        'std_intensity': stats.std_intensity,
                        'percentiles': stats.percentiles,
                        'non_zero_voxels': stats.non_zero_voxels,
                        'tumor_volume_mm3': stats.tumor_volume_mm3,
                        'tumor_percentage': stats.tumor_percentage
                    }
                else:
                    stats_dict[key] = stats
            
            json.dump(stats_dict, f, indent=2)


def batch_preprocess(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    config: Optional[Config] = None,
    num_workers: int = 4
) -> Dict[str, Any]:
    """
    Batch preprocess multiple BraTS cases
    
    Args:
        input_dir: Input directory containing cases
        output_dir: Output directory for processed cases
        config: Configuration object
        num_workers: Number of parallel workers
        
    Returns:
        Processing summary
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = config or Config()
    preprocessor = VolumePreprocessor(config, verbose=True)
    
    # Find all cases
    case_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(case_dirs)} cases to preprocess")
    
    successful = 0
    failed = 0
    
    with timer("Batch preprocessing"):
        for case_dir in case_dirs:
            case_id = case_dir.name
            print(f"\nProcessing case: {case_id}")
            
            # Find modalities
            modalities = {}
            modality_patterns = {
                't1': ['*t1.nii.gz', '*T1.nii.gz'],
                't1ce': ['*t1ce.nii.gz', '*T1CE.nii.gz', '*t1c.nii.gz'],
                't2': ['*t2.nii.gz', '*T2.nii.gz'],
                'flair': ['*flair.nii.gz', '*FLAIR.nii.gz']
            }
            
            for mod_name, patterns in modality_patterns.items():
                for pattern in patterns:
                    files = list(case_dir.glob(pattern))
                    if files:
                        modalities[mod_name] = files[0]
                        break
            
            # Find segmentation
            seg_files = list(case_dir.glob('*seg.nii.gz')) + list(case_dir.glob('*SEG.nii.gz'))
            segmentation = seg_files[0] if seg_files else None
            
            if not modalities:
                print(f"  No modalities found for case {case_id}")
                failed += 1
                continue
            
            # Preprocess case
            result = preprocessor.preprocess_case(
                modalities=modalities,
                segmentation=segmentation,
                output_dir=output_dir,
                case_id=case_id
            )
            
            if result['success']:
                successful += 1
                print(f"  ✓ Successfully preprocessed {case_id}")
            else:
                failed += 1
                print(f"  ✗ Failed to preprocess {case_id}: {result.get('error', 'Unknown error')}")
    
    summary = {
        'total_cases': len(case_dirs),
        'successful': successful,
        'failed': failed,
        'success_rate': successful / len(case_dirs) if case_dirs else 0.0
    }
    
    print(f"\nPreprocessing Summary:")
    print(f"  Total cases: {summary['total_cases']}")
    print(f"  Successful: {summary['successful']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Success rate: {summary['success_rate']:.1%}")
    
    return summary


def test_preprocessing():
    """Test preprocessing functionality"""
    print("Testing preprocessing pipeline...")
    
    # Create test data
    test_shape = (64, 64, 64)
    test_data = np.random.randn(*test_shape).astype(np.float32)
    test_seg = (np.random.rand(*test_shape) > 0.9).astype(np.float32)
    
    # Test config
    config = Config(
        input_size=(32, 32, 32),
        normalize_method="zscore"
    )
    
    preprocessor = VolumePreprocessor(config, verbose=True)
    
    # Test single modality preprocessing
    try:
        processed = preprocessor._preprocess_modality(
            test_data, "t1", (1.0, 1.0, 1.0)
        )
        print(f"✓ Modality preprocessing: {test_data.shape} -> {processed.shape}")
        
        # Test segmentation preprocessing
        processed_seg = preprocessor._preprocess_segmentation(
            test_seg, (1.0, 1.0, 1.0)
        )
        print(f"✓ Segmentation preprocessing: {test_seg.shape} -> {processed_seg.shape}")
        
        # Test volume statistics
        stats = preprocessor._calculate_volume_stats(
            processed, (1.0, 1.0, 1.0)
        )
        print(f"✓ Volume statistics calculated: {stats.shape}")
        
        # Test complete case preprocessing
        modalities = {
            't1': test_data,
            't1ce': test_data * 1.2,
            't2': test_data * 0.8,
            'flair': test_data * 1.5
        }
        
        result = preprocessor.preprocess_case(
            modalities=modalities,
            segmentation=test_seg,
            case_id="test_case"
        )
        
        if result['success']:
            print("✓ Complete case preprocessing successful")
            print(f"  Processed modalities: {list(result['modalities'].keys())}")
            print(f"  Statistics available: {list(result['stats'].keys())}")
        else:
            print(f"✗ Case preprocessing failed: {result.get('error', 'Unknown')}")
            
    except Exception as e:
        print(f"✗ Preprocessing test failed: {e}")
    
    print("✓ Preprocessing testing completed")


class QualityAssessment:
    """
    Quality assessment for preprocessed volumes
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
    
    def assess_volume_quality(self, volume: np.ndarray) -> Dict[str, float]:
        """
        Assess quality metrics for a preprocessed volume
        
        Args:
            volume: Preprocessed volume
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Signal-to-noise ratio (simplified)
        signal = np.mean(volume[volume > 0])
        noise = np.std(volume[volume == 0]) if np.any(volume == 0) else 0.1
        metrics['snr'] = float(signal / noise) if noise > 0 else float('inf')
        
        # Contrast-to-noise ratio
        if np.std(volume) > 0:
            metrics['cnr'] = float(signal / np.std(volume))
        else:
            metrics['cnr'] = 0.0
        
        # Intensity uniformity (coefficient of variation)
        brain_voxels = volume[volume > 0]
        if len(brain_voxels) > 0 and np.mean(brain_voxels) > 0:
            metrics['uniformity'] = float(np.std(brain_voxels) / np.mean(brain_voxels))
        else:
            metrics['uniformity'] = 0.0
        
        # Artifacts detection (simplified)
        # Look for extreme values that might indicate artifacts
        if len(brain_voxels) > 0:
            q99 = np.percentile(brain_voxels, 99)
            q01 = np.percentile(brain_voxels, 1)
            extreme_voxels = np.sum((brain_voxels > q99 * 2) | (brain_voxels < q01 / 2))
            metrics['artifact_ratio'] = float(extreme_voxels / len(brain_voxels))
        else:
            metrics['artifact_ratio'] = 0.0
        
        # Normalization check
        metrics['mean_intensity'] = float(np.mean(brain_voxels)) if len(brain_voxels) > 0 else 0.0
        metrics['std_intensity'] = float(np.std(brain_voxels)) if len(brain_voxels) > 0 else 0.0
        
        return metrics
    
    def assess_registration_quality(
        self, 
        modalities: Dict[str, np.ndarray],
        reference_modality: str = 't1'
    ) -> Dict[str, float]:
        """
        Assess quality of multi-modal registration
        
        Args:
            modalities: Dictionary of modality volumes
            reference_modality: Reference modality for registration
            
        Returns:
            Registration quality metrics
        """
        metrics = {}
        
        if reference_modality not in modalities:
            return {}
        
        reference = modalities[reference_modality]
        
        for mod_name, volume in modalities.items():
            if mod_name == reference_modality:
                continue
            
            # Mutual information (simplified)
            mi = self._calculate_mutual_information(reference, volume)
            metrics[f'{mod_name}_mutual_info'] = mi
            
            # Normalized cross-correlation
            ncc = self._calculate_ncc(reference, volume)
            metrics[f'{mod_name}_ncc'] = ncc
        
        return metrics
    
    def _calculate_mutual_information(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate mutual information between two images (simplified)
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Mutual information value
        """
        # Flatten and remove zero voxels
        flat1 = img1.flatten()
        flat2 = img2.flatten()
        
        mask = (flat1 != 0) & (flat2 != 0)
        flat1 = flat1[mask]
        flat2 = flat2[mask]
        
        if len(flat1) == 0:
            return 0.0
        
        # Create joint histogram
        hist_2d, x_edges, y_edges = np.histogram2d(flat1, flat2, bins=50)
        
        # Normalize to probabilities
        pxy = hist_2d / float(np.sum(hist_2d))
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        
        # Calculate mutual information
        px_py = px[:, None] * py[None, :]
        
        # Avoid log(0)
        nzs = pxy > 0
        
        mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
        
        return float(mi)
    
    def _calculate_ncc(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate normalized cross-correlation
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            NCC value
        """
        # Flatten and remove zero voxels
        flat1 = img1.flatten()
        flat2 = img2.flatten()
        
        mask = (flat1 != 0) & (flat2 != 0)
        flat1 = flat1[mask]
        flat2 = flat2[mask]
        
        if len(flat1) == 0:
            return 0.0
        
        # Normalize
        flat1 = (flat1 - np.mean(flat1)) / np.std(flat1)
        flat2 = (flat2 - np.mean(flat2)) / np.std(flat2)
        
        # Calculate NCC
        ncc = np.mean(flat1 * flat2)
        
        return float(ncc)


def create_preprocessing_pipeline(
    config: Optional[Config] = None,
    quality_assessment: bool = True
) -> Tuple[VolumePreprocessor, Optional[QualityAssessment]]:
    """
    Create a complete preprocessing pipeline
    
    Args:
        config: Configuration object
        quality_assessment: Whether to include quality assessment
        
    Returns:
        Preprocessor and optional quality assessor
    """
    config = config or Config()
    preprocessor = VolumePreprocessor(config, verbose=True)
    
    qa = QualityAssessment(config) if quality_assessment else None
    
    return preprocessor, qa


def preprocess_for_inference(
    volume_paths: Dict[str, Union[str, Path]],
    config: Optional[Config] = None,
    output_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Preprocess volumes for inference (without ground truth)
    
    Args:
        volume_paths: Dictionary mapping modality names to file paths
        config: Configuration object
        output_path: Optional path to save preprocessed volume
        
    Returns:
        Preprocessed data ready for inference
    """
    config = config or Config()
    preprocessor = VolumePreprocessor(config, verbose=False)
    
    # Load volumes
    volumes = {}
    reference_img = None
    first_volume_data = None
    
    for mod_name, path in volume_paths.items():
        data, nii_img = load_nifti(path)
        volumes[mod_name] = data
        if reference_img is None:
            reference_img = nii_img
            first_volume_data = data
    
    # Get voxel spacing
    if reference_img is not None:
        voxel_spacing = preprocessor._get_voxel_spacing(reference_img)
    else:
        voxel_spacing = (1.0, 1.0, 1.0)
    
    # Preprocess each modality
    processed_volumes = []
    provided_mods = list(volumes.keys())
    if len(provided_mods) == 1:
        # Replicate single modality across all 4 channels
        single = preprocessor._preprocess_modality(
            volumes[provided_mods[0]], provided_mods[0], voxel_spacing, skip_cropping=True
        )
        processed_volumes = [single, single, single, single]
    else:
        for mod_name in ['t1', 't1ce', 't2', 'flair']:
            if mod_name in volumes:
                processed = preprocessor._preprocess_modality(
                    volumes[mod_name], mod_name, voxel_spacing, skip_cropping=True
                )
            else:
                # Create dummy volume if modality is missing
                processed = np.zeros(config.input_size, dtype=np.float32)
            processed_volumes.append(processed)
    
    # Stack into 4D tensor
    volume_4d = np.stack(processed_volumes, axis=0)  # Shape: (4, D, H, W)
    
    # Convert to PyTorch tensor and add batch dimension
    tensor = torch.from_numpy(volume_4d).float().unsqueeze(0)  # Shape: (1, 4, D, H, W)
    
    result = {
        'tensor': tensor,
        'original_shape': first_volume_data.shape if first_volume_data is not None else config.input_size,
        'voxel_spacing': voxel_spacing,
        'reference_image': reference_img,
        'processed_shape': config.input_size,
        'original_scan': first_volume_data if first_volume_data is not None else np.zeros(config.input_size, dtype=np.float32),
        'affine': reference_img.affine if reference_img is not None else np.eye(4)
    }
    
    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, output_path)
    
    return result


# Additional utility functions for advanced preprocessing

def estimate_preprocessing_time(
    num_cases: int,
    input_size: Tuple[int, int, int] = (128, 128, 128),
    num_modalities: int = 4
) -> Dict[str, Any]:
    """
    Estimate preprocessing time based on dataset size
    
    Args:
        num_cases: Number of cases to preprocess
        input_size: Target input size
        num_modalities: Number of modalities per case
        
    Returns:
        Time estimates in various units
    """
    # Rough estimates based on typical processing times
    base_time_per_case = 30  # seconds per case for (128, 128, 128)
    
    # Scale by volume size
    volume_factor = np.prod(input_size) / (128 ** 3)
    time_per_case = base_time_per_case * volume_factor * num_modalities / 4
    
    total_seconds = num_cases * time_per_case
    
    return {
        'seconds_per_case': float(time_per_case),
        'total_seconds': float(total_seconds),
        'total_minutes': float(total_seconds / 60),
        'total_hours': float(total_seconds / 3600),
        'estimated_completion': f"{total_seconds / 3600:.1f} hours"
    }


if __name__ == "__main__":
    test_preprocessing()