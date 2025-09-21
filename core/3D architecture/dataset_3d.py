"""
3D Dataset Loader for Brain Tumor Segmentation

Handles loading and preprocessing of BraTS dataset with multi-modal MRI volumes.
Supports T1, T1CE, T2, FLAIR modalities and ground truth segmentation masks.

Key Features:
- Multi-modal MRI loading (4 channels)
- Robust preprocessing pipeline
- Data augmentation support
- Efficient caching system
- Memory-optimized loading
- Missing modality handling
"""

import os
import gc
import random
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
from scipy.ndimage import rotate, zoom, gaussian_filter
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from skimage.transform import resize

from utils_3d import Config, load_nifti, save_nifti, format_bytes, timer


@dataclass
class BraTSCase:
    """
    Data structure for a single BraTS case
    """
    case_id: str
    t1_path: Optional[Path] = None
    t1ce_path: Optional[Path] = None
    t2_path: Optional[Path] = None
    flair_path: Optional[Path] = None
    seg_path: Optional[Path] = None
    
    def __post_init__(self):
        """Convert string paths to Path objects"""
        for field in ['t1_path', 't1ce_path', 't2_path', 'flair_path', 'seg_path']:
            value = getattr(self, field)
            if value is not None and not isinstance(value, Path):
                setattr(self, field, Path(value))
    
    @property
    def modalities(self) -> Dict[str, Optional[Path]]:
        """Get dictionary of modality paths"""
        return {
            't1': self.t1_path,
            't1ce': self.t1ce_path,
            't2': self.t2_path,
            'flair': self.flair_path
        }
    
    @property
    def available_modalities(self) -> List[str]:
        """Get list of available modalities"""
        return [mod for mod, path in self.modalities.items() if path and path.exists()]
    
    def is_complete(self) -> bool:
        """Check if all modalities and segmentation are available"""
        all_paths = [self.t1_path, self.t1ce_path, self.t2_path, self.flair_path, self.seg_path]
        return all(path and path.exists() for path in all_paths)


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS brain tumor segmentation
    
    Args:
        data_dir: Root directory containing BraTS data
        split: Dataset split ("train", "val", "test")
        config: Configuration object
        transform: Optional transform function
        cache_data: Whether to cache preprocessed data
        preload: Whether to preload all data into memory
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        config: Optional[Config] = None,
        transform: Optional[Callable] = None,
        cache_data: bool = True,
        preload: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.config = config or Config()
        self.transform = transform
        self.cache_data = cache_data
        self.preload = preload
        
        # Initialize storage
        self.cases: List[BraTSCase] = []
        self.cache: Dict[str, Any] = {}
        self.cache_dir = Path(f"./cache/{split}")
        
        if self.cache_data:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        self._discover_cases()
        self._validate_cases()
        
        if self.preload:
            self._preload_data()
        
        print(f"Loaded {len(self.cases)} cases for {split} split")
    
    def _discover_cases(self):
        """Discover all BraTS cases in the data directory"""
        print(f"Discovering cases in {self.data_dir}...")
        
        # Common BraTS naming patterns
        modality_suffixes = {
            't1': ['_t1.nii.gz', '_T1.nii.gz'],
            't1ce': ['_t1ce.nii.gz', '_T1CE.nii.gz', '_t1c.nii.gz'],
            't2': ['_t2.nii.gz', '_T2.nii.gz'],
            'flair': ['_flair.nii.gz', '_FLAIR.nii.gz'],
            'seg': ['_seg.nii.gz', '_SEG.nii.gz']
        }
        
        case_dirs = []
        
        # Look for case directories
        if self.data_dir.exists():
            for item in self.data_dir.iterdir():
                if item.is_dir():
                    case_dirs.append(item)
        
        if not case_dirs:
            # Try flat structure
            nii_files = list(self.data_dir.glob("*.nii.gz"))
            case_ids = set()
            for file in nii_files:
                # Extract case ID from filename
                case_id = file.stem.replace('.nii', '')
                for mod, suffixes in modality_suffixes.items():
                    for suffix in suffixes:
                        if case_id.endswith(suffix.replace('.nii.gz', '')):
                            case_id = case_id[:-len(suffix.replace('.nii.gz', ''))]
                            break
                case_ids.add(case_id)
            
            # Create cases from flat structure
            for case_id in sorted(case_ids):
                case = BraTSCase(case_id=case_id)
                
                for mod, suffixes in modality_suffixes.items():
                    for suffix in suffixes:
                        file_path = self.data_dir / f"{case_id}{suffix}"
                        if file_path.exists():
                            setattr(case, f"{mod}_path", file_path)
                            break
                
                if any(getattr(case, f"{mod}_path") for mod in ['t1', 't1ce', 't2', 'flair']):
                    self.cases.append(case)
        else:
            # Hierarchical structure
            for case_dir in sorted(case_dirs):
                case_id = case_dir.name
                case = BraTSCase(case_id=case_id)
                
                for mod, suffixes in modality_suffixes.items():
                    for suffix in suffixes:
                        file_path = case_dir / f"{case_id}{suffix}"
                        if file_path.exists():
                            setattr(case, f"{mod}_path", file_path)
                            break
                        # Also try without case_id prefix
                        for alt_name in [f"{mod}{suffix}", suffix]:
                            alt_path = case_dir / alt_name
                            if alt_path.exists():
                                setattr(case, f"{mod}_path", alt_path)
                                break
                
                if any(getattr(case, f"{mod}_path") for mod in ['t1', 't1ce', 't2', 'flair']):
                    self.cases.append(case)
    
    def _validate_cases(self):
        """Validate discovered cases"""
        valid_cases = []
        
        for case in self.cases:
            available_mods = case.available_modalities
            
            if len(available_mods) == 0:
                warnings.warn(f"Case {case.case_id}: No modalities found")
                continue
            
            if len(available_mods) < 2:
                warnings.warn(f"Case {case.case_id}: Only {len(available_mods)} modalities available")
            
            valid_cases.append(case)
        
        self.cases = valid_cases
        
        # Split data if needed (simple random split for demo)
        if hasattr(self, '_split_data'):
            self._split_data()
    
    def _preload_data(self):
        """Preload all data into memory"""
        print("Preloading data into memory...")
        
        for i in range(len(self.cases)):
            if i % 10 == 0:
                print(f"Preloading: {i}/{len(self.cases)}")
            
            try:
                self._load_case_data(i, cache_result=True)
            except Exception as e:
                warnings.warn(f"Failed to preload case {i}: {e}")
        
        print("Data preloading completed")
    
    def _load_case_data(self, idx: int, cache_result: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """
        Load and preprocess data for a single case

        Args:
            idx: Case index
            cache_result: Whether to cache the result

        Returns:
            (input_tensor, target_tensor, metadata)
        """
        case = self.cases[idx]
        cache_key = f"{case.case_id}_{self.split}"

        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]

        if self.cache_data:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    if cache_result:
                        self.cache[cache_key] = data
                    return data
                except:
                    warnings.warn(f"Failed to load cache file {cache_file}")

        # Load modalities
        modalities = []
        reference_img = None
        metadata = {}

        for mod_name in ['t1', 't1ce', 't2', 'flair']:
            mod_path = getattr(case, f"{mod_name}_path")

            if mod_path and mod_path.exists():
                try:
                    data, nii_img = load_nifti(mod_path)
                    if reference_img is None:
                        reference_img = nii_img
                        metadata['affine'] = nii_img.affine
                        metadata['header'] = nii_img.header
                        metadata['original_shape'] = data.shape
                    modalities.append(data)
                except Exception as e:
                    warnings.warn(f"Failed to load {mod_name} for case {case.case_id}: {e}")
                    modalities.append(np.zeros_like(modalities[0]) if modalities else np.zeros(self.config.input_size))
            else:
                # Missing modality - use zeros or mean of available modalities
                if modalities:
                    modalities.append(np.zeros_like(modalities[0]))
                else:
                    modalities.append(np.zeros(self.config.input_size))

        if not modalities:
            raise ValueError(f"No valid modalities found for case {case.case_id}")

        # Ensure all modalities have the same shape
        target_shape = self.config.input_size
        processed_modalities = []

        for i, modality in enumerate(modalities):
            # Preprocess individual modality
            modality = self._preprocess_volume(modality, target_shape)
            processed_modalities.append(modality)

        # Stack modalities into 4-channel tensor
        input_volume = np.stack(processed_modalities, axis=0)  # Shape: (4, D, H, W)
        input_tensor = torch.from_numpy(input_volume).float()

        # Load segmentation if available
        target_tensor = None
        if case.seg_path and case.seg_path.exists() and self.split != "test":
            try:
                seg_data, _ = load_nifti(case.seg_path)
                seg_data = self._preprocess_segmentation(seg_data, target_shape)
                target_tensor = torch.from_numpy(seg_data).float().unsqueeze(0)  # Shape: (1, D, H, W)
            except Exception as e:
                warnings.warn(f"Failed to load segmentation for case {case.case_id}: {e}")

        result = (input_tensor, target_tensor, metadata)

        # Cache result
        if cache_result:
            if self.cache_data:
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(result, f)
                except Exception as e:
                    warnings.warn(f"Failed to save cache file {cache_file}: {e}")

            self.cache[cache_key] = result

        return result
    
    def _preprocess_volume(self, volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Preprocess a single volume
        
        Args:
            volume: Input volume
            target_shape: Target shape (D, H, W)
            
        Returns:
            Preprocessed volume
        """
        # Handle different input shapes
        if volume.ndim == 4:
            volume = volume.squeeze()
        
        # Resize to target shape
        if volume.shape != target_shape:
            # Use resize with anti-aliasing for better quality
            volume = resize(
                volume, 
                target_shape, 
                order=1, 
                preserve_range=True, 
                anti_aliasing=True
            ).astype(np.float32)
        
        # Normalization
        volume = self._normalize_volume(volume)
        
        return volume
    
    def _preprocess_segmentation(self, segmentation: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Preprocess segmentation mask
        
        Args:
            segmentation: Input segmentation
            target_shape: Target shape (D, H, W)
            
        Returns:
            Preprocessed segmentation
        """
        # Handle different input shapes
        if segmentation.ndim == 4:
            segmentation = segmentation.squeeze()
        
        # Convert multi-class to binary (tumor vs background)
        # BraTS labels: 0=background, 1=necrotic, 2=edema, 4=enhancing
        binary_mask = (segmentation > 0).astype(np.float32)
        
        # Resize to target shape using nearest neighbor
        if binary_mask.shape != target_shape:
            binary_mask = resize(
                binary_mask, 
                target_shape, 
                order=0,  # Nearest neighbor for labels
                preserve_range=True, 
                anti_aliasing=False
            ).astype(np.float32)
        
        # Ensure binary
        binary_mask = (binary_mask > 0.5).astype(np.float32)
        
        return binary_mask
    
    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Normalize volume based on configuration
        
        Args:
            volume: Input volume
            
        Returns:
            Normalized volume
        """
        # Remove NaN and infinite values
        volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.config.normalize_method == "zscore":
            # Z-score normalization (zero mean, unit variance)
            mean = np.mean(volume[volume > 0])  # Exclude background
            std = np.std(volume[volume > 0])
            
            if std > 0:
                volume = (volume - mean) / std
            else:
                volume = volume - mean
                
        elif self.config.normalize_method == "minmax":
            # Min-max normalization to [0, 1]
            min_val = np.min(volume)
            max_val = np.max(volume)
            
            if max_val > min_val:
                volume = (volume - min_val) / (max_val - min_val)
                
        elif self.config.normalize_method == "percentile":
            # Percentile clipping and normalization
            low, high = self.config.clip_percentiles
            p_low, p_high = np.percentile(volume[volume > 0], [low, high])
            
            volume = np.clip(volume, p_low, p_high)
            if p_high > p_low:
                volume = (volume - p_low) / (p_high - p_low)
        
        return volume.astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.cases)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """
        Get a single item from the dataset
        
        Args:
            idx: Item index
            
        Returns:
            (input_tensor, target_tensor, metadata)
        """
        # Load data
        input_tensor, target_tensor, metadata = self._load_case_data(idx)
        
        # Apply transforms
        if self.transform and self.split == "train":
            input_tensor, target_tensor = self.transform(input_tensor, target_tensor)
        
        return input_tensor, target_tensor, metadata
    
    def get_case_info(self, idx: int) -> BraTSCase:
        """Get case information"""
        return self.cases[idx]
    
    def clear_cache(self):
        """Clear memory cache"""
        self.cache.clear()
        gc.collect()
    
    # Ensure '_split_data' is defined to avoid attribute errors
    def _split_data(self):
        """Placeholder for data splitting logic."""
        pass


class DataAugmentation3D:
    """
    3D data augmentation for brain MRI
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.rotation_range = config.rotation_range
        self.elastic_alpha = config.elastic_alpha
        self.elastic_sigma = config.elastic_sigma
    
    def __call__(self, input_tensor: torch.Tensor, target_tensor: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply augmentations
        
        Args:
            input_tensor: Input tensor (C, D, H, W)
            target_tensor: Target tensor (1, D, H, W) or None
            
        Returns:
            Augmented tensors
        """
        if not self.config.use_augmentation:
            return input_tensor, target_tensor
        
        # Convert to numpy for augmentation
        input_np = input_tensor.numpy()
        target_np = target_tensor.numpy() if target_tensor is not None else None
        
        # Random rotation
        if random.random() < 0.5:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            axes = random.choice([(0, 1), (0, 2), (1, 2)])  # Random rotation plane
            
            for c in range(input_np.shape[0]):
                input_np[c] = rotate(input_np[c], angle, axes=axes, reshape=False, order=1, cval=0)
            
            if target_np is not None:
                target_np[0] = rotate(target_np[0], angle, axes=axes, reshape=False, order=0, cval=0)
        
        # Random flip
        if random.random() < 0.5:
            axis = random.choice([1, 2, 3])  # Don't flip channel dimension
            input_np = np.flip(input_np, axis=axis).copy()
            if target_np is not None:
                target_np = np.flip(target_np, axis=axis).copy()
        
        # Intensity augmentation (only for input)
        if random.random() < 0.3:
            # Random gamma correction
            gamma = random.uniform(0.8, 1.2)
            input_np = np.power(np.clip(input_np, 0, 1), gamma)
        
        if random.random() < 0.3:
            # Random noise
            noise_factor = random.uniform(0.0, 0.1)
            noise = np.random.normal(0, noise_factor, input_np.shape)
            input_np = input_np + noise
        
        # Convert back to tensors
        input_tensor = torch.from_numpy(input_np.astype(np.float32))
        target_tensor = torch.from_numpy(target_np.astype(np.float32)) if target_np is not None else None
        
        return input_tensor, target_tensor


def create_data_loaders(
    data_dir: Union[str, Path],
    config: Config,
    splits: List[str] = ["train", "val"]
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training and validation
    
    Args:
        data_dir: Path to data directory
        config: Configuration object
        splits: List of splits to create loaders for
        
    Returns:
        Dictionary of data loaders
    """
    data_loaders = {}
    
    # Create augmentation transform
    transform = DataAugmentation3D(config) if config.use_augmentation else None
    
    for split in splits:
        # Only use augmentation for training
        split_transform = transform if split == "train" else None
        
        dataset = BraTSDataset(
            data_dir=data_dir,
            split=split,
            config=config,
            transform=split_transform,
            cache_data=True,
            preload=False  # Set to True if you have enough RAM
        )
        
        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=(split == "train"),
            persistent_workers=config.num_workers > 0
        )
        
        data_loaders[split] = data_loader
        
        print(f"{split.capitalize()} loader: {len(dataset)} samples, {len(data_loader)} batches")
    
    return data_loaders


def test_dataset():
    """Test dataset functionality"""
    print("Testing BraTS dataset...")
    
    # Create test config
    config = Config(
        input_size=(64, 64, 64),  # Smaller for testing
        batch_size=2,
        num_workers=0,
        use_augmentation=True
    )
    
    # Test with dummy data directory
    data_dir = Path("./test_data")
    
    try:
        dataset = BraTSDataset(
            data_dir=data_dir,
            split="train",
            config=config,
            cache_data=False
        )
        
        if len(dataset) > 0:
            # Test loading a sample
            input_tensor, target_tensor, metadata = dataset[0]
            print(f"✓ Sample loaded: input {input_tensor.shape}, target {target_tensor.shape if target_tensor is not None else None}, metadata: {metadata}")
            
            # Test data loader
            data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
            batch = next(iter(data_loader))
            print(f"✓ Batch loaded: {batch[0].shape}")
        else:
            print("⚠ No data found in test directory")
            
    except FileNotFoundError:
        print("⚠ Test data directory not found - this is expected for testing")
    
    print("✓ Dataset testing completed")


if __name__ == "__main__":
    test_dataset()