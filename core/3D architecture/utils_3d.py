"""
Utility Functions for 3D Brain Tumor Segmentation

Core utilities providing configuration management, logging, device detection,
reproducibility, and common helper functions for the segmentation pipeline.

Key Features:
- Smart device detection (CUDA/CPU)
- Reproducible random seeds
- Flexible configuration loading
- Timestamped logging
- File I/O helpers
- Memory monitoring
- Progress tracking
"""

import os
import sys
import json
import yaml
import random
import logging
import datetime
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager

import torch
import numpy as np
import psutil
import GPUtil
import nibabel as nib
from nibabel.nifti1 import Nifti1Image, Nifti1Header
from nibabel.loadsave import load, save


@dataclass
class Config:
    """
    Configuration dataclass for the segmentation pipeline
    """
    # Model configuration
    model_name: str = "standard"
    in_channels: int = 4
    out_channels: int = 1
    base_filters: int = 32
    depth: int = 4
    dropout_rate: float = 0.1
    use_attention: bool = False
    
    # Data configuration
    input_size: Tuple[int, int, int] = (128, 128, 128)
    batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = True
    
    # Training configuration
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 200
    patience: int = 20
    min_delta: float = 1e-4
    
    # Loss and metrics
    loss_function: str = "combo"  # "dice", "focal", "combo"
    dice_weight: float = 0.7
    focal_weight: float = 0.3
    dice_smooth: float = 1e-5
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    class_weights: Optional[List[float]] = None
    smooth_factor: float = 1e-5
    
    # Paths
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Hardware
    device: str = "auto"
    mixed_precision: bool = True
    compile_model: bool = False
    
    # Preprocessing
    normalize_method: str = "zscore"  # "zscore", "minmax", "percentile"
    clip_percentiles: Tuple[float, float] = (1.0, 99.0)
    resample_spacing: Optional[Tuple[float, float, float]] = None
    
    # Augmentation
    use_augmentation: bool = True
    rotation_range: float = 15.0
    elastic_alpha: float = 200.0
    elastic_sigma: float = 20.0
    
    # Reproducibility
    random_seed: int = 42
    deterministic: bool = True
    benchmark: bool = False
    
    # Logging
    log_level: str = "INFO"
    save_predictions: bool = True
    visualize_every: int = 10
    
    # Optimizer
    optimizer_type: str = "adamw"
    max_grad_norm: float = 1.0
    model_config: str = "default_model_config"
    
    # Denoising
    denoise: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
        self._create_directories()
    
    def _validate_config(self):
        """Validate configuration values"""
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")
        
        if len(self.input_size) != 3:
            raise ValueError("input_size must have 3 dimensions")
        
        if not all(s > 0 for s in self.input_size):
            raise ValueError("All input_size dimensions must be > 0")
    
    def _create_directories(self):
        """Create necessary directories"""
        dirs = [self.output_dir, self.checkpoint_dir, self.log_dir]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'Config':
        """Load configuration from YAML or JSON file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        return cls(**data)
    
    def save(self, save_path: Union[str, Path]):
        """Save configuration to file"""
        save_path = Path(save_path)
        
        with open(save_path, 'w') as f:
            if save_path.suffix.lower() in ['.yml', '.yaml']:
                yaml.dump(asdict(self), f, default_flow_style=False)
            elif save_path.suffix.lower() == '.json':
                json.dump(asdict(self), f, indent=2)
            else:
                raise ValueError(f"Unsupported save format: {save_path.suffix}")
    
    def update(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn(f"Unknown config parameter: {key}")


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Union[str, Path]] = None,
    log_filename: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory to save log files
        log_filename: Custom log filename
        
    Returns:
        Configured logger
    """
    # Configure log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    logger = logging.getLogger('segmentation3d')
    logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_dir is not None:
        log_dir = Path(log_dir) if isinstance(log_dir, str) else log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_dir / (log_filename or "log.txt")))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def set_random_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
        deterministic: Enable deterministic algorithms (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set environment variable for additional determinism
        os.environ['PYTHONHASHSEED'] = str(seed)
    else:
        torch.backends.cudnn.benchmark = True


def get_device(device: str = "auto") -> torch.device:
    """
    Get the best available device
    
    Args:
        device: Device specification ("auto", "cpu", "cuda", "cuda:0", etc.)
        
    Returns:
        PyTorch device
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    torch_device = torch.device(device)
    
    # Validate CUDA device if specified
    if torch_device.type == "cuda":
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available, falling back to CPU")
            torch_device = torch.device("cpu")
        elif torch_device.index is not None:
            if torch_device.index >= torch.cuda.device_count():
                warnings.warn(f"CUDA device {torch_device.index} not available, using device 0")
                torch_device = torch.device("cuda:0")
    
    return torch_device


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for logging and debugging
    
    Returns:
        Dictionary with system information
    """
    info = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        })
        
        # GPU memory info
        try:
            gpus = GPUtil.getGPUs()
            info["gpu_memory_info"] = [
                {"name": gpu.name, "memory_total": gpu.memoryTotal, "memory_free": gpu.memoryFree}
                for gpu in gpus
            ]
        except:
            pass
    
    return info


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value = int(bytes_value / 1024.0)
    return f"{bytes_value:.1f} PB"


def format_time(seconds: float) -> str:
    """Format seconds to human-readable time string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:.0f}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"


@contextmanager
def timer(description: str = "Operation", logger: Optional[logging.Logger] = None):
    """
    Context manager for timing operations
    
    Args:
        description: Description of the operation
        logger: Logger to use for output
    """
    start_time = datetime.datetime.now()
    
    if logger:
        logger.info(f"Starting: {description}")
    else:
        print(f"Starting: {description}")
    
    try:
        yield
    finally:
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        message = f"Completed: {description} in {format_time(duration)}"
        if logger:
            logger.info(message)
        else:
            print(message)


def monitor_memory(device: torch.device) -> Dict[str, Any]:
    """
    Monitor memory usage
    
    Args:
        device: PyTorch device to monitor
        
    Returns:
        Memory usage information
    """
    memory_info = {}
    
    # CPU memory
    cpu_memory = psutil.virtual_memory()
    memory_info["cpu"] = {
        "total": cpu_memory.total,
        "available": cpu_memory.available,
        "used": cpu_memory.used,
        "percent": cpu_memory.percent
    }
    
    # GPU memory
    if device.type == "cuda":
        gpu_memory = torch.cuda.memory_stats(device)
        memory_info["gpu"] = {
            "allocated": torch.cuda.memory_allocated(device),
            "cached": torch.cuda.memory_reserved(device),
            "max_allocated": torch.cuda.max_memory_allocated(device),
            "max_cached": torch.cuda.max_memory_reserved(device)
        }
    
    return memory_info


def load_nifti(file_path: Union[str, Path]) -> Tuple[np.ndarray, Nifti1Image]:
    """
    Load NIfTI file and return data and image object
    
    Args:
        file_path: Path to NIfTI file
        
    Returns:
        (data_array, nifti_image)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {file_path}")
    
    try:
        nifti_img = load(str(file_path))
        if not isinstance(nifti_img, Nifti1Image):
            raise TypeError("Loaded file is not a Nifti1Image")
        data = nifti_img.get_fdata().astype(np.float32)
        return data, nifti_img
    except Exception as e:
        raise RuntimeError(f"Error loading NIfTI file {file_path}: {e}")


def save_nifti(
    data: np.ndarray,
    file_path: Union[str, Path],
    reference_img: Optional[Nifti1Image] = None,
    affine: Optional[np.ndarray] = None,
    header: Optional[Nifti1Header] = None
):
    """
    Save numpy array as NIfTI file
    
    Args:
        data: Data array to save
        file_path: Output file path
        reference_img: Reference NIfTI image for affine and header
        affine: Affine transformation matrix
        header: NIfTI header
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if reference_img is not None:
        affine = reference_img.affine
        header = reference_img.header
    elif affine is None:
        # Default identity affine
        affine = np.eye(4)
    
    try:
        nifti_img = Nifti1Image(data.astype(np.float32), affine, header)
        save(nifti_img, str(file_path))
    except Exception as e:
        raise RuntimeError(f"Error saving NIfTI file {file_path}: {e}")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="3D Brain Tumor Segmentation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file (YAML or JSON)"
    )
    
    parser.add_argument(
        "--mode",
        choices=["train", "inference", "evaluate"],
        default="train",
        help="Pipeline mode"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Path to data directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Path to output directory"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use (auto, cpu, cuda, cuda:0, etc.)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


class ProgressTracker:
    """
    Simple progress tracker for training and inference
    """
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.datetime.now()
    
    def update(self, increment: int = 1):
        """Update progress"""
        self.current += increment
        self._print_progress()
    
    def _print_progress(self):
        """Print progress bar"""
        percent = (self.current / self.total) * 100
        bar_length = 40
        filled_length = int(bar_length * self.current // self.total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = format_time(eta)
        else:
            eta_str = "Unknown"
        
        print(f'\r{self.description}: |{bar}| {self.current}/{self.total} ({percent:.1f}%) ETA: {eta_str}', end='')
        
        if self.current >= self.total:
            print()  # New line when complete


def test_utils():
    """Test utility functions"""
    print("Testing utility functions...")
    
    # Test configuration
    config = Config()
    print(f"✓ Default config created: {config.model_name}")
    
    # Test device detection
    device = get_device()
    print(f"✓ Device detected: {device}")
    
    # Test logging
    logger = setup_logging("INFO")
    logger.info("✓ Logging system initialized")
    
    # Test system info
    sys_info = get_system_info()
    print(f"✓ System info: {len(sys_info)} items")
    
    # Test memory monitoring
    memory_info = monitor_memory(device)
    print(f"✓ Memory monitoring: {list(memory_info.keys())}")
    
    # Test progress tracker
    tracker = ProgressTracker(10, "Test")
    for i in range(10):
        tracker.update()
    
    print("✓ All utility functions tested successfully!")


if __name__ == "__main__":
    test_utils()