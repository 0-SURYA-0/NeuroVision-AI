"""
3D Inference Pipeline for Brain Tumor Segmentation

Production-ready inference system for processing new MRI scans and generating
tumor segmentation masks. Supports single scan inference, batch processing,
and various output formats.

Key Features:
- Single scan and batch inference
- Test-time augmentation (TTA)
- Multi-scale inference
- Uncertainty estimation
- Real-time processing
- Memory-efficient sliding window
- Quality assessment
- Multiple output formats
"""

import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, gaussian_filter
from skimage.measure import label
from skimage.morphology import remove_small_objects, binary_closing

# Import our modules
from model_3d import UNet3D, create_unet3d
from preprocess_3d import VolumePreprocessor, preprocess_for_inference
from utils_3d import Config, load_nifti, save_nifti, get_device, timer, format_time, setup_logging
from postprocess_3d import PostProcessor3D


@dataclass
class InferenceResult:
    """Container for inference results"""
    case_id: str
    prediction_mask: np.ndarray
    probability_map: Optional[np.ndarray]
    uncertainty_map: Optional[np.ndarray]
    tumor_volume_mm3: float
    tumor_voxel_count: int
    confidence_score: float
    processing_time: float
    metadata: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None


class ModelInference:
    """
    3D model inference engine
    
    Args:
        model_path: Path to trained model checkpoint
        config: Configuration object
        device: Inference device
        use_tta: Enable test-time augmentation
        use_uncertainty: Enable uncertainty estimation
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        config: Optional[Config] = None,
        device: Optional[torch.device] = None,
        use_tta: bool = False,
        use_uncertainty: bool = False,
        verbose: bool = True
    ):
        self.model_path = Path(model_path)
        self.config = config or Config()
        self.device = device or get_device("auto")
        self.use_tta = use_tta
        self.use_uncertainty = use_uncertainty
        self.verbose = verbose
        
        # Setup logging (always initialize; control verbosity via level)
        self.logger = setup_logging("INFO" if verbose else "ERROR")
        
        # Load model
        self.model = self._load_model()
        
        # Setup preprocessor
        self.preprocessor = VolumePreprocessor(self.config, verbose=False)
        
        # Inference settings
        self.threshold = getattr(self.config, 'inference_threshold', 0.5)
        self.min_tumor_size = getattr(self.config, 'min_tumor_size', 100)  # voxels
        
        if self.verbose:
            self.logger.info("Inference engine initialized")
            self.logger.info(f"Model: {self.model_path.name}")
            self.logger.info(f"Device: {self.device}")
            self.logger.info(f"TTA: {self.use_tta}, Uncertainty: {self.use_uncertainty}")
    
    def _load_model(self) -> torch.nn.Module:
        """Load trained model from checkpoint"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract config from checkpoint if available
        if 'config' in checkpoint:
            checkpoint_config = Config(**checkpoint['config'])
            # Update current config with checkpoint config for model architecture
            self.config.base_filters = checkpoint_config.base_filters
            self.config.depth = checkpoint_config.depth
            self.config.use_attention = checkpoint_config.use_attention
        
        # Create model
        model = create_unet3d("standard")  # Will use config settings
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device and set eval mode
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def predict_single(
        self,
        volume_paths: Union[Dict[str, Union[str, Path]], str, Path],
        case_id: Optional[str] = None,
        save_outputs: bool = True,
        output_dir: Optional[Union[str, Path]] = None
    ) -> InferenceResult:
        """
        Run inference on a single case
        
        Args:
            volume_paths: Dictionary of modality paths {'t1': path, 't2': path, ...}
            case_id: Case identifier
            save_outputs: Whether to save output files
            output_dir: Directory to save outputs
            
        Returns:
            InferenceResult object
        """
        start_time = time.time()
        
        # Generate case_id if not provided
        if case_id is None:
            case_id = f"case_{int(time.time())}"
        
        if self.verbose:
            self.logger.info(f"Processing case: {case_id}")
        
        try:
            # Normalize inputs
            if isinstance(volume_paths, (str, Path)):
                volume_paths = {'t1': volume_paths}
            if save_outputs and output_dir is not None and not isinstance(output_dir, Path):
                output_dir = Path(output_dir)
            # Preprocess volumes
            with timer("Preprocessing", self.logger):
                preprocessed_data = preprocess_for_inference(
                    volume_paths, self.config
                )
            
            # Extract preprocessed tensor
            input_tensor = preprocessed_data['tensor'].to(self.device)
            
            # Run inference
            with timer("Inference", self.logger):
                if self.use_tta:
                    prediction, uncertainty = self._predict_with_tta(input_tensor)
                elif self.use_uncertainty:
                    prediction, uncertainty = self._predict_with_uncertainty(input_tensor)
                else:
                    prediction = self._predict_standard(input_tensor)
                    uncertainty = None
            
            # Post-process prediction
            with timer("Post-processing", self.logger):
                binary_mask, probability_map = self._postprocess_prediction(
                    prediction, preprocessed_data
                )
            
            # Calculate metrics
            tumor_volume_mm3, tumor_voxel_count = self._calculate_tumor_metrics(
                binary_mask, preprocessed_data['voxel_spacing']
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(prediction, binary_mask)
            
            # Create result
            result = InferenceResult(
                case_id=case_id,
                prediction_mask=binary_mask,
                probability_map=probability_map,
                uncertainty_map=uncertainty.squeeze().cpu().numpy() if isinstance(uncertainty, torch.Tensor) else uncertainty,
                tumor_volume_mm3=tumor_volume_mm3,
                tumor_voxel_count=tumor_voxel_count,
                confidence_score=confidence_score,
                processing_time=time.time() - start_time,
                metadata={
                    'original_shape': preprocessed_data['original_shape'],
                    'processed_shape': preprocessed_data['processed_shape'],
                    'voxel_spacing': preprocessed_data['voxel_spacing'],
                    'threshold': self.threshold,
                    'min_tumor_size': self.min_tumor_size
                },
                success=True
            )
            
            # Save outputs
            if save_outputs and output_dir:
                output_dir_path = output_dir if isinstance(output_dir, Path) else Path(output_dir)
                self._save_outputs(result, output_dir_path, preprocessed_data)
            
            if self.verbose:
                self.logger.info(f"Successfully processed {case_id}")
                self.logger.info(f"Tumor volume: {tumor_volume_mm3:.1f} mm³")
                self.logger.info(f"Processing time: {format_time(result.processing_time)}")
            
            # Crop and save tumor region
            if save_outputs and output_dir:
                post_processor = PostProcessor3D()
                output_dir_path = output_dir if isinstance(output_dir, Path) else Path(output_dir)
                try:
                    post_processor.crop_and_save_tumor_region(
                        mask=result.prediction_mask,
                        original_scan=preprocessed_data['original_scan'],
                        affine=preprocessed_data['affine'],
                        output_path=str(output_dir_path / f"{case_id}_cropped_tumor.nii.gz")
                    )
                except ValueError:
                    # No tumor found; skip cropping silently
                    pass
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing {case_id}: {str(e)}"
            if self.verbose:
                self.logger.error(error_msg)
            
            return InferenceResult(
                case_id=case_id,
                prediction_mask=np.array([]),
                probability_map=None,
                uncertainty_map=None,
                tumor_volume_mm3=0.0,
                tumor_voxel_count=0,
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                metadata={},
                success=False,
                error_message=error_msg
            )
    
    def _predict_standard(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Standard inference without augmentation"""
        with torch.no_grad():
            if hasattr(self.config, 'mixed_precision') and self.config.mixed_precision:
                with autocast():
                    output = self.model(input_tensor)
            else:
                output = self.model(input_tensor)
        
        return output
    
    def _predict_with_tta(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Test-time augmentation inference"""
        predictions = []
        
        # Original prediction
        pred = self._predict_standard(input_tensor)
        predictions.append(pred)
        
        # Flipped predictions
        for axis in [2, 3, 4]:  # D, H, W axes
            flipped_input = torch.flip(input_tensor, dims=[axis])
            flipped_pred = self._predict_standard(flipped_input)
            unflipped_pred = torch.flip(flipped_pred, dims=[axis])
            predictions.append(unflipped_pred)
        
        # Rotated predictions (90, 180, 270 degrees in axial plane)
        for angle in [1, 2, 3]:  # 90*angle degrees
            rotated_input = torch.rot90(input_tensor, k=angle, dims=[3, 4])
            rotated_pred = self._predict_standard(rotated_input)
            unrotated_pred = torch.rot90(rotated_pred, k=-angle, dims=[3, 4])
            predictions.append(unrotated_pred)
        
        # Stack predictions
        predictions_stack = torch.stack(predictions, dim=0)
        
        # Calculate mean and uncertainty (std)
        mean_pred = torch.mean(predictions_stack, dim=0)
        uncertainty = torch.std(predictions_stack, dim=0)
        
        return mean_pred, uncertainty
    
    def _predict_with_uncertainty(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Monte Carlo Dropout uncertainty estimation"""
        # Enable dropout during inference
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout3d):
                module.train()
        
        predictions = []
        n_samples = getattr(self.config, 'mc_samples', 10)
        
        for _ in range(n_samples):
            pred = self._predict_standard(input_tensor)
            predictions.append(pred)
        
        # Set back to eval mode
        self.model.eval()
        
        # Stack predictions
        predictions_stack = torch.stack(predictions, dim=0)
        
        # Calculate mean and uncertainty
        mean_pred = torch.mean(predictions_stack, dim=0)
        uncertainty = torch.std(predictions_stack, dim=0)
        
        return mean_pred, uncertainty
    
    def _postprocess_prediction(
        self,
        prediction: torch.Tensor,
        preprocessed_data: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Post-process model prediction
        
        Args:
            prediction: Raw model output
            preprocessed_data: Preprocessing metadata
            
        Returns:
            (binary_mask, probability_map)
        """
        # Move to CPU and convert to numpy
        prediction_np = prediction.squeeze().cpu().numpy()
        
        # Apply sigmoid to get probabilities
        probability_map = 1.0 / (1.0 + np.exp(-prediction_np))  # Sigmoid
        
        # Threshold to get binary mask
        binary_mask = (probability_map.astype(np.float32) > float(self.threshold)).astype(np.uint8)
        
        # Remove small components
        if self.min_tumor_size > 0:
            binary_mask = self._remove_small_components(binary_mask, self.min_tumor_size)
        
        # Morphological cleanup
        binary_mask = self._morphological_cleanup(binary_mask)
        
        # Resize back to original shape if needed
        original_shape = preprocessed_data['original_shape']
        if binary_mask.shape != original_shape:
            # Resize binary mask
            zoom_factors = [
                original_shape[i] / binary_mask.shape[i] for i in range(3)
            ]
            binary_mask = zoom(binary_mask.astype(np.float32), zoom_factors, order=0)
            binary_mask = (np.asarray(binary_mask, dtype=np.float32) > 0.5).astype(np.uint8)
            
            # Resize probability map
            probability_map = zoom(probability_map, zoom_factors, order=1)
        
        return binary_mask, probability_map
    
    def _remove_small_components(self, mask: np.ndarray, min_size: int) -> np.ndarray:
        """Remove small connected components"""
        labeled_mask = label(mask)
        cleaned_mask = remove_small_objects(
            labeled_mask, min_size=min_size, connectivity=3
        )
        return (cleaned_mask > 0).astype(np.uint8)
    
    def _morphological_cleanup(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean mask"""
        # Closing to fill small holes
        cleaned_mask = binary_closing(mask, np.ones((3, 3, 3)))
        return cleaned_mask.astype(np.uint8)
    
    def _calculate_tumor_metrics(
        self,
        binary_mask: np.ndarray,
        voxel_spacing: Tuple[float, float, float]
    ) -> Tuple[float, int]:
        """Calculate tumor volume metrics"""
        tumor_voxel_count = int(np.sum(binary_mask))
        voxel_volume_mm3 = np.prod(voxel_spacing)
        tumor_volume_mm3 = float(tumor_voxel_count * voxel_volume_mm3)
        
        return tumor_volume_mm3, tumor_voxel_count
    
    def _calculate_confidence(
        self,
        prediction: torch.Tensor,
        binary_mask: np.ndarray
    ) -> float:
        """Calculate confidence score based on prediction certainty"""
        # Convert prediction to probabilities
        prob_map = torch.sigmoid(prediction).squeeze().cpu().numpy()
        
        if np.sum(binary_mask) == 0:
            return 0.0
        
        # Calculate mean probability in tumor region
        tumor_probs = prob_map[binary_mask > 0]
        confidence = float(np.mean(tumor_probs)) if len(tumor_probs) > 0 else 0.0
        
        return confidence
    
    def _save_outputs(
        self,
        result: InferenceResult,
        output_dir: Union[str, Path],
        preprocessed_data: Dict[str, Any]
    ):
        """Save inference outputs"""
        # Ensure logger is initialized
        if self.logger is None:
            import logging
            self.logger = logging.getLogger("InferenceLogger")
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
                self.logger.addHandler(handler)

        # Ensure output_dir is a Path object
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        reference_img = preprocessed_data['reference_image']

        # Save binary mask
        mask_path = output_dir / f"{result.case_id}_tumor_mask.nii.gz"
        save_nifti(result.prediction_mask, mask_path, reference_img=reference_img)

        # Save probability map
        if result.probability_map is not None:
            prob_path = output_dir / f"{result.case_id}_probability_map.nii.gz"
            save_nifti(result.probability_map, prob_path, reference_img=reference_img)

        # Save uncertainty map
        if result.uncertainty_map is not None:
            umap = result.uncertainty_map
            if isinstance(umap, torch.Tensor):
                uncertainty_np = umap.detach().squeeze().cpu().numpy()
            else:
                uncertainty_np = np.asarray(umap).squeeze()
            uncertainty_path = output_dir / f"{result.case_id}_uncertainty_map.nii.gz"
            save_nifti(uncertainty_np, uncertainty_path, reference_img=reference_img)

        # Save metadata
        metadata_path = output_dir / f"{result.case_id}_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            metadata = {
                'case_id': result.case_id,
                'tumor_volume_mm3': result.tumor_volume_mm3,
                'tumor_voxel_count': result.tumor_voxel_count,
                'confidence_score': result.confidence_score,
                'processing_time': result.processing_time,
                'metadata': result.metadata
            }
            json.dump(metadata, f, indent=2)
    
    def predict_batch(
        self,
        cases: List[Dict[str, Union[str, Path]]],
        case_ids: Optional[List[str]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        max_workers: int = 1,
        save_outputs: bool = True
    ) -> List[InferenceResult]:
        """
        Run batch inference on multiple cases
        
        Args:
            cases: List of case dictionaries with modality paths
            case_ids: Optional list of case identifiers
            output_dir: Directory to save outputs
            max_workers: Number of parallel workers (1 for sequential)
            save_outputs: Whether to save output files
            
        Returns:
            List of InferenceResult objects
        """
        if case_ids is None:
            case_ids = [f"case_{i:03d}" for i in range(len(cases))]
        
        if len(case_ids) != len(cases):
            raise ValueError("Number of case_ids must match number of cases")
        
        if self.verbose:
            self.logger.info(f"Starting batch inference on {len(cases)} cases")
        
        results = []
        start_time = time.time()
        
        if max_workers == 1:
            # Sequential processing
            for i, (case, case_id) in enumerate(zip(cases, case_ids)):
                if self.verbose:
                    self.logger.info(f"Processing case {i+1}/{len(cases)}: {case_id}")
                
                result = self.predict_single(
                    volume_paths=case,
                    case_id=case_id,
                    save_outputs=save_outputs,
                    output_dir=output_dir
                )
                results.append(result)
        else:
            # Parallel processing (requires careful memory management)
            warnings.warn("Parallel inference may cause GPU memory issues. Consider using max_workers=1.")
            
            def process_case(args):
                case, case_id = args
                return self.predict_single(
                    volume_paths=case,
                    case_id=case_id,
                    save_outputs=save_outputs,
                    output_dir=output_dir
                )
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(process_case, zip(cases, case_ids)))
        
        # Summary statistics
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_time = time.time() - start_time
        
        if self.verbose:
            self.logger.info(f"\nBatch inference completed:")
            self.logger.info(f"  Total cases: {len(cases)}")
            self.logger.info(f"  Successful: {successful}")
            self.logger.info(f"  Failed: {failed}")
            self.logger.info(f"  Total time: {format_time(total_time)}")
            self.logger.info(f"  Average time per case: {format_time(total_time / len(cases))}")
        
        return results


class SlidingWindowInference:
    """
    Sliding window inference for very large volumes
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        window_size: Tuple[int, int, int] = (128, 128, 128),
        overlap: float = 0.25,
        device: torch.device = torch.device('cpu')
    ):
        self.model = model
        self.window_size = window_size
        self.overlap = overlap
        self.device = device
        
        # Calculate stride
        self.stride = tuple(int(ws * (1 - overlap)) for ws in window_size)
    
    def predict(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Run sliding window inference
        
        Args:
            volume: Input volume tensor (1, C, D, H, W)
            
        Returns:
            Prediction tensor (1, 1, D, H, W)
        """
        batch_size, channels, depth, height, width = volume.shape
        
        # Initialize output volume
        prediction = torch.zeros(
            (batch_size, 1, depth, height, width),
            dtype=torch.float32,
            device=self.device
        )
        count_map = torch.zeros_like(prediction)
        
        # Generate window positions
        d_positions = self._get_positions(depth, self.window_size[0], self.stride[0])
        h_positions = self._get_positions(height, self.window_size[1], self.stride[1])
        w_positions = self._get_positions(width, self.window_size[2], self.stride[2])
        
        # Process each window
        total_windows = len(d_positions) * len(h_positions) * len(w_positions)
        window_idx = 0
        
        for d_start, d_end in d_positions:
            for h_start, h_end in h_positions:
                for w_start, w_end in w_positions:
                    window_idx += 1
                    
                    # Extract window
                    window = volume[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                    
                    # Predict
                    with torch.no_grad():
                        window_pred = self.model(window)
                    
                    # Add to prediction
                    prediction[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += window_pred
                    count_map[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += 1
        
        # Average overlapping regions
        prediction = prediction / torch.clamp(count_map, min=1)
        
        return prediction
    
    def _get_positions(self, total_size: int, window_size: int, stride: int) -> List[Tuple[int, int]]:
        """Get sliding window positions"""
        positions = []
        
        for start in range(0, total_size - window_size + 1, stride):
            end = start + window_size
            positions.append((start, end))
        
        # Ensure we cover the entire volume
        if positions[-1][1] < total_size:
            positions.append((total_size - window_size, total_size))
        
        return positions


def create_inference_engine(
    model_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    device: str = "auto",
    use_tta: bool = False,
    use_uncertainty: bool = False
) -> ModelInference:
    """
    Factory function to create inference engine
    
    Args:
        model_path: Path to trained model checkpoint
        config_path: Optional path to configuration file
        device: Inference device
        use_tta: Enable test-time augmentation
        use_uncertainty: Enable uncertainty estimation
        
    Returns:
        ModelInference instance
    """
    # Load config
    if config_path:
        config = Config.from_file(config_path)
    else:
        config = Config()
    
    # Get device
    inference_device = get_device(device)
    
    # Create inference engine
    inference_engine = ModelInference(
        model_path=model_path,
        config=config,
        device=inference_device,
        use_tta=use_tta,
        use_uncertainty=use_uncertainty
    )
    
    return inference_engine


def main():
    """Main inference script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run 3D Brain Tumor Segmentation Inference")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--input", "-i", type=str, help="Input scan directory or file paths")
    parser.add_argument("--output", "-o", type=str, default="./inference_outputs", help="Output directory")
    parser.add_argument("--config", "-c", type=str, help="Configuration file path")
    parser.add_argument("--device", default="auto", help="Inference device")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    parser.add_argument("--uncertainty", action="store_true", help="Enable uncertainty estimation")
    parser.add_argument("--batch", action="store_true", help="Batch processing mode")
    
    # Modality arguments
    parser.add_argument("--t1", type=str, help="T1 modality path")
    parser.add_argument("--t1ce", type=str, help="T1CE modality path")
    parser.add_argument("--t2", type=str, help="T2 modality path")
    parser.add_argument("--flair", type=str, help="FLAIR modality path")
    
    args = parser.parse_args()
    
    # Create inference engine
    inference_engine = create_inference_engine(
        model_path=args.model,
        config_path=args.config,
        device=args.device,
        use_tta=args.tta,
        use_uncertainty=args.uncertainty
    )
    
    if args.batch:
        # Batch processing mode
        if not args.input:
            raise ValueError("Input directory required for batch processing")
        
        input_dir = Path(args.input)
        cases = []
        case_ids = []
        
        # Discover cases in input directory
        for case_dir in input_dir.iterdir():
            if case_dir.is_dir():
                modalities = {}
                # Look for modality files
                for mod in ['t1', 't1ce', 't2', 'flair']:
                    for pattern in [f"*{mod}*.nii.gz", f"*{mod.upper()}*.nii.gz"]:
                        files = list(case_dir.glob(pattern))
                        if files:
                            modalities[mod] = files[0]
                            break
                
                if modalities:
                    cases.append(modalities)
                    case_ids.append(case_dir.name)
        
        # Run batch inference
        results = inference_engine.predict_batch(
            cases=cases,
            case_ids=case_ids,
            output_dir=args.output,
            save_outputs=True
        )
        
        print(f"Processed {len(results)} cases")
        
    else:
        # Single case processing
        modalities = {}
        
        if args.t1:
            modalities['t1'] = args.t1
        if args.t1ce:
            modalities['t1ce'] = args.t1ce
        if args.t2:
            modalities['t2'] = args.t2
        if args.flair:
            modalities['flair'] = args.flair
        
        if not modalities:
            raise ValueError("At least one modality must be specified")
        
        # Run single inference
        result = inference_engine.predict_single(
            volume_paths=modalities,
            case_id="inference_case",
            save_outputs=True,
            output_dir=args.output
        )
        
        if result.success:
            print(f"Inference completed successfully!")
            print(f"Tumor volume: {result.tumor_volume_mm3:.1f} mm³")
            print(f"Confidence: {result.confidence_score:.3f}")
            print(f"Processing time: {format_time(result.processing_time)}")
        else:
            print(f"Inference failed: {result.error_message}")


if __name__ == "__main__":
    main()