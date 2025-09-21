"""
Comprehensive 3D Evaluation Metrics for Brain Tumor Segmentation

Advanced metrics suite for evaluating segmentation quality including volumetric,
surface-based, and clinical metrics. Supports both binary and multi-class evaluation.

Key Features:
- Volumetric metrics (Dice, IoU, Sensitivity, Specificity)
- Surface-based metrics (Hausdorff Distance, Average Surface Distance)
- Clinical metrics (Volume difference, Lesion detection)
- Multi-class evaluation (Whole tumor, Core, Enhancing)
- Statistical analysis and confidence intervals
- BraTS challenge compatible evaluation
"""

import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from collections import defaultdict
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import binary_erosion, binary_dilation, label
from skimage.measure import regionprops, label
from skimage.segmentation import find_boundaries

from utils_3d import Config, format_time, timer


@dataclass
class SegmentationMetrics:
    """Container for segmentation metrics"""
    dice_score: float
    jaccard_index: float
    sensitivity: float
    specificity: float
    precision: float
    hausdorff_distance: float
    avg_surface_distance: float
    volume_difference: float
    relative_volume_difference: float
    lesion_detection_rate: float
    false_positive_rate: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'dice_score': self.dice_score,
            'jaccard_index': self.jaccard_index,
            'sensitivity': self.sensitivity,
            'specificity': self.specificity,
            'precision': self.precision,
            'hausdorff_distance': self.hausdorff_distance,
            'avg_surface_distance': self.avg_surface_distance,
            'volume_difference': self.volume_difference,
            'relative_volume_difference': self.relative_volume_difference,
            'lesion_detection_rate': self.lesion_detection_rate,
            'false_positive_rate': self.false_positive_rate
        }


@dataclass
class MultiClassMetrics:
    """Container for multi-class evaluation metrics"""
    whole_tumor: SegmentationMetrics
    tumor_core: Optional[SegmentationMetrics] = None
    enhancing_tumor: Optional[SegmentationMetrics] = None
    
    def to_dict(self) -> Dict[str, Dict[str, float]]:
        result = {'whole_tumor': self.whole_tumor.to_dict()}
        if self.tumor_core:
            result['tumor_core'] = self.tumor_core.to_dict()
        if self.enhancing_tumor:
            result['enhancing_tumor'] = self.enhancing_tumor.to_dict()
        return result


class MetricsCalculator:
    """
    Comprehensive metrics calculator for 3D segmentation evaluation
    
    Args:
        voxel_spacing: Voxel spacing in mm (D, H, W)
        connectivity: Connectivity for connected components (6, 18, 26)
        percentile: Percentile for robust Hausdorff distance calculation
    """
    
    def __init__(
        self,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        connectivity: int = 26,
        percentile: float = 95.0
    ):
        self.voxel_spacing = voxel_spacing
        self.connectivity = connectivity
        self.percentile = percentile
        
        # Precompute voxel volume
        self.voxel_volume_mm3 = np.prod(voxel_spacing)
    
    def calculate_binary_metrics(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        empty_prediction_score: float = 0.0
    ) -> SegmentationMetrics:
        """
        Calculate comprehensive metrics for binary segmentation
        
        Args:
            prediction: Binary prediction mask (0, 1)
            ground_truth: Binary ground truth mask (0, 1)
            empty_prediction_score: Score to assign when prediction is empty
            
        Returns:
            SegmentationMetrics object
        """
        # Ensure binary masks
        pred_binary = (prediction > 0).astype(np.uint8)
        gt_binary = (ground_truth > 0).astype(np.uint8)
        
        # Handle edge cases
        pred_sum = np.sum(pred_binary)
        gt_sum = np.sum(gt_binary)
        
        if gt_sum == 0 and pred_sum == 0:
            # Both empty - perfect score
            return self._create_perfect_metrics()
        elif gt_sum == 0 and pred_sum > 0:
            # False positive case
            return self._create_false_positive_metrics(pred_binary)
        elif gt_sum > 0 and pred_sum == 0:
            # Empty prediction case
            return self._create_empty_prediction_metrics(gt_binary, empty_prediction_score)
        
        # Calculate volumetric metrics
        dice = self._calculate_dice_coefficient(pred_binary, gt_binary)
        jaccard = self._calculate_jaccard_index(pred_binary, gt_binary)
        sensitivity = self._calculate_sensitivity(pred_binary, gt_binary)
        specificity = self._calculate_specificity(pred_binary, gt_binary)
        precision = self._calculate_precision(pred_binary, gt_binary)
        
        # Calculate surface-based metrics
        hausdorff_dist = self._calculate_hausdorff_distance(pred_binary, gt_binary)
        avg_surface_dist = self._calculate_average_surface_distance(pred_binary, gt_binary)
        
        # Calculate volume metrics
        volume_diff = self._calculate_volume_difference(pred_binary, gt_binary)
        rel_volume_diff = self._calculate_relative_volume_difference(pred_binary, gt_binary)
        
        # Calculate detection metrics
        lesion_detection = self._calculate_lesion_detection_rate(pred_binary, gt_binary)
        fp_rate = self._calculate_false_positive_rate(pred_binary, gt_binary)
        
        return SegmentationMetrics(
            dice_score=dice,
            jaccard_index=jaccard,
            sensitivity=sensitivity,
            specificity=specificity,
            precision=precision,
            hausdorff_distance=hausdorff_dist,
            avg_surface_distance=avg_surface_dist,
            volume_difference=volume_diff,
            relative_volume_difference=rel_volume_diff,
            lesion_detection_rate=lesion_detection,
            false_positive_rate=fp_rate
        )
    
    def calculate_multiclass_metrics(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        class_mapping: Optional[Dict[str, List[int]]] = None
    ) -> MultiClassMetrics:
        """
        Calculate metrics for multi-class BraTS evaluation
        
        Args:
            prediction: Multi-class prediction (0=bg, 1=necrotic, 2=edema, 4=enhancing)
            ground_truth: Multi-class ground truth with same labels
            class_mapping: Custom class mapping for evaluation regions
            
        Returns:
            MultiClassMetrics object
        """
        if class_mapping is None:
            # Standard BraTS evaluation regions
            class_mapping = {
                'whole_tumor': [1, 2, 4],    # All tumor classes
                'tumor_core': [1, 4],        # Necrotic + Enhancing
                'enhancing_tumor': [4]       # Only enhancing
            }
        
        results = {}
        
        for region_name, label_list in class_mapping.items():
            # Create binary masks for this region
            pred_binary = np.isin(prediction, label_list).astype(np.uint8)
            gt_binary = np.isin(ground_truth, label_list).astype(np.uint8)
            
            # Calculate metrics
            metrics = self.calculate_binary_metrics(pred_binary, gt_binary)
            results[region_name] = metrics
        
        return MultiClassMetrics(
            whole_tumor=results['whole_tumor'],
            tumor_core=results.get('tumor_core'),
            enhancing_tumor=results.get('enhancing_tumor')
        )
    
    def _calculate_dice_coefficient(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate Dice similarity coefficient"""
        intersection = np.sum(pred * gt)
        union = np.sum(pred) + np.sum(gt)
        
        if union == 0:
            return 1.0  # Both empty
        
        dice = (2.0 * intersection) / union
        return float(dice)
    
    def _calculate_jaccard_index(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate Jaccard index (IoU)"""
        intersection = np.sum(pred * gt)
        union = np.sum(pred) + np.sum(gt) - intersection
        
        if union == 0:
            return 1.0  # Both empty
        
        jaccard = intersection / union
        return float(jaccard)
    
    def _calculate_sensitivity(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate sensitivity (recall, true positive rate)"""
        true_positive = np.sum(pred * gt)
        false_negative = np.sum((1 - pred) * gt)
        
        if true_positive + false_negative == 0:
            return 1.0  # No positive ground truth
        
        sensitivity = true_positive / (true_positive + false_negative)
        return float(sensitivity)
    
    def _calculate_specificity(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate specificity (true negative rate)"""
        true_negative = np.sum((1 - pred) * (1 - gt))
        false_positive = np.sum(pred * (1 - gt))
        
        if true_negative + false_positive == 0:
            return 1.0  # No negative ground truth
        
        specificity = true_negative / (true_negative + false_positive)
        return float(specificity)
    
    def _calculate_precision(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate precision (positive predictive value)"""
        true_positive = np.sum(pred * gt)
        false_positive = np.sum(pred * (1 - gt))
        
        if true_positive + false_positive == 0:
            return 1.0  # No positive predictions
        
        precision = true_positive / (true_positive + false_positive)
        return float(precision)
    
    def _calculate_hausdorff_distance(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate Hausdorff distance between surfaces
        
        Returns distance in mm
        """
        try:
            # Get surface points
            pred_surface = self._get_surface_points(pred)
            gt_surface = self._get_surface_points(gt)
            
            if len(pred_surface) == 0 or len(gt_surface) == 0:
                return float('inf')
            
            # Calculate directed Hausdorff distances
            dist1 = directed_hausdorff(pred_surface, gt_surface)[0]
            dist2 = directed_hausdorff(gt_surface, pred_surface)[0]
            
            # Use specified percentile for robustness
            if self.percentile < 100:
                # Calculate percentile-based Hausdorff distance
                dist1_percentile = self._percentile_hausdorff(pred_surface, gt_surface)
                dist2_percentile = self._percentile_hausdorff(gt_surface, pred_surface)
                hausdorff_dist = max(dist1_percentile, dist2_percentile)
            else:
                hausdorff_dist = max(dist1, dist2)
            
            # Convert to mm using voxel spacing
            hausdorff_dist_mm = hausdorff_dist * np.mean(self.voxel_spacing)
            
            return float(hausdorff_dist_mm)
            
        except Exception:
            return float('inf')
    
    def _calculate_average_surface_distance(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate average surface distance
        
        Returns distance in mm
        """
        try:
            # Get surface points
            pred_surface = self._get_surface_points(pred)
            gt_surface = self._get_surface_points(gt)
            
            if len(pred_surface) == 0 or len(gt_surface) == 0:
                return float('inf')
            
            # Calculate distances from each prediction surface point to GT surface
            from scipy.spatial.distance import cdist
            distances = cdist(pred_surface, gt_surface)
            min_distances_pred_to_gt = np.min(distances, axis=1)
            
            # Calculate distances from each GT surface point to prediction surface
            min_distances_gt_to_pred = np.min(distances, axis=0)
            
            # Average surface distance
            avg_dist = (np.mean(min_distances_pred_to_gt) + np.mean(min_distances_gt_to_pred)) / 2
            
            # Convert to mm
            avg_dist_mm = avg_dist * np.mean(self.voxel_spacing)
            
            return float(avg_dist_mm)
            
        except Exception:
            return float('inf')
    
    def _get_surface_points(self, binary_mask: np.ndarray) -> np.ndarray:
        """Extract surface points from binary mask"""
        # Find boundaries
        boundaries = find_boundaries(binary_mask, mode='inner')
        surface_points = np.array(np.where(boundaries)).T
        
        return surface_points
    
    def _percentile_hausdorff(self, points1: np.ndarray, points2: np.ndarray) -> float:
        """Calculate percentile-based Hausdorff distance"""
        from scipy.spatial.distance import cdist
        
        distances = cdist(points1, points2)
        min_distances = np.min(distances, axis=1)
        percentile_dist = np.percentile(min_distances, self.percentile)
        
        return float(percentile_dist)
    
    def _calculate_volume_difference(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate absolute volume difference in mm³"""
        pred_volume = np.sum(pred) * self.voxel_volume_mm3
        gt_volume = np.sum(gt) * self.voxel_volume_mm3
        
        volume_diff = abs(pred_volume - gt_volume)
        return float(volume_diff)
    
    def _calculate_relative_volume_difference(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate relative volume difference as percentage"""
        pred_volume = np.sum(pred)
        gt_volume = np.sum(gt)
        
        if gt_volume == 0:
            return 100.0 if pred_volume > 0 else 0.0
        
        rel_diff = abs(pred_volume - gt_volume) / gt_volume * 100
        return float(rel_diff)
    
    def _calculate_lesion_detection_rate(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate lesion detection rate (percentage of GT lesions detected)"""
        # Label connected components in ground truth
        gt_labeled, num_gt_lesions = label(gt)
        
        if num_gt_lesions == 0:
            return 1.0  # No lesions to detect
        
        # Check which GT lesions are detected
        detected_lesions = 0
        
        for lesion_id in range(1, num_gt_lesions + 1):
            lesion_mask = (gt_labeled == lesion_id)
            if np.any(pred * lesion_mask):  # Prediction overlaps with this lesion
                detected_lesions += 1
        
        detection_rate = detected_lesions / num_gt_lesions
        return float(detection_rate)
    
    def _calculate_false_positive_rate(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate false positive rate (FP lesions per volume)"""
        # Find prediction regions that don't overlap with ground truth
        fp_mask = pred * (1 - gt)
        
        # Label connected components in false positive regions
        fp_labeled, num_fp_lesions = label(fp_mask)
        
        # Calculate volume in cm³ for rate calculation
        total_volume_cm3 = np.prod(pred.shape) * self.voxel_volume_mm3 / 1000
        
        fp_rate = num_fp_lesions / total_volume_cm3 if total_volume_cm3 > 0 else 0.0
        return float(fp_rate)
    
    def _create_perfect_metrics(self) -> SegmentationMetrics:
        """Create metrics object for perfect segmentation (both empty)"""
        return SegmentationMetrics(
            dice_score=1.0,
            jaccard_index=1.0,
            sensitivity=1.0,
            specificity=1.0,
            precision=1.0,
            hausdorff_distance=0.0,
            avg_surface_distance=0.0,
            volume_difference=0.0,
            relative_volume_difference=0.0,
            lesion_detection_rate=1.0,
            false_positive_rate=0.0
        )
    
    def _create_false_positive_metrics(self, pred: np.ndarray) -> SegmentationMetrics:
        """Create metrics object for false positive case (empty GT, non-empty pred)"""
        return SegmentationMetrics(
            dice_score=0.0,
            jaccard_index=0.0,
            sensitivity=0.0,  # No true positives possible
            specificity=0.0,  # All negatives are false positives
            precision=0.0,    # All predictions are false positives
            hausdorff_distance=float('inf'),
            avg_surface_distance=float('inf'),
            volume_difference=float(np.sum(pred) * self.voxel_volume_mm3),
            relative_volume_difference=100.0,
            lesion_detection_rate=1.0,  # No lesions to miss
            false_positive_rate=float('inf')
        )
    
    def _create_empty_prediction_metrics(
        self, 
        gt: np.ndarray, 
        empty_score: float
    ) -> SegmentationMetrics:
        """Create metrics object for empty prediction case"""
        return SegmentationMetrics(
            dice_score=empty_score,
            jaccard_index=empty_score,
            sensitivity=0.0,  # No true positives
            specificity=1.0,  # No false positives
            precision=1.0,    # No predictions to be wrong
            hausdorff_distance=float('inf'),
            avg_surface_distance=float('inf'),
            volume_difference=float(np.sum(gt) * self.voxel_volume_mm3),
            relative_volume_difference=100.0,
            lesion_detection_rate=0.0,
            false_positive_rate=0.0
        )


class BraTSEvaluator:
    """
    BraTS challenge compatible evaluator
    
    Implements evaluation protocol used in BraTS challenges
    """
    
    def __init__(self, voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.metrics_calculator = MetricsCalculator(voxel_spacing)
    
    def evaluate_case(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        case_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Evaluate a single case using BraTS protocol
        
        Args:
            prediction: Prediction mask (0, 1, 2, 4)
            ground_truth: Ground truth mask (0, 1, 2, 4)
            case_id: Case identifier
            
        Returns:
            Dictionary with BraTS metrics
        """
        # Calculate multi-class metrics
        multiclass_metrics = self.metrics_calculator.calculate_multiclass_metrics(
            prediction, ground_truth
        )
        
        # Format results according to BraTS convention
        result = {
            'case_id': case_id,
            'whole_tumor': {
                'dice': multiclass_metrics.whole_tumor.dice_score,
                'hausdorff95': multiclass_metrics.whole_tumor.hausdorff_distance,
                'sensitivity': multiclass_metrics.whole_tumor.sensitivity,
                'specificity': multiclass_metrics.whole_tumor.specificity
            }
        }
        
        if multiclass_metrics.tumor_core:
            result['tumor_core'] = {
                'dice': multiclass_metrics.tumor_core.dice_score,
                'hausdorff95': multiclass_metrics.tumor_core.hausdorff_distance,
                'sensitivity': multiclass_metrics.tumor_core.sensitivity,
                'specificity': multiclass_metrics.tumor_core.specificity
            }
        
        if multiclass_metrics.enhancing_tumor:
            result['enhancing_tumor'] = {
                'dice': multiclass_metrics.enhancing_tumor.dice_score,
                'hausdorff95': multiclass_metrics.enhancing_tumor.hausdorff_distance,
                'sensitivity': multiclass_metrics.enhancing_tumor.sensitivity,
                'specificity': multiclass_metrics.enhancing_tumor.specificity
            }
        
        return result
    
    def evaluate_dataset(
        self,
        predictions: List[np.ndarray],
        ground_truths: List[np.ndarray],
        case_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate entire dataset
        
        Args:
            predictions: List of prediction masks
            ground_truths: List of ground truth masks
            case_ids: Optional list of case identifiers
            
        Returns:
            Dataset evaluation results with statistics
        """
        if case_ids is None:
            case_ids = [f"case_{i:03d}" for i in range(len(predictions))]
        
        if len(predictions) != len(ground_truths) or len(predictions) != len(case_ids):
            raise ValueError("Predictions, ground truths, and case IDs must have same length")
        
        # Evaluate each case
        case_results = []
        for pred, gt, case_id in zip(predictions, ground_truths, case_ids):
            case_result = self.evaluate_case(pred, gt, case_id)
            case_results.append(case_result)
        
        # Aggregate statistics
        dataset_stats = self._calculate_dataset_statistics(case_results)
        
        return {
            'case_results': case_results,
            'dataset_statistics': dataset_stats,
            'num_cases': len(case_results)
        }
    
    def _calculate_dataset_statistics(self, case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate dataset-level statistics"""
        stats = defaultdict(lambda: defaultdict(list))
        
        # Collect metrics for each region
        for case in case_results:
            for region in ['whole_tumor', 'tumor_core', 'enhancing_tumor']:
                if region in case:
                    for metric in ['dice', 'hausdorff95', 'sensitivity', 'specificity']:
                        if metric in case[region]:
                            stats[region][metric].append(case[region][metric])
        
        # Calculate statistics
        dataset_stats = {}
        for region, metrics in stats.items():
            region_stats = {}
            for metric_name, values in metrics.items():
                if values:  # Only if we have values
                    region_stats[metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'median': float(np.median(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'count': len(values)
                    }
            dataset_stats[region] = region_stats
        
        return dataset_stats


class MetricsBenchmark:
    """
    Benchmarking utilities for metrics calculation performance
    """
    
    def __init__(self, voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.calculator = MetricsCalculator(voxel_spacing)
    
    def benchmark_metrics(
        self,
        volume_sizes: List[Tuple[int, int, int]] = [(64, 64, 64), (128, 128, 128), (256, 256, 256)],
        num_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark metrics calculation performance
        
        Args:
            volume_sizes: List of volume sizes to test
            num_iterations: Number of iterations for timing
            
        Returns:
            Benchmark results
        """
        results = {}
        
        for size in volume_sizes:
            size_key = f"{size[0]}x{size[1]}x{size[2]}"
            
            # Generate test data
            pred = np.random.rand(*size) > 0.7
            gt = np.random.rand(*size) > 0.7
            
            # Time metrics calculation
            times = []
            for _ in range(num_iterations):
                start_time = time.time()
                metrics = self.calculator.calculate_binary_metrics(pred, gt)
                end_time = time.time()
                times.append(end_time - start_time)
            
            results[size_key] = {
                'volume_size': size,
                'voxel_count': np.prod(size),
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times)
            }
        
        return results


def calculate_batch_metrics(
    predictions: List[np.ndarray],
    ground_truths: List[np.ndarray],
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    multiclass: bool = False,
    case_ids: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Calculate metrics for a batch of predictions
    
    Args:
        predictions: List of prediction arrays
        ground_truths: List of ground truth arrays
        voxel_spacing: Voxel spacing for distance calculations
        multiclass: Whether to use multiclass evaluation
        case_ids: Optional case identifiers
        verbose: Enable verbose output
        
    Returns:
        Dictionary with batch metrics and statistics
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Number of predictions must match number of ground truths")
    
    if case_ids and len(case_ids) != len(predictions):
        raise ValueError("Number of case IDs must match number of predictions")
    
    if case_ids is None:
        case_ids = [f"case_{i:03d}" for i in range(len(predictions))]
    
    if verbose:
        print(f"Calculating metrics for {len(predictions)} cases...")
    
    if multiclass:
        # Use BraTS evaluator for multiclass
        evaluator = BraTSEvaluator(voxel_spacing)
        results = evaluator.evaluate_dataset(predictions, ground_truths, case_ids)
    else:
        # Binary evaluation
        calculator = MetricsCalculator(voxel_spacing)
        case_results = []
        
        for i, (pred, gt, case_id) in enumerate(zip(predictions, ground_truths, case_ids)):
            if verbose and i % 10 == 0:
                print(f"Processing case {i+1}/{len(predictions)}")
            
            metrics = calculator.calculate_binary_metrics(pred, gt)
            case_result = {
                'case_id': case_id,
                'metrics': metrics.to_dict()
            }
            case_results.append(case_result)
        
        # Calculate aggregate statistics
        all_metrics = [case['metrics'] for case in case_results]
        aggregate_stats = {}
        
        for metric_name in all_metrics[0].keys():
            values = [m[metric_name] for m in all_metrics if not np.isinf(m[metric_name])]
            if values:
                aggregate_stats[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
        
        results = {
            'case_results': case_results,
            'aggregate_statistics': aggregate_stats,
            'num_cases': len(case_results)
        }
    
    if verbose:
        print("Metrics calculation completed!")
        
        # Print summary
        if multiclass and 'dataset_statistics' in results:
            stats = results['dataset_statistics']
            for region in ['whole_tumor', 'tumor_core', 'enhancing_tumor']:
                if region in stats and 'dice' in stats[region]:
                    dice_mean = stats[region]['dice']['mean']
                    print(f"{region.replace('_', ' ').title()}: Dice = {dice_mean:.3f}")
        elif not multiclass and 'aggregate_statistics' in results:
            dice_mean = results['aggregate_statistics']['dice_score']['mean']
            print(f"Overall Dice Score: {dice_mean:.3f}")
    
    return results


def main():
    """Example usage and testing"""
    print("Testing 3D segmentation metrics...")
    
    # Create test data
    volume_size = (64, 64, 64)
    pred = (np.random.rand(*volume_size) > 0.8).astype(np.uint8)
    gt = (np.random.rand(*volume_size) > 0.8).astype(np.uint8)
    
    # Test binary metrics
    calculator = MetricsCalculator(voxel_spacing=(1.0, 1.0, 1.0))
    
    with timer("Binary metrics calculation"):
        binary_metrics = calculator.calculate_binary_metrics(pred, gt)
    
    print("Binary Metrics:")
    for key, value in binary_metrics.to_dict().items():
        if np.isinf(value):
            print(f"  {key}: inf")
        else:
            print(f"  {key}: {value:.4f}")