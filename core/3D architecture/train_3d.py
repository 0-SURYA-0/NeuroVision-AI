"""
3D Training Pipeline for Brain Tumor Segmentation

Complete training pipeline with mixed precision, advanced loss functions,
learning rate scheduling, early stopping, and comprehensive logging.

Key Features:
- Mixed precision training (AMP) for RTX 4060
- Advanced loss functions (Dice, Focal, Combo)
- Learning rate scheduling with warm-up
- Early stopping and checkpointing
- Real-time monitoring and visualization
- Gradient clipping and regularization
- Memory-efficient training
"""

import os
import sys
import time
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.sgd import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import numpy as np

# Disable TensorBoard to avoid protobuf conflicts
TENSORBOARD_AVAILABLE = False
SummaryWriter = None

# # Optional TensorBoard import (disable due to protobuf compatibility issues)
# try:
#     from torch.utils.tensorboard.writer import SummaryWriter
#     TENSORBOARD_AVAILABLE = True
# except ImportError:
#     TENSORBOARD_AVAILABLE = False
#     SummaryWriter = None

# Import our modules
from model_3d import UNet3D, create_unet3d
from dataset_3d import BraTSDataset, create_data_loaders, DataAugmentation3D
from utils_3d import Config, setup_logging, set_random_seed, get_device, monitor_memory, timer, format_time


@dataclass
class TrainingMetrics:
    """Container for training metrics"""
    epoch: int
    train_loss: float
    train_dice: float
    val_loss: float
    val_dice: float
    learning_rate: float
    training_time: float
    memory_usage: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks
    """
    
    def __init__(self, smooth: float = 1e-5, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, D, H, W) - raw logits
            targets: (B, C, D, H, W) - binary targets
        """
        # Apply sigmoid to predictions
        predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        pred_flat = predictions.view(predictions.size(0), -1)
        target_flat = targets.view(targets.size(0), -1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Convert to loss (1 - Dice)
        dice_loss = 1.0 - dice
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, D, H, W) - raw logits
            targets: (B, C, D, H, W) - binary targets
        """
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )
        
        # Calculate probabilities
        probs = torch.sigmoid(predictions)
        
        # Calculate focal weight
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ComboLoss(nn.Module):
    """
    Combination of Dice Loss and Focal Loss
    """
    
    def __init__(
        self,
        dice_weight: float = 0.7,
        focal_weight: float = 0.3,
        dice_smooth: float = 1e-5,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(predictions, targets)
        focal = self.focal_loss(predictions, targets)
        
        return self.dice_weight * dice + self.focal_weight * focal


def calculate_dice_score(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Calculate Dice score for evaluation
    
    Args:
        predictions: (B, C, D, H, W) - raw logits or probabilities
        targets: (B, C, D, H, W) - binary targets
        threshold: Threshold for binarizing predictions
        
    Returns:
        Dice score tensor
    """
    # Apply sigmoid if needed
    if predictions.min() < 0 or predictions.max() > 1:
        predictions = torch.sigmoid(predictions)
    
    # Binarize predictions
    predictions_binary = (predictions > threshold).float()
    
    # Flatten tensors
    pred_flat = predictions_binary.view(predictions_binary.size(0), -1)
    target_flat = targets.view(targets.size(0), -1)
    
    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    
    # Calculate Dice coefficient
    smooth = 1e-5
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return dice


class EarlyStopping:
    """
    Early stopping utility
    """
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-4, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.wait = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score):
            self.best_score = score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def _is_better(self, score: float) -> bool:
        if self.best_score is None or self.min_delta is None:
            return False
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


class ModelTrainer:
    """
    Main training class for 3D brain tumor segmentation
    
    Args:
        config: Configuration object
        model: 3D U-Net model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Training device
        logger: Logger instance
    """
    
    def __init__(
        self,
        config: Config,
        model: nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        logger
    ):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logger
        
        # Setup training components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_function()
        self._setup_mixed_precision()
        self._setup_monitoring()
        
        # Training state
        self.current_epoch = 0
        self.best_val_dice = 0.0
        self.best_val_loss = float('inf')
        self.training_history = []
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
            mode='max'  # We want to maximize Dice score
        )
    
    def _setup_optimizer(self):
        """Setup optimizer"""
        if hasattr(self.config, 'optimizer_type'):
            optimizer_type = self.config.optimizer_type.lower()
        else:
            optimizer_type = 'adamw'
        
        if optimizer_type == 'adamw':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type == 'adam':
            self.optimizer = Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif optimizer_type == 'sgd':
            self.optimizer = SGD(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        self.logger.info(f"Optimizer: {type(self.optimizer).__name__}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        scheduler_type = getattr(self.config, 'scheduler_type', 'cosine')
        
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=int(self.config.learning_rate * 0.01)
            )
        elif scheduler_type == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.num_epochs,
                steps_per_epoch=len(self.train_loader),
                pct_start=0.1,
                anneal_strategy='cos'
            )
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                threshold=1e-4,
                min_lr=self.config.learning_rate * 0.001
            )
        else:
            self.scheduler = None
        
        self.logger.info(f"Scheduler: {scheduler_type}")
    
    def _setup_loss_function(self):
        """Setup loss function"""
        loss_type = self.config.loss_function.lower()
        
        if loss_type == 'dice':
            self.criterion = DiceLoss(smooth=self.config.smooth_factor)
        elif loss_type == 'focal':
            self.criterion = FocalLoss(alpha=1.0, gamma=2.0)
        elif loss_type == 'combo':
            self.criterion = ComboLoss()
        elif loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")
        
        self.logger.info(f"Loss function: {type(self.criterion).__name__}")
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training"""
        self.use_amp = self.config.mixed_precision and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
            self.logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
    
    def _setup_monitoring(self):
        """Setup monitoring and logging"""
        # TensorBoard (optional)
        if TENSORBOARD_AVAILABLE and SummaryWriter is not None:
            log_dir = Path(self.config.log_dir) / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        
        # Checkpoints directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_dice = 0.0
        num_batches = 0
        
        epoch_start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            # Move data to device
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True) if targets is not None else None
            
            if targets is None:
                continue  # Skip if no targets (test data)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass
                if self.scaler is None:
                    self.scaler = torch.cuda.amp.GradScaler()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if hasattr(self.config, 'max_grad_norm') and self.config.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if hasattr(self.config, 'max_grad_norm') and self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
            
            # Update scheduler (if OneCycleLR)
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
            
            # Calculate metrics
            with torch.no_grad():
                dice_score = calculate_dice_score(outputs, targets).mean()
            
            total_loss += loss.item()
            total_dice += dice_score.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 50 == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}: "
                    f"Loss={loss.item():.4f}, Dice={dice_score.item():.4f}"
                )
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_dice = total_dice / num_batches if num_batches > 0 else 0.0
        epoch_time = time.time() - epoch_start_time
        
        return {
            'loss': avg_loss,
            'dice': avg_dice,
            'time': epoch_time
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_dice = 0.0
        num_batches = 0
        
        val_start_time = time.time()
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                # Move data to device
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True) if targets is not None else None
                
                if targets is None:
                    continue
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Calculate metrics
                dice_score = calculate_dice_score(outputs, targets).mean()
                
                total_loss += loss.item()
                total_dice += dice_score.item()
                num_batches += 1
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_dice = total_dice / num_batches if num_batches > 0 else 0.0
        val_time = time.time() - val_start_time
        
        return {
            'loss': avg_loss,
            'dice': avg_dice,
            'time': val_time
        }
    
    def save_checkpoint(self, metrics: TrainingMetrics, is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            metrics: Training metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': metrics.to_dict(),
            'config': asdict(self.config),
            'best_val_dice': self.best_val_dice,
            'training_history': self.training_history
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model with Dice score: {metrics.val_dice:.4f}")
        
        # Keep only last N checkpoints
        max_checkpoints = getattr(self.config, 'max_checkpoints', 5)
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        if len(checkpoints) > max_checkpoints:
            for old_checkpoint in checkpoints[:-max_checkpoints]:
                old_checkpoint.unlink()
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path], resume_training: bool = True):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            resume_training: Whether to resume training state
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if resume_training:
            # Load training state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if self.scaler and checkpoint.get('scaler_state_dict'):
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.current_epoch = checkpoint['epoch']
            self.best_val_dice = checkpoint.get('best_val_dice', 0.0)
            self.training_history = checkpoint.get('training_history', [])
        
        self.logger.info(f"Checkpoint loaded successfully from epoch {checkpoint['epoch']}")
    
    def train(self, resume_from_checkpoint: Optional[Union[str, Path]] = None) -> List[TrainingMetrics]:
        """
        Main training loop
        
        Args:
            resume_from_checkpoint: Optional checkpoint path to resume from
            
        Returns:
            List of training metrics for each epoch
        """
        # Load checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint, resume_training=True)
            start_epoch = self.current_epoch + 1
        else:
            start_epoch = 0
        
        self.logger.info("Starting training...")
        self.logger.info(f"Model: {type(self.model).__name__}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        self.logger.info(f"Epochs: {start_epoch} -> {self.config.num_epochs}")
        
        training_start_time = time.time()
        
        try:
            for epoch in range(start_epoch, self.config.num_epochs):
                self.current_epoch = epoch
                
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
                self.logger.info(f"{'='*60}")
                
                # Training phase
                with timer("Training phase", self.logger):
                    train_metrics = self.train_epoch()
                
                # Validation phase
                with timer("Validation phase", self.logger):
                    val_metrics = self.validate_epoch()
                
                # Update scheduler (except OneCycleLR which is updated per batch)
                if self.scheduler and not isinstance(self.scheduler, OneCycleLR):
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['dice'])
                    else:
                        self.scheduler.step()
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Monitor memory
                memory_info = monitor_memory(self.device)
                
                # Create metrics object
                metrics = TrainingMetrics(
                    epoch=epoch,
                    train_loss=train_metrics['loss'],
                    train_dice=train_metrics['dice'],
                    val_loss=val_metrics['loss'],
                    val_dice=val_metrics['dice'],
                    learning_rate=current_lr,
                    training_time=train_metrics['time'] + val_metrics['time'],
                    memory_usage=memory_info
                )
                
                # Log metrics
                self.logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Dice: {train_metrics['dice']:.4f}")
                self.logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Dice: {val_metrics['dice']:.4f}")
                self.logger.info(f"Learning Rate: {current_lr:.6f}")
                self.logger.info(f"Epoch Time: {format_time(metrics.training_time)}")
                
                # TensorBoard logging (optional)
                if self.writer is not None:
                    self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
                    self.writer.add_scalar('Train/Dice', train_metrics['dice'], epoch)
                    self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
                    self.writer.add_scalar('Val/Dice', val_metrics['dice'], epoch)
                    self.writer.add_scalar('Learning_Rate', current_lr, epoch)
                    
                    if self.device.type == 'cuda':
                        self.writer.add_scalar('GPU_Memory/Allocated_MB', 
                                             memory_info['gpu']['allocated'] / 1024**2, epoch)
                
                # Save checkpoint
                is_best = val_metrics['dice'] > self.best_val_dice
                if is_best:
                    self.best_val_dice = val_metrics['dice']
                    self.best_val_loss = val_metrics['loss']
                
                self.save_checkpoint(metrics, is_best=is_best)
                self.training_history.append(metrics)
                
                # Early stopping
                if self.early_stopping(val_metrics['dice']):
                    self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise
        
        finally:
            # Final cleanup
            total_training_time = time.time() - training_start_time
            self.logger.info(f"\nTraining completed in {format_time(total_training_time)}")
            self.logger.info(f"Best validation Dice score: {self.best_val_dice:.4f}")
            
            # Close TensorBoard writer (optional)
            if self.writer is not None:
                self.writer.close()
            
            # Save final training history
            history_path = self.checkpoint_dir / "training_history.json"
            with open(history_path, 'w') as f:
                history_data = [metrics.to_dict() for metrics in self.training_history]
                json.dump(history_data, f, indent=2)
        
        return self.training_history


# Use Config from utils_3d
from utils_3d import Config


def create_trainer(
    config: Config,
    data_dir: Union[str, Path],
    resume_from: Optional[Union[str, Path]] = None
) -> ModelTrainer:
    """
    Factory function to create a complete trainer
    
    Args:
        config: Configuration object
        data_dir: Path to training data
        resume_from: Optional checkpoint to resume from
        
    Returns:
        Configured ModelTrainer instance
    """
    # Setup logging
    logger = setup_logging(
        log_level=config.log_level,
        log_dir=config.log_dir
    )
    
    # Set random seed for reproducibility
    set_random_seed(config.random_seed, config.deterministic)
    
    # Get device
    device = get_device(config.device)
    logger.info(f"Using device: {device}")
    
    # Create model
    if hasattr(config, 'model_config'):
        model = create_unet3d(config.model_config)
    else:
        model = create_unet3d("standard")
    
    # Log model info
    num_params, model_size_mb = model.get_model_size()
    memory_estimate = model.estimate_memory_usage(config.batch_size)
    
    logger.info(f"Model: {type(model).__name__}")
    logger.info(f"Parameters: {num_params:,} ({model_size_mb:.1f} MB)")
    logger.info(f"Estimated GPU memory: {memory_estimate:.1f} MB")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    data_loaders = create_data_loaders(
        data_dir=data_dir,
        config=config,
        splits=["train", "val"]
    )
    
    train_loader = data_loaders["train"]
    val_loader = data_loaders["val"]
    
    # Create trainer
    trainer = ModelTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        logger=logger
    )
    
    return trainer


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train 3D Brain Tumor Segmentation Model")
    parser.add_argument("--config", "-c", type=str, help="Configuration file path")
    parser.add_argument("--data-dir", type=str, required=True, help="Training data directory")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--device", default="auto", help="Training device")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = Config.from_file(args.config)
    else:
        config = Config()
    
    # Override device if specified
    if args.device != "auto":
        config.device = args.device
    
    # Create and run trainer
    trainer = create_trainer(
        config=config,
        data_dir=args.data_dir,
        resume_from=args.resume
    )
    
    # Start training
    training_history = trainer.train(resume_from_checkpoint=args.resume)
    
    print(f"\nTraining completed!")
    print(f"Best validation Dice score: {trainer.best_val_dice:.4f}")


if __name__ == "__main__":
    main()