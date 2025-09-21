"""
3D Visualization Tools for Brain Tumor Segmentation

Comprehensive visualization tools for 3D medical image analysis including
slice viewers, 3D renderings, overlay visualizations, and interactive plots.

Key Features:
- Multi-planar slice visualization (axial, sagittal, coronal)
- 3D volume rendering and surface plots
- Segmentation overlays with transparency
- Interactive matplotlib widgets
- Uncertainty visualization
- Comparative visualizations
- Export functionality (PNG, GIF, HTML)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import nibabel as nib
from skimage import measure
from matplotlib.figure import Figure


class VolumeVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (12, 8), cmap: str = 'gray', overlay_alpha: float = 0.6):
        self.figsize = figsize
        self.cmap = cmap
        self.overlay_alpha = overlay_alpha

    def plot_slices(self, volume: np.ndarray, mask: Optional[np.ndarray] = None, 
                   slice_indices: Optional[List[int]] = None, axis: int = 2, 
                   title: str = "Multiplanar View") -> Figure:
        if slice_indices is not None and not all(isinstance(idx, int) for idx in slice_indices):
            raise TypeError("slice_indices must be a list of integers or None")
        
        if slice_indices is None:
            # Select evenly spaced slices
            n_slices = 6
            slice_indices = list(np.linspace(0, volume.shape[axis]-1, n_slices, dtype=int))
        
        n_slices = len(slice_indices)
        cols = min(3, n_slices)
        rows = (n_slices + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=self.figsize)
        axes = np.atleast_2d(axes)

        for i, slice_idx in enumerate(slice_indices):
            row, col = divmod(i, cols)
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # Extract slice
            if axis == 0:  # Sagittal
                img_slice = volume[slice_idx, :, :]
                mask_slice = mask[slice_idx, :, :] if mask is not None else None
            elif axis == 1:  # Coronal
                img_slice = volume[:, slice_idx, :]
                mask_slice = mask[:, slice_idx, :] if mask is not None else None
            else:  # Axial
                img_slice = volume[:, :, slice_idx]
                mask_slice = mask[:, :, slice_idx] if mask is not None else None
            
            # Plot volume slice
            ax.imshow(img_slice, cmap=self.cmap, aspect='equal')
            
            # Overlay mask if provided
            if mask_slice is not None:
                masked = np.ma.masked_where(mask_slice == 0, mask_slice)
                ax.imshow(masked, cmap='Reds', alpha=self.overlay_alpha, aspect='equal')
            
            ax.set_title(f'Slice {slice_idx}')
            ax.axis('off')
        
        # Hide empty subplots
        for i in range(n_slices, rows * cols):
            row, col = divmod(i, cols)
            axes[row, col].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        return fig
    
    def interactive_slice_viewer(self, volume: np.ndarray, mask: Optional[np.ndarray] = None, 
                               title: str = "Interactive Volume Viewer") -> Figure:
        fig, ax = plt.subplots(figsize=self.figsize)
        plt.subplots_adjust(bottom=0.25)
        
        # Initial slice (middle)
        initial_slice = volume.shape[2] // 2
        
        # Display initial slice
        im = ax.imshow(volume[:, :, initial_slice], cmap=self.cmap, aspect='equal')
        overlay = ax.imshow(np.zeros_like(volume[:, :, 0]), cmap='Reds', alpha=self.overlay_alpha, aspect='equal')
        ax.set_title(f'{title} - Slice {initial_slice}')
        ax.axis('off')
        
        # Add slider
        ax_slider = plt.axes((0.2, 0.1, 0.6, 0.03))
        slider = Slider(ax_slider, 'Slice', 0, volume.shape[2]-1, valinit=initial_slice, valfmt='%d')
        
        def update(val):
            slice_idx = int(slider.val)
            im.set_array(volume[:, :, slice_idx])
            masked = np.ma.masked_where(mask[:, :, slice_idx] == 0, mask[:, :, slice_idx]) if mask is not None else None
            overlay.set_array(masked if masked is not None else np.zeros_like(overlay.get_array()))
            ax.set_title(f'{title} - Slice {slice_idx}')
            fig.canvas.draw()
        
        slider.on_changed(update)
        
        return fig
    
    def plot_3d_surface(self, mask: np.ndarray, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                       title: str = "3D Surface Rendering", save_path: Optional[str] = None) -> go.Figure:
        """
        Create 3D surface plot using plotly
        
        Args:
            mask: Binary segmentation mask
            spacing: Voxel spacing
            title: Plot title
            save_path: Path to save HTML file
            
        Returns:
            plotly Figure object
        """
        # Create mesh from binary mask
        verts, faces, _, _ = measure.marching_cubes(mask, level=0.5, spacing=spacing)
        
        # Create plotly 3D mesh
        fig = go.Figure(data=[
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1], 
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color='lightpink',
                opacity=0.50
            )
        ])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='data'
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_multiplanar(self, volume: np.ndarray, mask: Optional[np.ndarray] = None,
                        slice_coords: Optional[Tuple[int, int, int]] = None,
                        title: str = "Multiplanar View") -> Figure:
        """
        Create multiplanar view (axial, sagittal, coronal)
        
        Args:
            volume: 3D volume array
            mask: Optional segmentation mask
            slice_coords: (x, y, z) coordinates for slice intersection
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        if slice_coords is None:
            slice_coords = (volume.shape[0]//2, volume.shape[1]//2, volume.shape[2]//2)
        
        x, y, z = slice_coords
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Axial slice (z-axis)
        ax = axes[0, 0]
        ax.imshow(volume[:, :, z], cmap=self.cmap, aspect='equal')
        if mask is not None:
            masked = np.ma.masked_where(mask[:, :, z] == 0, mask[:, :, z])
            ax.imshow(masked, cmap='Reds', alpha=self.overlay_alpha, aspect='equal')
        ax.axhline(y, color='yellow', linestyle='--', alpha=0.7)
        ax.axvline(x, color='yellow', linestyle='--', alpha=0.7)
        ax.set_title(f'Axial (z={z})')
        ax.axis('off')
        
        # Sagittal slice (x-axis)
        ax = axes[0, 1]
        ax.imshow(volume[x, :, :], cmap=self.cmap, aspect='equal')
        if mask is not None:
            masked = np.ma.masked_where(mask[x, :, :] == 0, mask[x, :, :])
            ax.imshow(masked, cmap='Reds', alpha=self.overlay_alpha, aspect='equal')
        ax.axhline(y, color='yellow', linestyle='--', alpha=0.7)
        ax.axvline(z, color='yellow', linestyle='--', alpha=0.7)
        ax.set_title(f'Sagittal (x={x})')
        ax.axis('off')
        
        # Coronal slice (y-axis)
        ax = axes[1, 0]
        ax.imshow(volume[:, y, :], cmap=self.cmap, aspect='equal')
        if mask is not None:
            masked = np.ma.masked_where(mask[:, y, :] == 0, mask[:, y, :])
            ax.imshow(masked, cmap='Reds', alpha=self.overlay_alpha, aspect='equal')
        ax.axhline(x, color='yellow', linestyle='--', alpha=0.7)
        ax.axvline(z, color='yellow', linestyle='--', alpha=0.7)
        ax.set_title(f'Coronal (y={y})')
        ax.axis('off')
        
        # 3D view placeholder
        ax = axes[1, 1]
        ax.text(0.5, 0.5, '3D View\n(Use plot_3d_surface)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        return fig
    
    def plot_uncertainty_overlay(self, volume: np.ndarray, prediction: np.ndarray, 
                               uncertainty: np.ndarray, slice_idx: Optional[int] = None,
                               axis: int = 2, title: str = "Prediction with Uncertainty") -> Figure:
        """
        Visualize prediction with uncertainty overlay
        
        Args:
            volume: Original volume
            prediction: Prediction probabilities
            uncertainty: Uncertainty map
            slice_idx: Slice index to display
            axis: Slice axis
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        if slice_idx is None:
            slice_idx = volume.shape[axis] // 2
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Extract slices
        if axis == 2:  # Axial
            vol_slice = volume[:, :, slice_idx]
            pred_slice = prediction[:, :, slice_idx]
            unc_slice = uncertainty[:, :, slice_idx]
        elif axis == 1:  # Coronal
            vol_slice = volume[:, slice_idx, :]
            pred_slice = prediction[:, slice_idx, :]
            unc_slice = uncertainty[:, slice_idx, :]
        else:  # Sagittal
            vol_slice = volume[slice_idx, :, :]
            pred_slice = prediction[slice_idx, :, :]
            unc_slice = uncertainty[slice_idx, :, :]
        
        # Original volume
        axes[0].imshow(vol_slice, cmap='gray', aspect='equal')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Prediction
        axes[1].imshow(vol_slice, cmap='gray', aspect='equal')
        pred_overlay = np.ma.masked_where(pred_slice < 0.5, pred_slice)
        axes[1].imshow(pred_overlay, cmap='Reds', alpha=0.6, aspect='equal')
        axes[1].set_title('Prediction')
        axes[1].axis('off')
        
        # Uncertainty
        axes[2].imshow(unc_slice, cmap='viridis', aspect='equal')
        axes[2].set_title('Uncertainty')
        axes[2].axis('off')
        
        # Combined view
        axes[3].imshow(vol_slice, cmap='gray', aspect='equal')
        # High uncertainty in blue, low uncertainty predictions in red
        high_unc = unc_slice > np.percentile(unc_slice, 75)
        low_unc_pred = (pred_slice > 0.5) & (unc_slice < np.percentile(unc_slice, 25))
        
        high_unc_overlay = np.ma.masked_where(~high_unc, np.ones_like(high_unc))
        low_unc_overlay = np.ma.masked_where(~low_unc_pred, np.ones_like(low_unc_pred))
        
        axes[3].imshow(high_unc_overlay, cmap='Blues', alpha=0.4, aspect='equal')
        axes[3].imshow(low_unc_overlay, cmap='Reds', alpha=0.6, aspect='equal')
        axes[3].set_title('High Conf. (Red) / High Unc. (Blue)')
        axes[3].axis('off')
        
        plt.suptitle(f'{title} - Slice {slice_idx}', fontsize=16)
        plt.tight_layout()
        
        return fig


def create_comparison_plot(volumes: Dict[str, np.ndarray], masks: Optional[Dict[str, np.ndarray]] = None,
                         slice_idx: Optional[int] = None, axis: int = 2) -> Figure:
    """
    Create comparison plot of multiple volumes/predictions
    
    Args:
        volumes: Dictionary of volume names to arrays
        masks: Dictionary of mask names to arrays
        slice_idx: Slice index to display
        axis: Slice axis
        
    Returns:
        matplotlib Figure object
    """
    n_volumes = len(volumes)
    n_masks = len(masks) if masks else 0
    total_plots = n_volumes + n_masks
    
    cols = min(4, total_plots)
    rows = (total_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), squeeze=False)
    
    plot_idx = 0
    
    # Plot volumes
    for name, volume in volumes.items():
        if slice_idx is None:
            slice_idx = volume.shape[axis] // 2
            
        row, col = divmod(plot_idx, cols)
        ax = axes[row, col]
        
        if axis == 2:
            img_slice = volume[:, :, slice_idx]
        elif axis == 1:
            img_slice = volume[:, slice_idx, :]
        else:
            img_slice = volume[slice_idx, :, :]
        
        ax.imshow(img_slice, cmap='gray', aspect='equal')
        ax.set_title(name)
        ax.axis('off')
        
        plot_idx += 1
    
    # Plot masks
    if masks:
        for name, mask in masks.items():
            row, col = divmod(plot_idx, cols)
            ax = axes[row, col]
            
            if axis == 2:
                mask_slice = mask[:, :, slice_idx]
            elif axis == 1:
                mask_slice = mask[:, slice_idx, :]
            else:
                mask_slice = mask[slice_idx, :, :]
            
            ax.imshow(mask_slice, cmap='Reds', aspect='equal')
            ax.set_title(f'{name} (Mask)')
            ax.axis('off')
            plot_idx += 1
    
    # Hide empty subplots
    for i in range(plot_idx, rows * cols):
        row, col = divmod(i, cols)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig


def save_slice_montage(volume: np.ndarray, mask: Optional[np.ndarray] = None,
                      output_path: str = "slice_montage.png", n_slices: int = 12) -> None:
    """
    Save a montage of slices as a single image
    
    Args:
        volume: 3D volume array
        mask: Optional segmentation mask
        output_path: Output file path
        n_slices: Number of slices to include
    """
    visualizer = VolumeVisualizer()
    
    # Select evenly spaced slices
    slice_indices = list(np.linspace(0, volume.shape[2]-1, n_slices, dtype=int))
    
    fig = visualizer.plot_slices(volume, mask, slice_indices, 
                               title="Volume Slice Montage")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Slice montage saved to: {output_path}")