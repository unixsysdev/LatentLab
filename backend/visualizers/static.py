"""
Static Visualization with matplotlib

Generates PNG plots for experiments.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import Optional
from io import BytesIO
import matplotlib

# Use non-interactive backend for server
matplotlib.use('Agg')

from ..models import ExperimentResult


# Dark theme colors
COLORS = {
    'background': '#0a0a0f',
    'grid': '#1a1a2e',
    'text': '#e0e0e0',
    'primary': '#00d4ff',  # Cyan
    'secondary': '#ff00aa',  # Magenta
    'tertiary': '#00ff88',  # Green
    'accent': '#ffaa00',  # Orange
    'negative': '#ff4444',  # Red
    'positive': '#44ff44',  # Green
}


class StaticVisualizer:
    """
    Generates static matplotlib visualizations for experiments.
    """
    
    def __init__(self):
        # Set up matplotlib dark theme
        plt.rcParams.update({
            'figure.facecolor': COLORS['background'],
            'axes.facecolor': COLORS['background'],
            'axes.edgecolor': COLORS['grid'],
            'axes.labelcolor': COLORS['text'],
            'text.color': COLORS['text'],
            'xtick.color': COLORS['text'],
            'ytick.color': COLORS['text'],
            'grid.color': COLORS['grid'],
            'figure.figsize': (12, 10),
            'font.size': 10,
        })
    
    def visualize(
        self,
        result: ExperimentResult,
        format: str = 'png'
    ) -> bytes:
        """
        Generate visualization for any experiment result.
        
        Returns image bytes.
        """
        exp_type = result.experiment_type
        
        if exp_type == "wormhole":
            return self._viz_wormhole(result, format)
        elif exp_type == "supernova":
            return self._viz_supernova(result, format)
        elif exp_type == "mirror":
            return self._viz_mirror(result, format)
        elif exp_type == "steering":
            return self._viz_steering(result, format)
        else:
            return self._viz_generic(result, format)
    
    def _viz_wormhole(self, result: ExperimentResult, format: str) -> bytes:
        """Wormhole: 3D trajectory with gradient"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        points = result.points
        coords = np.array([p.coords_3d for p in points])
        labels = [p.label for p in points]
        
        # Draw trajectory line
        ax.plot(
            coords[:, 0], coords[:, 1], coords[:, 2],
            color=COLORS['primary'],
            linewidth=2,
            alpha=0.8
        )
        
        # Draw points with gradient from start to end
        for i, (coord, label) in enumerate(zip(coords, labels)):
            alpha_val = i / (len(coords) - 1) if len(coords) > 1 else 0.5
            color = self._interpolate_color(
                COLORS['secondary'], COLORS['tertiary'], alpha_val
            )
            
            size = 200 if i == 0 or i == len(coords) - 1 else 100
            
            ax.scatter(
                coord[0], coord[1], coord[2],
                c=[color],
                s=size,
                marker='o',
                edgecolors='white',
                linewidths=1
            )
            
            ax.text(
                coord[0], coord[1], coord[2],
                f"  {label}",
                fontsize=8 if i not in [0, len(coords)-1] else 10,
                color=COLORS['text']
            )
        
        ax.set_title(result.description, fontsize=14, pad=20)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        
        return self._save_to_bytes(fig, format)
    
    def _viz_supernova(self, result: ExperimentResult, format: str) -> bytes:
        """Supernova: Radial starburst"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        points = result.points
        center = points[0]
        attributes = [p for p in points if p.metadata.get('type') == 'attribute']
        anti = [p for p in points if p.metadata.get('type') == 'anti']
        
        center_coord = np.array(center.coords_3d)
        
        # Draw center
        ax.scatter(
            center_coord[0], center_coord[1], center_coord[2],
            s=400,
            c=[COLORS['accent']],
            marker='o',
            edgecolors='white',
            linewidths=2
        )
        ax.text(
            center_coord[0], center_coord[1], center_coord[2],
            f"  {center.label}",
            fontsize=12,
            fontweight='bold',
            color=COLORS['accent']
        )
        
        # Draw attribute rays
        for attr in attributes:
            coord = np.array(attr.coords_3d)
            ax.plot(
                [center_coord[0], coord[0]],
                [center_coord[1], coord[1]],
                [center_coord[2], coord[2]],
                color=COLORS['primary'],
                linewidth=1,
                alpha=0.6
            )
            ax.scatter(
                coord[0], coord[1], coord[2],
                s=80,
                c=[COLORS['primary']],
                marker='o',
                alpha=0.8
            )
            ax.text(
                coord[0], coord[1], coord[2],
                f"  {attr.label}",
                fontsize=7,
                color=COLORS['text']
            )
        
        # Draw anti-concept
        if anti:
            anti_point = anti[0]
            coord = np.array(anti_point.coords_3d)
            ax.scatter(
                coord[0], coord[1], coord[2],
                s=300,
                c=[COLORS['secondary']],
                marker='^',
                edgecolors='white',
                linewidths=2
            )
            ax.text(
                coord[0], coord[1], coord[2],
                f"  {anti_point.label}",
                fontsize=10,
                color=COLORS['secondary']
            )
        
        ax.set_title(result.description, fontsize=14, pad=20)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        
        return self._save_to_bytes(fig, format)
    
    def _viz_mirror(self, result: ExperimentResult, format: str) -> bytes:
        """Mirror: Side-by-side chains with connections"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        source_points = [p for p in result.points if p.metadata.get('type') == 'source']
        target_points = [p for p in result.points if p.metadata.get('type') == 'target']
        
        # Draw source chain
        if source_points:
            coords = np.array([p.coords_3d for p in source_points])
            ax.plot(
                coords[:, 0], coords[:, 1], coords[:, 2],
                color=COLORS['primary'],
                linewidth=2,
                label='Source'
            )
            for i, p in enumerate(source_points):
                ax.scatter(
                    p.coords_3d[0], p.coords_3d[1], p.coords_3d[2],
                    s=150,
                    c=[COLORS['primary']],
                    marker='o'
                )
                ax.text(
                    p.coords_3d[0], p.coords_3d[1], p.coords_3d[2],
                    f"  {p.label}",
                    fontsize=9,
                    color=COLORS['primary']
                )
        
        # Draw target chain
        if target_points:
            coords = np.array([p.coords_3d for p in target_points])
            ax.plot(
                coords[:, 0], coords[:, 1], coords[:, 2],
                color=COLORS['secondary'],
                linewidth=2,
                label='Target'
            )
            for i, p in enumerate(target_points):
                ax.scatter(
                    p.coords_3d[0], p.coords_3d[1], p.coords_3d[2],
                    s=150,
                    c=[COLORS['secondary']],
                    marker='s'
                )
                ax.text(
                    p.coords_3d[0], p.coords_3d[1], p.coords_3d[2],
                    f"  {p.label}",
                    fontsize=9,
                    color=COLORS['secondary']
                )
        
        # Draw mirror connections (dashed)
        for sp, tp in zip(source_points, target_points):
            ax.plot(
                [sp.coords_3d[0], tp.coords_3d[0]],
                [sp.coords_3d[1], tp.coords_3d[1]],
                [sp.coords_3d[2], tp.coords_3d[2]],
                color=COLORS['grid'],
                linestyle='--',
                linewidth=1,
                alpha=0.5
            )
        
        ax.set_title(result.description, fontsize=14, pad=20)
        ax.legend()
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        
        return self._save_to_bytes(fig, format)
    
    def _viz_steering(self, result: ExperimentResult, format: str) -> bytes:
        """Steering: Show prompt, outputs, and steering direction"""
        fig = plt.figure(figsize=(16, 10))
        
        # 3D plot on left
        ax1 = fig.add_subplot(121, projection='3d')
        
        colors_map = {
            'prompt': COLORS['text'],
            'positive': COLORS['positive'],
            'negative': COLORS['negative'],
            'original': COLORS['primary'],
            'steered': COLORS['secondary']
        }
        
        markers_map = {
            'prompt': 's',
            'positive': '^',
            'negative': 'v',
            'original': 'o',
            'steered': 'D'
        }
        
        for p in result.points:
            ptype = p.metadata.get('type', 'unknown')
            color = colors_map.get(ptype, COLORS['text'])
            marker = markers_map.get(ptype, 'o')
            
            ax1.scatter(
                p.coords_3d[0], p.coords_3d[1], p.coords_3d[2],
                s=200,
                c=[color],
                marker=marker,
                edgecolors='white',
                linewidths=1
            )
            ax1.text(
                p.coords_3d[0], p.coords_3d[1], p.coords_3d[2],
                f"  {p.label[:30]}",
                fontsize=8,
                color=color
            )
        
        ax1.set_title('Semantic Space', fontsize=12)
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.set_zlabel('PC3')
        
        # Text comparison on right
        ax2 = fig.add_subplot(122)
        ax2.axis('off')
        
        meta = result.metadata
        text_content = f"""PROMPT:
{meta.get('prompt', 'N/A')}

STEERING: {meta.get('negative', '?')} → {meta.get('positive', '?')}
Layer: {meta.get('layer', '?')}, Strength: {meta.get('strength', '?')}

ORIGINAL OUTPUT:
{meta.get('original_output', 'N/A')[:300]}

STEERED OUTPUT:
{meta.get('steered_output', 'N/A')[:300]}"""
        
        ax2.text(
            0.05, 0.95, text_content,
            transform=ax2.transAxes,
            fontsize=9,
            verticalalignment='top',
            fontfamily='monospace',
            color=COLORS['text'],
            bbox=dict(boxstyle='round', facecolor=COLORS['grid'], alpha=0.5)
        )
        
        fig.suptitle(result.description, fontsize=14)
        
        return self._save_to_bytes(fig, format)
    
    def _viz_generic(self, result: ExperimentResult, format: str) -> bytes:
        """Generic 3D scatter for unknown experiment types"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for p in result.points:
            ax.scatter(
                p.coords_3d[0], p.coords_3d[1], p.coords_3d[2],
                s=100,
                c=[COLORS['primary']],
                marker='o'
            )
            ax.text(
                p.coords_3d[0], p.coords_3d[1], p.coords_3d[2],
                f"  {p.label}",
                fontsize=8,
                color=COLORS['text']
            )
        
        # Draw connections
        for i, j in result.connections:
            if i < len(result.points) and j < len(result.points):
                p1, p2 = result.points[i], result.points[j]
                ax.plot(
                    [p1.coords_3d[0], p2.coords_3d[0]],
                    [p1.coords_3d[1], p2.coords_3d[1]],
                    [p1.coords_3d[2], p2.coords_3d[2]],
                    color=COLORS['grid'],
                    linewidth=1
                )
        
        ax.set_title(result.description, fontsize=14, pad=20)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        
        return self._save_to_bytes(fig, format)
    
    def _interpolate_color(self, c1: str, c2: str, t: float) -> str:
        """Interpolate between two hex colors"""
        def hex_to_rgb(h):
            h = h.lstrip('#')
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        
        def rgb_to_hex(r, g, b):
            return f'#{int(r):02x}{int(g):02x}{int(b):02x}'
        
        r1, g1, b1 = hex_to_rgb(c1)
        r2, g2, b2 = hex_to_rgb(c2)
        
        r = r1 + (r2 - r1) * t
        g = g1 + (g2 - g1) * t
        b = b1 + (b2 - b1) * t
        
        return rgb_to_hex(r, g, b)
    
    def _save_to_bytes(self, fig, format: str) -> bytes:
        """Save figure to bytes"""
        buf = BytesIO()
        fig.savefig(
            buf,
            format=format,
            dpi=150,
            bbox_inches='tight',
            facecolor=COLORS['background'],
            edgecolor='none'
        )
        plt.close(fig)
        buf.seek(0)
        return buf.read()
