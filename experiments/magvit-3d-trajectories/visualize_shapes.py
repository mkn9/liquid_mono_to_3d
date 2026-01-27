#!/usr/bin/env python3
"""
Generate visual samples of shape rendering.

Creates images showing what cube/cylinder/cone look like in 2D.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from magvit_3d_generator import _render_shape_simple


def main():
    """Generate shape visualization samples."""
    
    img_size = 128
    shapes = ['cube', 'cylinder', 'cone']
    shape_names = ['Cube\n(Square)', 'Cylinder\n(Circle)', 'Cone\n(Triangle)']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_names = ['Red', 'Green', 'Blue']
    
    # 1. Basic shape samples at center
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, shape, name, color, color_name in zip(axes, shapes, shape_names, colors, color_names):
        img = _render_shape_simple(shape, np.array([0.0, 0.0, 0.0]), color, img_size)
        
        ax.imshow(img)
        ax.set_title(f'{name}\n{color_name} pixels', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add crosshair at center
        center = img_size // 2
        ax.axhline(center, color='white', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(center, color='white', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.suptitle('Shape Rendering: How Cube/Cylinder/Cone Appear in 2D', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_dir = Path(__file__).parent / 'results'
    output_file = output_dir / 'shape_rendering_basic.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    print(f"   Size: {output_file.stat().st_size / 1024:.1f} KB")
    plt.close()
    
    # 2. Shapes at different positions
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    positions = [
        (np.array([-0.2, -0.2, 0.0]), 'Top-Left'),
        (np.array([0.0, -0.2, 0.0]), 'Top-Center'),
        (np.array([0.2, -0.2, 0.0]), 'Top-Right'),
        (np.array([-0.2, 0.0, 0.0]), 'Middle-Left'),
        (np.array([0.0, 0.0, 0.0]), 'Center'),
        (np.array([0.2, 0.0, 0.0]), 'Middle-Right'),
        (np.array([-0.2, 0.2, 0.0]), 'Bottom-Left'),
        (np.array([0.0, 0.2, 0.0]), 'Bottom-Center'),
        (np.array([0.2, 0.2, 0.0]), 'Bottom-Right'),
    ]
    
    for idx, (ax, (pos, label)) in enumerate(zip(axes.flat, positions)):
        # Cycle through shapes
        shape_idx = idx % 3
        shape = shapes[shape_idx]
        color = colors[shape_idx]
        
        img = _render_shape_simple(shape, pos, color, img_size)
        
        ax.imshow(img)
        ax.set_title(f'{label}\n{shapes[shape_idx]}', fontsize=10)
        ax.axis('off')
    
    plt.suptitle('Shape Positioning Test: 9 Different Positions', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / 'shape_rendering_positions.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    print(f"   Size: {output_file.stat().st_size / 1024:.1f} KB")
    plt.close()
    
    # 3. Side-by-side comparison with measurements
    fig = plt.figure(figsize=(16, 6))
    
    for i, (shape, name, color, color_name) in enumerate(zip(shapes, shape_names, colors, color_names)):
        img = _render_shape_simple(shape, np.array([0.0, 0.0, 0.0]), color, img_size)
        
        # Find colored pixels
        if color == (255, 0, 0):  # Red
            pixels = np.where(img[:, :, 0] == 255)
        elif color == (0, 255, 0):  # Green
            pixels = np.where(img[:, :, 1] == 255)
        else:  # Blue
            pixels = np.where(img[:, :, 2] == 255)
        
        pixel_count = len(pixels[0])
        
        if len(pixels[0]) > 0:
            y_range = pixels[0].max() - pixels[0].min() + 1
            x_range = pixels[1].max() - pixels[1].min() + 1
            center_y = np.mean(pixels[0])
            center_x = np.mean(pixels[1])
        else:
            y_range, x_range = 0, 0
            center_y, center_x = 0, 0
        
        # Plot
        ax = fig.add_subplot(1, 3, i + 1)
        ax.imshow(img)
        ax.set_title(f'{name}\n{color_name}', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add measurements as text
        ax.text(5, 15, f'Pixels: {pixel_count}', 
                color='white', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.text(5, 30, f'Width: {x_range} px', 
                color='white', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.text(5, 45, f'Height: {y_range} px', 
                color='white', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.text(5, 60, f'Center: ({center_x:.0f}, {center_y:.0f})', 
                color='white', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Mark center with crosshair
        ax.plot(center_x, center_y, 'w+', markersize=15, markeredgewidth=2)
    
    plt.suptitle('Shape Rendering with Measurements', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / 'shape_rendering_measurements.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    print(f"   Size: {output_file.stat().st_size / 1024:.1f} KB")
    plt.close()
    
    # 4. Zoomed in view (enlarged)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, shape, name, color in zip(axes, shapes, shape_names, colors):
        img = _render_shape_simple(shape, np.array([0.0, 0.0, 0.0]), color, img_size)
        
        # Zoom in on center region (±30 pixels from center)
        center = img_size // 2
        zoomed = img[center-30:center+30, center-30:center+30]
        
        ax.imshow(zoomed, interpolation='nearest')
        ax.set_title(f'{name}\n(Zoomed 4x)', fontsize=14, fontweight='bold')
        ax.grid(True, color='white', alpha=0.3, linewidth=0.5)
        ax.set_xticks(range(0, 60, 10))
        ax.set_yticks(range(0, 60, 10))
    
    plt.suptitle('Shape Rendering: Zoomed View (4x magnification)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / 'shape_rendering_zoomed.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    print(f"   Size: {output_file.stat().st_size / 1024:.1f} KB")
    plt.close()
    
    # Print summary
    print("")
    print("="*70)
    print("Shape Rendering Verification Complete")
    print("="*70)
    print("")
    print("Measurements:")
    for shape, color in zip(shapes, colors):
        img = _render_shape_simple(shape, np.array([0.0, 0.0, 0.0]), color, img_size)
        
        if color == (255, 0, 0):
            pixels = np.where(img[:, :, 0] == 255)
        elif color == (0, 255, 0):
            pixels = np.where(img[:, :, 1] == 255)
        else:
            pixels = np.where(img[:, :, 2] == 255)
        
        if len(pixels[0]) > 0:
            pixel_count = len(pixels[0])
            y_range = pixels[0].max() - pixels[0].min() + 1
            x_range = pixels[1].max() - pixels[1].min() + 1
            
            print(f"  {shape.capitalize():10s}: {pixel_count:3d} pixels, "
                  f"width={x_range:2d}, height={y_range:2d}")
            
            if shape == 'cube':
                print(f"             Expected: ~400 pixels (20x20 square)")
            elif shape == 'cylinder':
                expected = np.pi * 10**2
                print(f"             Expected: ~{expected:.0f} pixels (π*10² circle)")
            elif shape == 'cone':
                expected = 0.5 * 20 * 20
                print(f"             Expected: ~{expected:.0f} pixels (triangle)")
    
    print("")
    print("All visualizations saved to:", output_dir)


if __name__ == "__main__":
    main()

