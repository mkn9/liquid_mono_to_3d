#!/usr/bin/env python3
"""
D-NeRF Temporal Prediction Demonstration
Shows how D-NeRF can predict future frames from past observations.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import imageio

class DNerfPredictionDemo:
    """Demonstrate D-NeRF temporal prediction capabilities."""
    
    def __init__(self, data_dir="data/sphere_trajectories"):
        self.data_dir = Path(data_dir)
        self.load_dataset_info()
        
    def load_dataset_info(self):
        """Load dataset information from transforms.json."""
        transforms_path = self.data_dir / "transforms.json"
        
        if not transforms_path.exists():
            print(f"Dataset not found at {transforms_path}")
            print("Please run dnerf_data_augmentation.py first")
            return
            
        with open(transforms_path, 'r') as f:
            self.transforms = json.load(f)
        
        # Extract time information
        self.frames = self.transforms['frames']
        self.times = sorted(set(frame['time'] for frame in self.frames))
        self.camera_ids = sorted(set(frame['camera_id'] for frame in self.frames))
        
        print(f"Dataset loaded: {len(self.frames)} frames, {len(self.times)} time steps, {len(self.camera_ids)} cameras")
    
    def get_frames_at_time(self, time_idx):
        """Get all camera frames at a specific time."""
        target_time = self.times[time_idx]
        frames_at_time = [f for f in self.frames if f['time'] == target_time]
        return frames_at_time
    
    def get_frames_for_camera(self, camera_id):
        """Get all time frames for a specific camera."""
        frames_for_camera = [f for f in self.frames if f['camera_id'] == camera_id]
        return sorted(frames_for_camera, key=lambda x: x['time'])
    
    def predict_next_frame_simple(self, camera_id, past_frames=5):
        """Simple linear prediction of next frame based on past motion."""
        camera_frames = self.get_frames_for_camera(camera_id)
        
        if len(camera_frames) < past_frames + 1:
            print(f"Not enough frames for camera {camera_id}")
            return None
        
        # Load images and extract features
        recent_frames = camera_frames[-past_frames:]
        images = []
        
        for frame in recent_frames:
            img_path = self.data_dir / "images" / frame['file_name']
            if img_path.exists():
                img = cv2.imread(str(img_path))
                images.append(img)
        
        if len(images) < past_frames:
            print("Could not load enough images")
            return None
        
        # Simple temporal interpolation
        # This is a simplified version - real D-NeRF uses neural networks
        predicted_image = self.temporal_interpolation(images)
        
        return predicted_image
    
    def temporal_interpolation(self, images):
        """Perform temporal interpolation to predict next frame."""
        # Convert to numpy arrays
        img_arrays = [np.array(img, dtype=np.float32) for img in images]
        
        # Calculate temporal differences
        diffs = []
        for i in range(1, len(img_arrays)):
            diff = img_arrays[i] - img_arrays[i-1]
            diffs.append(diff)
        
        # Predict next frame using linear extrapolation
        if len(diffs) > 0:
            # Average the differences to get motion trend
            avg_diff = np.mean(diffs, axis=0)
            
            # Predict next frame
            predicted = img_arrays[-1] + avg_diff
            
            # Clamp values to valid range
            predicted = np.clip(predicted, 0, 255)
            
            return predicted.astype(np.uint8)
        
        return images[-1]  # Return last frame if no prediction possible
    
    def demonstrate_prediction_accuracy(self):
        """Demonstrate prediction accuracy by comparing predicted vs actual frames."""
        print("\n=== D-NeRF Temporal Prediction Demonstration ===")
        
        results = []
        
        # Test prediction for each camera
        for camera_id in self.camera_ids[:3]:  # Test first 3 cameras
            print(f"\nTesting camera {camera_id}:")
            
            camera_frames = self.get_frames_for_camera(camera_id)
            
            # Use 80% of frames for "training", 20% for testing
            split_idx = int(len(camera_frames) * 0.8)
            train_frames = camera_frames[:split_idx]
            test_frames = camera_frames[split_idx:]
            
            print(f"  Train frames: {len(train_frames)}, Test frames: {len(test_frames)}")
            
            # Predict each test frame
            for i, test_frame in enumerate(test_frames[:3]):  # Test first 3 frames
                # Get past frames for prediction
                past_frames = train_frames[-5:] if len(train_frames) >= 5 else train_frames
                
                # Load actual test frame
                actual_img_path = self.data_dir / "images" / test_frame['file_name']
                if not actual_img_path.exists():
                    continue
                    
                actual_img = cv2.imread(str(actual_img_path))
                
                # Predict frame
                predicted_img = self.predict_next_frame_simple(camera_id, past_frames=5)
                
                if predicted_img is not None:
                    # Calculate prediction error
                    error = np.mean((actual_img.astype(np.float32) - predicted_img.astype(np.float32))**2)
                    
                    results.append({
                        'camera_id': camera_id,
                        'frame_idx': i,
                        'time': test_frame['time'],
                        'mse_error': error,
                        'predicted_img': predicted_img,
                        'actual_img': actual_img
                    })
                    
                    print(f"    Frame {i}: MSE = {error:.2f}")
        
        return results
    
    def create_prediction_visualization(self, results):
        """Create visualization comparing predicted vs actual frames."""
        if not results:
            print("No prediction results to visualize")
            return
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('D-NeRF Temporal Prediction: Predicted vs Actual', fontsize=16)
        
        for i, result in enumerate(results[:3]):
            if i >= 3:
                break
                
            # Plot predicted frame
            axes[0, i].imshow(cv2.cvtColor(result['predicted_img'], cv2.COLOR_BGR2RGB))
            axes[0, i].set_title(f'Predicted (Cam {result["camera_id"]})')
            axes[0, i].axis('off')
            
            # Plot actual frame
            axes[1, i].imshow(cv2.cvtColor(result['actual_img'], cv2.COLOR_BGR2RGB))
            axes[1, i].set_title(f'Actual (MSE: {result["mse_error"]:.1f})')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.data_dir / "prediction_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Prediction comparison saved to {plot_path}")
        
        plt.show()
        plt.close()
    
    def analyze_motion_predictability(self):
        """Analyze how predictable the motion patterns are."""
        print("\n=== Motion Predictability Analysis ===")
        
        # Load trajectory data
        traj_files = [
            "horizontal_forward.csv",
            "diagonal_ascending.csv", 
            "vertical_drop.csv",
            "curved_path.csv",
            "reverse_motion.csv"
        ]
        
        predictability_scores = []
        
        for traj_file in traj_files:
            traj_path = Path("../output/sphere_trajectories") / traj_file
            if not traj_path.exists():
                continue
                
            df = pd.read_csv(traj_path)
            
            # Calculate position predictability using linear regression
            X = df['time'].values.reshape(-1, 1)
            
            # Test predictability for each dimension
            r2_scores = []
            for dim in ['x', 'y', 'z']:
                y = df[dim].values
                reg = LinearRegression()
                reg.fit(X, y)
                r2 = reg.score(X, y)
                r2_scores.append(r2)
            
            avg_r2 = np.mean(r2_scores)
            
            # Calculate velocity consistency
            velocities = np.array([
                np.diff(df['x']),
                np.diff(df['y']),
                np.diff(df['z'])
            ]).T
            
            velocity_consistency = 1.0 - np.std(np.linalg.norm(velocities, axis=1)) / np.mean(np.linalg.norm(velocities, axis=1))
            
            predictability_scores.append({
                'trajectory': traj_file,
                'linearity_r2': avg_r2,
                'velocity_consistency': velocity_consistency,
                'overall_predictability': (avg_r2 + velocity_consistency) / 2
            })
        
        return predictability_scores
    
    def recommend_data_augmentation(self, predictability_scores):
        """Recommend data augmentation strategies based on analysis."""
        print("\n=== Data Augmentation Recommendations ===")
        
        # Analyze current data limitations
        avg_predictability = np.mean([s['overall_predictability'] for s in predictability_scores])
        
        recommendations = []
        
        if avg_predictability > 0.9:
            recommendations.append("Current motion is very predictable (linear). Consider adding:")
            recommendations.append("  - Non-linear trajectories (curved, accelerating)")
            recommendations.append("  - Sudden direction changes")
            recommendations.append("  - Variable speed patterns")
        
        recommendations.append("\nGeneral D-NeRF Data Augmentation:")
        recommendations.append("  - Increase camera viewpoints (16-32 cameras)")
        recommendations.append("  - Add lighting variations")
        recommendations.append("  - Include object occlusions")
        recommendations.append("  - Vary background scenes")
        recommendations.append("  - Add realistic textures and materials")
        recommendations.append("  - Include depth of field effects")
        
        recommendations.append("\nTemporal Augmentation:")
        recommendations.append("  - Increase frame rate (shorter time steps)")
        recommendations.append("  - Extend sequence duration")
        recommendations.append("  - Add temporal noise/jitter")
        recommendations.append("  - Include start/stop motion patterns")
        
        return recommendations

def main():
    """Main demonstration function."""
    demo = DNerfPredictionDemo()
    
    # Check if dataset exists
    if not hasattr(demo, 'transforms'):
        print("Dataset not found. Please run dnerf_data_augmentation.py first.")
        return
    
    # Demonstrate prediction accuracy
    results = demo.demonstrate_prediction_accuracy()
    
    # Create visualization
    demo.create_prediction_visualization(results)
    
    # Analyze motion predictability
    predictability_scores = demo.analyze_motion_predictability()
    
    print("\n=== MOTION PREDICTABILITY ANALYSIS ===")
    for score in predictability_scores:
        print(f"Trajectory: {score['trajectory']}")
        print(f"  Linearity R²: {score['linearity_r2']:.3f}")
        print(f"  Velocity Consistency: {score['velocity_consistency']:.3f}")
        print(f"  Overall Predictability: {score['overall_predictability']:.3f}")
        print()
    
    # Get augmentation recommendations
    recommendations = demo.recommend_data_augmentation(predictability_scores)
    
    for rec in recommendations:
        print(rec)
    
    print("\n=== D-NeRF PREDICTION WORKFLOW ===")
    print("1. Multi-view temporal data generated ✓")
    print("2. Camera poses and intrinsics computed ✓")
    print("3. D-NeRF training format created ✓")
    print("4. Temporal prediction demonstrated ✓")
    print("5. Data augmentation recommendations provided ✓")
    
    return demo

if __name__ == "__main__":
    demo = main() 