# Session Summary: 3D Track Reconstruction from 2D Mono Tracks

**Date**: December 10, 2025  
**Session Focus**: Unified comparison of triangulation methods with corrected camera setup

## Overview
This session focused on creating a unified comparison of 4 different triangulation methods for reconstructing 3D tracks from 2D mono camera tracks, all using the same camera configuration and trajectory data.

## Key Accomplishments

### 1. Output File Naming Convention Established
- Created `basic/output/` directory for all outputs
- Implemented timestamped naming: `YYYYMMDD_HHMMSS_descriptive_name.ext`
- Created `output_utils.py` helper functions for consistent file saving
- Updated `requirements.md` with full documentation

### 2. Camera Setup Configuration
- **Camera 1**: Position (24, 0, 0), pointing directly at (0, 0, 0)
- **Camera 2**: Position (24, 12, 0), same orientation as Camera 1
- **FOV Calculation**: Corrected to ensure trajectory end point reaches halfway to edge of display
  - Focal lengths: fx=480 pixels, fy=270 pixels (separate for horizontal/vertical)
  - FOV: Horizontal=106.26°, Vertical=106.26°
  - Image size: 1280x720 pixels

### 3. Trajectory Configuration
- **Start Point**: (0, 0, 0)
- **End Point**: (0, 16, 16)
- **Motion**: Constant velocity, 50 points
- **End point in image**: (960, 180) pixels - exactly halfway to edge as specified

### 4. Triangulation Methods Tested
All methods used the same camera setup and trajectory:

1. **SVD (Tracker 8/9)**: RMSE: 0.000000 m, Max error: 0.000000 m
2. **ChatGPT Style**: RMSE: 0.000000 m, Max error: 0.000000 m  
3. **OpenCV**: RMSE: 0.000002 m, Max error: 0.000004 m

All methods achieved near-perfect reconstruction with 50/50 valid points.

### 5. Visualizations Created

#### Individual Method Comparisons
Each method has its own visualization showing True track vs Reconstruction:
- `20251210_175346_3d_track_svd_tracker89_vs_true.png`
- `20251210_175346_3d_track_chatgpt_style_vs_true.png`
- `20251210_175346_3d_track_opencv_vs_true.png`

#### 2D Camera Views
- `20251210_175347_2d_camera_views_corrected_fov.png`
  - Shows full field of view (0-1280, 0-720) with boundaries marked
  - Both Camera 1 and Camera 2 views
  - Trajectory clearly visible from start to end point

### 6. Key Findings

#### FOV Correction
- Initial FOV (70°) was too small - trajectory went out of bounds
- Corrected FOV ensures end point at (960, 180) - halfway to edge
- Separate fx and fy focal lengths needed for proper aspect ratio

#### Camera Visibility
- Initially cameras were outside plot bounds
- Fixed by including camera positions in axis limit calculations
- Cameras now visible in all 3D visualizations

#### Track Visibility
- True track was occluding reconstructed tracks in unified view
- Solution: Created separate visualizations for each method
- Each plot shows True track (black, semi-transparent) vs one reconstruction method

#### Symmetry Analysis
- Created symmetry test showing both up-right and down-left trajectories
- Camera 1: Symmetric behavior (29/50 points visible in both directions)
- Camera 2: Asymmetric due to Y-offset (42/50 up-right, 0/50 down-left)

## Files Created/Modified

### Scripts
- `basic/unified_same_cameras_corrected.py` - Main comparison script with corrected FOV
- `basic/output_utils.py` - Helper functions for timestamped output naming
- `requirements.md` - Updated with output naming convention documentation

### Output Files (in `basic/output/`)
All files use timestamped naming convention:
- `20251210_175346_3d_track_svd_tracker89_vs_true.png`
- `20251210_175346_3d_track_chatgpt_style_vs_true.png`
- `20251210_175346_3d_track_opencv_vs_true.png`
- `20251210_175347_2d_camera_views_corrected_fov.png`

## Technical Details

### Camera Intrinsic Matrix
```
K = [[fx,  0,  cx],
     [ 0, fy,  cy],
     [ 0,  0,   1]]
```
Where:
- fx = 480 pixels (horizontal focal length)
- fy = 270 pixels (vertical focal length)
- cx = 640 pixels (principal point x)
- cy = 360 pixels (principal point y)

### Camera Extrinsics
- Camera 1: Position (24, 0, 0), Rotation matrix for -X viewing direction
- Camera 2: Position (24, 12, 0), Same rotation as Camera 1

### Projection
- All methods use same projection matrices P1 and P2
- Projection: pixel = P @ [X, Y, Z, 1]^T, then normalize by w component

### Triangulation
- All methods solve linear system A @ X = 0 using SVD
- SVD method: Direct matrix construction
- ChatGPT method: Similar approach with different matrix construction
- OpenCV method: Uses cv2.triangulatePoints() function

## Next Steps / Recommendations

1. **Add Noise Analysis**: Test methods with varying levels of pixel noise
2. **Error Visualization**: Create plots showing error distribution over trajectory
3. **Multiple Trajectories**: Test with different trajectory shapes and speeds
4. **Camera Baseline Analysis**: Vary camera separation to study baseline effects
5. **Real Data Integration**: Apply methods to real camera tracking data

## Notes

- All computation performed on EC2 instance (34.196.155.11)
- Results downloaded to MacBook in `basic/output/` directory
- All outputs follow timestamped naming for chronological sorting
- Visualization includes error metrics (RMSE, Max Error) displayed on plots

