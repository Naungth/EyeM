# IBVS Controller Moving Away from Features - Debug Analysis

## Problem
The IBVS controller appears to be moving the robot **away** from the detected feature corners over successive timesteps, instead of moving towards them.

## Root Cause Analysis

### 1. **Feature Correspondence Issue (MOST LIKELY)**
Shi-Tomasi corner detection picks the "best" corners each frame, but **different corners may be selected**:
- Frame 1: Feature 0 might be corner A at position (100, 100)
- Frame 2: Feature 0 might be corner B at position (150, 120)

The controller tries to move feature 0 to desired position (320, 240), but since it's tracking different physical corners, it creates instability.

**Solution**: Implement proper feature tracking (optical flow, KLT tracker, or feature matching) to ensure the same physical features are tracked across frames.

### 2. **Error Sign Issue (POSSIBLE)**
The error is defined as `error = current - desired`, and control law is `v = -λ · L⁺ · error`.

**To test**: Try flipping the error sign:
```python
error = (desired_uv_2d - current_uv_2d).flatten()  # Flipped
```

### 3. **Interaction Matrix Sign Issue (LESS LIKELY)**
The interaction matrix signs might be wrong for the camera frame convention. Check if camera frame is right-handed vs left-handed.

## Current Implementation
- Error: `error = current_uv - desired_uv` (line 200 in ibvs_controller.py)
- Control: `v = -λ · L_DLS⁺ · error` (line 180 in ibvs_controller.py)
- This matches the real-world `ibvs.py` implementation

## Debug Steps
1. Run with debug prints enabled (already added)
2. Check if features are jumping around (different corners detected each frame)
3. Try flipping error sign to test
4. Implement feature tracking to ensure consistent correspondence

## Quick Fix to Test
To test if error sign is the issue, change line 200 in `ibvs_controller.py`:
```python
# Original:
error = (current_uv_2d - desired_uv_2d).flatten()

# Test (flipped):
error = (desired_uv_2d - current_uv_2d).flatten()
```

