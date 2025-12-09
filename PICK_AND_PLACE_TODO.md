# Pick and Place Visual Servoing - TODO Checklist

## ✅ Completed
- [x] State machine framework with 8 states
- [x] Dynamic DOF mask control per state
- [x] State transitions based on convergence
- [x] Visual feedback (state name, error display in screenshots)
- [x] Feature tracking and depth estimation
- [x] IBVS controller with DLS

## 🔧 Critical Missing Components

### 1. **Gripper Control** ⚠️ HIGH PRIORITY
**Location**: `main_sim.py` (needs new system)

**What's missing**: The `GRASP` state currently just waits 1 second but doesn't actually close the gripper.

**To implement**:
- Create a `GripperController` LeafSystem that:
  - Takes state machine input (grasp command)
  - Outputs gripper joint velocities/positions
  - Connects to gripper actuator ports in the plant
- In `main_sim.py`, connect state machine to gripper controller
- Modify `GRASP` state to send close command, then verify grasp before transitioning

**Files to modify**:
- `main_sim.py`: Add gripper controller system and wiring
- `src/state_machine.py`: Add gripper command output port (optional, or use state name)

### 2. **Depth-Based Descent** ⚠️ HIGH PRIORITY
**Location**: `src/ibvs_states.py` - `DESCEND_TO_GRASP` state

**What's missing**: Currently uses visual servoing to descend, but should use depth sensor to know when to stop (e.g., stop at 5cm above object).

**To implement**:
- Modify `DESCEND_TO_GRASP` transition logic in `state_machine.py` to check depth
- Add depth input to state machine (or use existing depth from context)
- Transition when depth reaches target (e.g., `depth < 0.05m`)

**Files to modify**:
- `src/state_machine.py`: Add depth check in `DESCEND_TO_GRASP` transition logic
- `src/ibvs_states.py`: Update `DESCEND_TO_GRASP` config with depth threshold

### 3. **Goal Position Definition** ⚠️ MEDIUM PRIORITY
**Location**: `src/ibvs_states.py` - `_goal_features()` function

**What's missing**: Goal is hardcoded to `(400, 200)` pixels. Need to define actual goal location.

**Options**:
- **Option A**: Use cube detection to find goal location (e.g., detect bin/table)
- **Option B**: Use fixed world coordinates (convert to image coordinates)
- **Option C**: User-provided goal position

**To implement**:
- Modify `_goal_features()` to accept goal position (world coords or image coords)
- Add goal detection or configuration
- Update `MOVE_TO_GOAL` state to use actual goal

**Files to modify**:
- `src/ibvs_states.py`: Update `_goal_features()` function
- `main_sim.py`: Add goal detection/configuration

### 4. **Feature Loss Handling** ⚠️ MEDIUM PRIORITY
**Location**: `src/state_machine.py` - all states

**What's missing**: If features are lost during task (e.g., during grasp), robot should recover.

**To implement**:
- Add feature loss detection in state machine
- Add recovery state or transition back to `WAIT_FOR_FEATURES`
- Or: use last known good features when features are temporarily lost

**Files to modify**:
- `src/state_machine.py`: Add feature loss detection and recovery logic

### 5. **Lift Height Control** ⚠️ MEDIUM PRIORITY
**Location**: `src/ibvs_states.py` - `LIFT` state

**What's missing**: Currently just "move up" indefinitely. Should lift to specific height.

**To implement**:
- Use depth sensor or world position to know when lift is complete
- Add height threshold (e.g., lift to 0.3m above table)
- Transition when height reached

**Files to modify**:
- `src/state_machine.py`: Add height check in `LIFT` transition logic
- `src/ibvs_states.py`: Update `LIFT` config with height threshold

### 6. **Grasp Verification** ⚠️ MEDIUM PRIORITY
**Location**: `src/state_machine.py` - `GRASP` → `LIFT` transition

**What's missing**: After closing gripper, should verify object was actually grasped before lifting.

**To implement**:
- Check gripper force/position feedback
- Or: Check if object moves with gripper (track features during lift)
- Transition to `LIFT` only if grasp verified, else retry or abort

**Files to modify**:
- `src/state_machine.py`: Add grasp verification in `GRASP` state
- `main_sim.py`: Add gripper feedback reading

### 7. **Error Recovery** ⚠️ LOW PRIORITY
**Location**: `src/state_machine.py`

**What's missing**: No error handling for edge cases (collisions, stuck states, etc.)

**To implement**:
- Add timeout for each state
- Add error state for recovery
- Add abort/reset functionality

**Files to modify**:
- `src/state_machine.py`: Add timeout and error handling

## 📝 Implementation Priority

1. **Gripper Control** - Cannot complete pick and place without this
2. **Depth-Based Descent** - Critical for safe grasping
3. **Goal Position** - Needed for place operation
4. **Feature Loss Handling** - Improves robustness
5. **Lift Height Control** - Improves precision
6. **Grasp Verification** - Improves reliability
7. **Error Recovery** - Nice to have

## 🔍 Quick Reference: Where to Edit

| Component | File | What to Change |
|-----------|------|----------------|
| State objectives | `src/ibvs_states.py` | `STATE_CONFIGS` dictionary |
| State transitions | `src/state_machine.py` | `_update_state()` method |
| Gripper control | `main_sim.py` | Add `GripperController` system |
| Goal position | `src/ibvs_states.py` | `_goal_features()` function |
| Depth checks | `src/state_machine.py` | Transition logic in `_update_state()` |
| Feature loss | `src/state_machine.py` | Add recovery logic |
| Visualization | `src/visualization.py` | Overlay display logic |

## 💡 Tips

- Start with gripper control - it's the most critical missing piece
- Test each component incrementally (gripper → descent → goal → etc.)
- Use depth sensor for height/distance checks (more reliable than visual servoing alone)
- Consider adding debug prints/logging for each state transition
- Test with simple scenarios first (e.g., just pick, no place)

