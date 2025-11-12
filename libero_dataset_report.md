# LIBERO Dataset Structure Analysis Report

## Dataset Format Summary

### File Structure
- **File**: Each `.hdf5` file contains 50 demos (demo_0 to demo_49)
- **Path**: `data/demo_X/`

### Data Arrays per Demo

#### Observations (`demo['obs']`)
- `joint_states`: (T, 7) float64 - 7 DoF robot joint positions
- `gripper_states`: (T, 2) float64 - 2 gripper joint positions  
  - Range: [-0.04, 0.04] approximately
  - Min observed: [-0.04000677]
  - Max observed: [0.0399059]
- `agentview_rgb`: (T, 128, 128, 3) uint8 - External camera view
- `eye_in_hand_rgb`: (T, 128, 128, 3) uint8 - Wrist camera view
- `ee_pos`: (T, 3) float64 - End-effector position
- `ee_ori`: (T, 3) float64 - End-effector orientation
- `ee_states`: (T, 6) float64 - Combined EE pose

#### Actions (`demo['actions']`)
- Shape: (T, 7) float64
- Format: 6D end-effector velocities + 1D gripper control
  - First 6 dims: [vel_x, vel_y, vel_z, ang_roll, ang_pitch, ang_yaw]
  - Last dim: gripper control {-1, 1}
- Range: 
  - EEF velocities: [-0.76, 0.79] approximately
  - Gripper: {-1, 1}

#### Other Data
- `rewards`: (T,) uint8
- `dones`: (T,) uint8
- `robot_states`: (T, 9) float64
- `states`: (T, 47) float64

---

## Code Verification: `hdf5_libero_dataset.py`

### ✅ CORRECT Implementation

1. **Data Structure Access**
   - ✅ `f['data'].keys()` - Correctly accesses demo keys
   - ✅ `demo['obs']['joint_states']` - Correct path
   - ✅ `demo['obs']['gripper_states']` - Correct path
   - ✅ `demo['actions']` - Correct path

2. **State Concatenation**
   - ✅ `qpos = np.concatenate([joint_states, gripper_states], axis=1)`
   - ✅ Creates (T, 9) array: 7 joints + 2 grippers

3. **Gripper Normalization**
   - ✅ `qpos_min = -0.04245, qpos_max = 0.05185`
   - ✅ Matches observed range
   - ✅ `qpos[..., -2:] = (qpos[..., -2:] - qpos_min) / (qpos_max - qpos_min)`

4. **Action Mapping**
   - ✅ Maps to indices 39-44 (EEF velocities) + 25 (gripper velocity)
   - ✅ Correct `fill_in_action()` implementation

5. **Image Loading**
   - ✅ Image keys: 'agentview_rgb', 'eye_in_hand_rgb'
   - ✅ Images are numpy arrays (uint8), no decoding needed
   - ✅ Error handling for missing keys

6. **State Mapping**
   - ✅ Maps to indices 0-6 (arm joints) + 10-11 (gripper joints)
   - ✅ Correct `fill_in_state()` implementation

---

## ⚠️ ISSUES FOUND

### 1. **CRITICAL: Missing Instruction File**
**Location**: Line 150-152

```python
with open(os.path.join(dir_path, 'expanded_instruction_gpt-4-turbo.json'), 'r') as f_instr:
    instruction_dict = json.load(f_instr)
```

**Problem**: This file doesn't exist in the LIBERO dataset structure.

**Solution**: Extract instruction from filename or use a default instruction.

**Suggested Fix**:
```python
# Extract task name from filename
task_name = os.path.basename(file_path).replace('_demo.hdf5', '').replace('_', ' ')
instruction = task_name
```

---

## Recommended Changes

### Option 1: Extract from Filename
```python
# Line 148-162: Replace instruction loading
task_name = os.path.basename(file_path).replace('_demo.hdf5', '')
# Convert "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it" 
# to "turn on the stove and put the moka pot on it"
parts = task_name.split('_')
# Find where the actual task description starts (after SCENE#)
for i, part in enumerate(parts):
    if part.startswith('SCENE'):
        instruction = ' '.join(parts[i+1:])
        break
else:
    instruction = task_name.replace('_', ' ')
```

### Option 2: Create Instruction Mapping File
Create a JSON file mapping filenames to instructions, or use the LIBERO task descriptions.

---

## Summary

✅ **Data loading**: Correct
✅ **Normalization**: Correct  
✅ **State/Action mapping**: Correct
✅ **Image handling**: Correct
❌ **Instruction loading**: Needs fix (file doesn't exist)

**Action Required**: Fix instruction loading before running training.

