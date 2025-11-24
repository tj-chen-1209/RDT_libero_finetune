import os
import fnmatch
import json

import h5py
import yaml
import numpy as np

from configs.state_vec import STATE_VEC_IDX_MAPPING


class HDF5LiberoSFTDataset:
    """
    This class is used to sample episodes from the embododiment dataset
    stored in HDF5.
    """

    def __init__(self) -> None:
        # [Modify] The path to the HDF5 dataset directory
        # Each HDF5 file contains one episode 
        # TODO: change the dataset name
        HDF5_DIR = "data/datasets/libero_10/"
        self.DATASET_NAME = "libero_10"

        self.file_paths = []
        for root, _, files in os.walk(HDF5_DIR):
            for filename in fnmatch.filter(files, '*.hdf5'):
                file_path = os.path.join(root, filename)
                self.file_paths.append(file_path)

        # Load the config
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']

        # Get each episode's len
        '''
        episode_lens = []
        for file_path in self.file_paths:
            valid, res = self.parse_hdf5_file_state_only(file_path)
            _len = res['state'].shape[0] if valid else 0
            episode_lens.append(_len)
        self.episode_sample_weights = np.array(
            episode_lens) / np.sum(episode_lens)
        '''
        # siqi change
        # self.episode_paths = []  # 用于存储每个 demo_X 路径
        # for file_path in self.file_paths:
        #     with h5py.File(file_path, 'r') as f:
        #         demo_keys = list(f['data'].keys())  # 获取 demo_X 名称
        #         for demo_key in demo_keys:
        #             demo = f['data'][demo_key]
        #             num_steps = demo['actions'].shape[0]

        #             if num_steps >= 128:  # 丢弃太短的 demo
        #                 self.episode_paths.append((file_path, demo_key, num_steps))
        # episode_lens = [episode[2]
        #                 for episode in self.episode_paths]  # 获取每个 demo_X 的长度
        # self.episode_sample_weights = np.array(
        #     episode_lens) / np.sum(episode_lens)
        # 只输入demo_0的路径
        self.episode_paths = []  # 用于存储每个 demo_0 的路径
        for file_path in self.file_paths:
            with h5py.File(file_path, 'r') as f:
                # 确保结构里有 data/demo_0
                if 'data' not in f:
                    continue
                if 'demo_0' not in f['data']:
                    continue

                demo_key = 'demo_0'
                demo = f['data'][demo_key]
                num_steps = demo['actions'].shape[0]

                if num_steps >= 128:  # 丢弃太短的 demo
                    self.episode_paths.append((file_path, demo_key, num_steps))
        episode_lens = [ep[2] for ep in self.episode_paths]
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)


    def __len__(self):
        return len(self.episode_paths)

    def get_dataset_name(self):
        return self.DATASET_NAME

    def get_item(self, index: int = None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        '''
        while True:
            if index is None:
                file_path = np.random.choice(
                    self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            valid, sample = self.parse_hdf5_file(file_path) \
                if not state_only else self.parse_hdf5_file_state_only(file_path)
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))
        '''
        while True:
            if index is None:
                demo_idx = np.random.choice(
                    len(self.episode_paths), p=self.episode_sample_weights)
                demo_path = self.episode_paths[demo_idx]
            else:
                demo_path = self.episode_paths[index]

            file_path, demo_key, _ = demo_path  # 获取文件路径和 demo_key
            valid, sample = self.parse_hdf5_file(file_path, demo_key) \
                if not state_only else self.parse_hdf5_file_state_only(file_path, demo_key)
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.episode_paths))

    def parse_hdf5_file(self, file_path, demo_key):
        """[Modify] Parse a hdf5 file to generate a training sample at
            a random timestep.

        Args:
            file_path (str): the path to the hdf5 file
            demo_key (str): the key of the demo 
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "dataset_name": str,    # the name of your dataset.
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },                           
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, STATE_DIM).
                    "state_std": ndarray,       # std(state[:]), (STATE_DIM,).
                    "state_mean": ndarray,      # mean(state[:]), (STATE_DIM,).
                    "state_norm": ndarray,      # norm(state[:]), (STATE_DIM,).
                    "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
                    "state_indicator", ndarray, # indicates the validness of each dim, (STATE_DIM,).
                    "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
                                                # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
                    "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_left_wrist_mask": ndarray,
                    "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                                                # If only one wrist, make it right wrist, plz.
                    "cam_right_wrist_mask": ndarray
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            # LIBERO dataset structure: data/demo_X/
            demo = f['data'][demo_key]
            joint_states = demo['obs']['joint_states'][:]
            gripper_states = demo['obs']['gripper_states'][:]

            # Concatenate joint states and gripper states 7DoF+2gripper
            qpos = np.concatenate([joint_states, gripper_states], axis=1)
            num_steps = qpos.shape[0]
            # [Optional] We drop too-short episode
            # if num_steps < 128:
            #     return False, None

            # [Optional] We skip the first few still steps TODO
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")

            # We randomly sample a timestep TODO
            step_id = np.random.randint(first_idx-1, num_steps)

            # TODO: add instruction 这个地方要跟 eval 对齐
            def extract_instruction(file_path):
                """
                从 HDF5 文件名中提取任务指令
                
                Args:
                    file_path: HDF5 文件路径，例如：
                        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5"
                        "pick_up_the_black_bowl_and_place_it_on_the_plate_demo.hdf5"
                
                Returns:
                    str: 提取的指令，例如：
                        "turn on the stove and put the moka pot on it"
                        "pick up the black bowl and place it on the plate"
                """
                # 获取文件名（不含路径和扩展名）
                task_name = os.path.basename(file_path).replace('_demo.hdf5', '')
                 # 如果文件名以大写字母开头（如 KITCHEN_SCENE3_...）
                if task_name and task_name[0].isupper():
                    # 查找 SCENE 的位置
                    scene_pos = task_name.find("SCENE")
                    if scene_pos != -1:
                        # 检查是否是 SCENE10（需要跳过 8 个字符：SCENE10_）
                        if "SCENE10" in task_name:
                            # 从 SCENE10_ 之后开始提取
                            language_part = task_name[scene_pos + 8:]
                        else:
                            # 从 SCENE#_ 之后开始提取（跳过 7 个字符：SCENE + 数字 + _）
                            # 例如：SCENE3_ -> 跳过 7 个字符
                            language_part = task_name[scene_pos + 7:]
                        
                        # 将下划线替换为空格
                        instruction = language_part.replace('_', ' ')
                    else:
                        # 没有找到 SCENE，直接替换所有下划线
                        instruction = task_name.replace('_', ' ')
                else:
                    # 文件名不以大写字母开头，直接替换所有下划线
                    instruction = task_name.replace('_', ' ')
                
                return instruction.strip()
            
            # embeddied instruction
            task_name = os.path.basename(file_path).replace('_demo.hdf5', '')
            lang_embed_path = os.path.join("outs/libero_embeddings", self.DATASET_NAME, task_name + ".pt")

            if os.path.exists(lang_embed_path):
                instruction = lang_embed_path
            else:
                instruction = extract_instruction(file_path)
            # You can also use precomputed language embeddings (recommended)
            # instruction = "path/to/lang_embed.pt"

            # Assemble the meta
            meta = {
                "dataset_name": self.DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction
            }

            # Max-min normalization for the gripper states (last 2 dimensions) TODO: max-min qpos correct
            qpos_min = -0.04245
            qpos_max = 0.05185
            qpos[..., -2:] = (qpos[..., -2:] - qpos_min) / \
                (qpos_max - qpos_min)

            # Extract actions: 6D EEF velocities + 1D gripper velocity
            target_qpos = demo['actions'][step_id:step_id+self.CHUNK_SIZE]

            # Parse the state and action
            state = qpos[step_id:step_id+1]
            state_std = np.std(qpos, axis=0)
            state_mean = np.mean(qpos, axis=0)
            state_norm = np.sqrt(np.mean(qpos**2, axis=0))
            actions = target_qpos
            if actions.shape[0] < self.CHUNK_SIZE:
                # Pad the actions using the last action
                actions = np.concatenate([
                    actions,
                    np.tile(actions[-1:],
                            (self.CHUNK_SIZE-actions.shape[0], 1))
                ], axis=0)

            # Fill the state/action into the unified vector
            def fill_in_state(values):
                # Target indices corresponding to your state space
                # In this example: 7 joints + 2 gripper for each arm
                UNI_STATE_INDICES = [
                    STATE_VEC_IDX_MAPPING[f"arm_joint_{i}_pos"] for i in range(7)
                ] + [
                    STATE_VEC_IDX_MAPPING[f"gripper_joint_{i}_pos"] for i in range(2)
                ]
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec
            state = fill_in_state(state)
            state_indicator = fill_in_state(np.ones_like(state_std))
            state_std = fill_in_state(state_std)
            state_mean = fill_in_state(state_mean)
            state_norm = fill_in_state(state_norm)

            # Fill the action into the unified vector 都认为是velocity
            def fill_in_action(values):
                UNI_ACTION_INDICES = [
                    STATE_VEC_IDX_MAPPING["eef_vel_x"],
                    STATE_VEC_IDX_MAPPING["eef_vel_y"],
                    STATE_VEC_IDX_MAPPING["eef_vel_z"],
                    STATE_VEC_IDX_MAPPING["eef_angular_vel_roll"],
                    STATE_VEC_IDX_MAPPING["eef_angular_vel_pitch"],
                    STATE_VEC_IDX_MAPPING["eef_angular_vel_yaw"],
                ] + [
                    STATE_VEC_IDX_MAPPING["gripper_open"]
                ]
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_ACTION_INDICES] = values
                return uni_vec
            actions = fill_in_action(actions)

            # Parse the images libero的图片是numpy数组
            def parse_img(key):
                # Check if the key exists in the demo
                if key not in demo['obs']:
                    return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0))
                imgs = []
                for i in range(max(step_id-self.IMG_HISORY_SIZE+1, 0), step_id+1):
                    img = demo['obs'][key][i]
                    imgs.append(img)
                imgs = np.stack(imgs)
                if imgs.shape[0] < self.IMG_HISORY_SIZE:
                    # Pad the images using the first image
                    imgs = np.concatenate([
                        np.tile(imgs[:1], (self.IMG_HISORY_SIZE -
                                imgs.shape[0], 1, 1, 1)),
                        imgs
                    ], axis=0)
                return imgs

            # For step_id = first_idx - 1, the valid_len should be one
            valid_len = min(step_id - (first_idx - 1) +
                            1, self.IMG_HISORY_SIZE)
            cam_high_mask = np.array(
                [False] * (self.IMG_HISORY_SIZE - valid_len) +
                [True] * valid_len
            )

            # `cam_high` is the external camera image
            cam_high = parse_img('agentview_rgb')
            cam_high_mask = cam_high_mask if cam_high.shape[1] > 0 else np.zeros(
                self.IMG_HISORY_SIZE, dtype=bool)

            # Left wrist camera is not available in LIBERO
            cam_left_wrist = np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0))
            cam_left_wrist_mask = np.zeros(self.IMG_HISORY_SIZE, dtype=bool)

            # Right wrist camera (eye-in-hand)
            cam_right_wrist = parse_img('eye_in_hand_rgb')
            cam_right_wrist_mask = cam_high_mask.copy(
            ) if cam_right_wrist.shape[1] > 0 else np.zeros(self.IMG_HISORY_SIZE, dtype=bool)

            # Return the resulting sample
            # For unavailable images, return zero-shape arrays, i.e., (IMG_HISORY_SIZE, 0, 0, 0)
            # E.g., return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0)) for the key "cam_left_wrist",
            # if the left-wrist camera is unavailable on your robot
            return True, {
                "meta": meta,
                "state": state,
                "state_std": state_std,
                "state_mean": state_mean,
                "state_norm": state_norm,
                "actions": actions,
                "state_indicator": state_indicator,
                "cam_high": cam_high,
                "cam_high_mask": cam_high_mask,
                "cam_left_wrist": cam_left_wrist,
                "cam_left_wrist_mask": cam_left_wrist_mask,
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_right_wrist_mask
            }

    def parse_hdf5_file_state_only(self, file_path, demo_key):
        """[Modify] Parse a hdf5 file to generate a state trajectory.

        Args:
            file_path (str): the path to the hdf5 file
            demo_key (str): the key of the demo 
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "state": ndarray,           # state[:], (T, STATE_DIM).
                    "action": ndarray,          # action[:], (T, STATE_DIM).
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            # LIBERO dataset structure: data/demo_X/
            demo = f['data'][demo_key]
            joint_states = demo['obs']['joint_states'][:]
            gripper_states = demo['obs']['gripper_states'][:]
            actions = demo['actions'][:]
            # Concatenate joint states and gripper states 7DoF+2gripper
            qpos = np.concatenate([joint_states, gripper_states], axis=1)
            num_steps = qpos.shape[0]
            # [Optional] We drop too-short episode
            # if num_steps < 128:
            #     return False, None

            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")

            # Max-min normalization for the gripper states
            qpos_min = -0.04245
            qpos_max = 0.05185
            qpos[..., -2:] = (qpos[..., -2:] - qpos_min) / \
                (qpos_max - qpos_min)

            target_qpos = actions

            # Parse the state and action
            state = qpos[first_idx-1:]
            action = target_qpos[first_idx-1:]

            # Fill the state/action into the unified vector
            def fill_in_state(values):
                # Target indices corresponding to your state space
                # In this example: 7 joints + 2 gripper for each arm
                UNI_STATE_INDICES = [
                    STATE_VEC_IDX_MAPPING[f"arm_joint_{i}_pos"] for i in range(7)
                ] + [
                    STATE_VEC_IDX_MAPPING[f"gripper_joint_{i}_pos"] for i in range(2)
                ]
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec

            def fill_in_action(values):
                UNI_ACTION_INDICES = [
                    STATE_VEC_IDX_MAPPING["eef_vel_x"],
                    STATE_VEC_IDX_MAPPING["eef_vel_y"],
                    STATE_VEC_IDX_MAPPING["eef_vel_z"],
                    STATE_VEC_IDX_MAPPING["eef_angular_vel_roll"],
                    STATE_VEC_IDX_MAPPING["eef_angular_vel_pitch"],
                    STATE_VEC_IDX_MAPPING["eef_angular_vel_yaw"],
                ] + [
                    STATE_VEC_IDX_MAPPING["gripper_open"]
                ]
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_ACTION_INDICES] = values
                return uni_vec

            state = fill_in_state(state)
            action = fill_in_action(action)

            # Return the resulting sample
            return True, {
                "state": state,
                "action": action
            }


if __name__ == "__main__":
    ds = HDF5VLADataset()
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        ds.get_item(i)
