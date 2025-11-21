import os
import sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from configs.state_vec import STATE_VEC_IDX_MAPPING
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from models.rdt_runner import RDTRunner


LIBERO_STATE_INDICES = [
    STATE_VEC_IDX_MAPPING[f"arm_joint_{i}_pos"] for i in range(7)
] + [
    STATE_VEC_IDX_MAPPING[f"gripper_joint_{i}_pos"] for i in range(2)
]
LIBERO_ACTION_INDICES = [
    STATE_VEC_IDX_MAPPING["eef_vel_x"],
    STATE_VEC_IDX_MAPPING["eef_vel_y"],
    STATE_VEC_IDX_MAPPING["eef_vel_z"],
    STATE_VEC_IDX_MAPPING["eef_angular_vel_roll"],
    STATE_VEC_IDX_MAPPING["eef_angular_vel_pitch"],
    STATE_VEC_IDX_MAPPING["eef_angular_vel_yaw"],
] + [
    STATE_VEC_IDX_MAPPING["gripper_open"]
]


def create_model(args, pretrained, **kwargs):
    model = RoboticDiffusionTransformerModel(args, **kwargs)
    if pretrained is not None:
        model.load_pretrained_weights(pretrained)
    return model

class RoboticDiffusionTransformerModel(object):
    """A wrapper for the RDT model, which handles
            1. Model initialization
            2. Encodings of instructions
            3. Model inference
    """
    def __init__(
        self, args, 
        device='cuda',
        dtype=torch.bfloat16,
        image_size=None,
        control_frequency=20,#LIBERO dataset is 20Hz
        pretrained_text_encoder_name_or_path=None,
        pretrained_vision_encoder_name_or_path=None,
    ):
        self.args = args
        self.dtype = dtype
        self.image_size = image_size
        self.device = device
        self.control_frequency = control_frequency
        self.text_tokenizer, self.text_model = self.get_text_encoder(pretrained_text_encoder_name_or_path)
        self.image_processor, self.vision_model = self.get_vision_encoder(pretrained_vision_encoder_name_or_path)
        self.policy = self.get_policy()

        self.reset()

    def get_policy(self):
        """Initialize the model."""
        # Initialize model with arguments
        img_cond_len = (self.args["common"]["img_history_size"] 
                        * self.args["common"]["num_cameras"] 
                        * self.vision_model.num_patches)
        
        _model = RDTRunner(
            action_dim=self.args["common"]["state_dim"],
            pred_horizon=self.args["common"]["action_chunk_size"],
            config=self.args["model"],
            lang_token_dim=self.args["model"]["lang_token_dim"],
            img_token_dim=self.args["model"]["img_token_dim"],
            state_token_dim=self.args["model"]["state_token_dim"],
            max_lang_cond_len=self.args["dataset"]["tokenizer_max_length"],
            img_cond_len=img_cond_len,
            img_pos_embed_config=[
                # No initial pos embed in the last grid size
                # since we've already done in ViT
                ("image", (self.args["common"]["img_history_size"], 
                    self.args["common"]["num_cameras"], 
                    -self.vision_model.num_patches)),  
            ],
            lang_pos_embed_config=[
                # Similarly, no initial pos embed for language
                ("lang", -self.args["dataset"]["tokenizer_max_length"]),
            ],
            dtype=self.dtype,
        )

        return _model

    def get_text_encoder(self, pretrained_text_encoder_name_or_path):
        text_embedder = T5Embedder(from_pretrained=pretrained_text_encoder_name_or_path, 
                                   model_max_length=self.args["dataset"]["tokenizer_max_length"], 
                                   device=self.device)
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
        return tokenizer, text_encoder

    def get_vision_encoder(self, pretrained_vision_encoder_name_or_path):
        vision_encoder = SiglipVisionTower(vision_tower=pretrained_vision_encoder_name_or_path, args=None)
        image_processor = vision_encoder.image_processor
        return image_processor, vision_encoder

    def reset(self):
        """Set model to evaluation mode.
        """
        device = self.device
        weight_dtype = self.dtype
        self.policy.eval()
        self.text_model.eval()
        self.vision_model.eval()

        self.policy = self.policy.to(device, dtype=weight_dtype)
        self.text_model = self.text_model.to(device, dtype=weight_dtype)
        self.vision_model = self.vision_model.to(device, dtype=weight_dtype)

    def load_pretrained_weights(self, pretrained=None):
        if pretrained is None:
            return 
        print(f'Loading weights from {pretrained}')
        
        def _extract_state_dict(checkpoint):
            """Extract state_dict from various checkpoint formats."""
            # DeepSpeed format: checkpoint contains "module" key
            if isinstance(checkpoint, dict) and "module" in checkpoint:
                print("  → Detected DeepSpeed checkpoint format (with 'module' key)")
                return checkpoint["module"]
            # Direct state_dict
            elif isinstance(checkpoint, dict):
                # Check if it looks like a state_dict (has model parameter keys)
                if any(k.startswith(('model.', 'transformer.', 'diffusion.')) for k in checkpoint.keys()):
                    print("  → Detected direct state_dict format")
                    return checkpoint
                # Might be a wrapper with other keys
                elif "state_dict" in checkpoint:
                    print("  → Detected checkpoint with 'state_dict' key")
                    return checkpoint["state_dict"]
                else:
                    print("  → Using checkpoint as-is")
                    return checkpoint
            else:
                return checkpoint
        
        # Handle directory paths
        if os.path.isdir(pretrained):
            # Priority 1: Try EMA weights first (usually better performance)
            ema_path = os.path.join(pretrained, 'ema', 'pytorch_model.bin')
            if os.path.exists(ema_path):
                print("  → Found EMA weights, loading...")
                checkpoint = torch.load(ema_path, map_location='cpu')
                state_dict = _extract_state_dict(checkpoint)
                self.policy.load_state_dict(state_dict)
                print("  ✓ EMA weights loaded successfully")
                return
            
            # Priority 2: Try pytorch_model.bin (standard HuggingFace/DeepSpeed format)
            bin_path = os.path.join(pretrained, 'pytorch_model.bin')
            if os.path.exists(bin_path):
                print("  → Loading pytorch_model.bin...")
                checkpoint = torch.load(bin_path, map_location='cpu')
                state_dict = _extract_state_dict(checkpoint)
                self.policy.load_state_dict(state_dict)
                print("  ✓ Weights loaded successfully")
                return
            
            # Priority 3: Try .pt files
            pt_files = [f for f in os.listdir(pretrained) if f.endswith('.pt')]
            if pt_files:
                pt_path = os.path.join(pretrained, pt_files[0])
                print(f"  → Loading {pt_files[0]}...")
                checkpoint = torch.load(pt_path, map_location='cpu')
                state_dict = _extract_state_dict(checkpoint)
                self.policy.load_state_dict(state_dict)
                print("  ✓ Weights loaded successfully")
                return
            
            # Priority 4: Try .safetensors
            sf_files = [f for f in os.listdir(pretrained) if f.endswith('.safetensors')]
            if sf_files:
                sf_path = os.path.join(pretrained, sf_files[0])
                print(f"  → Loading {sf_files[0]}...")
                from safetensors.torch import load_model
                load_model(self.policy, sf_path)
                print("  ✓ Weights loaded successfully")
                return
            
            raise FileNotFoundError(
                f"No valid checkpoint found in directory: {pretrained}\n"
                f"Looked for: ema/pytorch_model.bin, pytorch_model.bin, *.pt, *.safetensors"
            )
        
        # Handle file paths
        filename = os.path.basename(pretrained)
        if filename.endswith('.pt') or filename.endswith('.bin'):
            checkpoint = torch.load(pretrained, map_location='cpu')
            state_dict = _extract_state_dict(checkpoint)
            self.policy.load_state_dict(state_dict)
            print("  ✓ Weights loaded successfully")
        elif filename.endswith('.safetensors'):
            from safetensors.torch import load_model
            load_model(self.policy, pretrained)
            print("  ✓ Weights loaded successfully")
        else:
            raise NotImplementedError(f"Unknown checkpoint format: {pretrained}")

    def encode_instruction(self, instruction, device="cuda"):
        """Encode string instruction to latent embeddings.

        Args:
            instruction: a string of instruction
            device: a string of device
        
        Returns:
            pred: a tensor of latent embeddings of shape (text_max_length, 512)
        """
        tokens = self.text_tokenizer(
            instruction, return_tensors="pt",
            padding="longest",
            truncation=True
        )["input_ids"].to(device)

        tokens = tokens.view(1, -1)
        with torch.no_grad():
            pred = self.text_model(tokens).last_hidden_state.detach()

        return pred

    def _format_joint_to_state(self, joints):
        """
        Format the robot joint state into the unified state vector.

        Args:
            joints (torch.Tensor): The joint state to be formatted. 
                qpos ([B, N, 14]).

        Returns:
            state (torch.Tensor): The formatted state for RDT ([B, N, 128]). 
        """
        # Max-min normalization for the gripper states (last 2 dimensions) 
        # qpos_min = -0.04245
        # qpos_max = 0.05185
        # qpos[..., -2:] = (qpos[..., -2:] - qpos_min) / \
        #     (qpos_max - qpos_min) 反归一化:
        # joints[..., -2:] = joints[..., -2:] * (gripper_max - gripper_min) + gripper_min
        
        gripper_min = -0.04245
        gripper_max = 0.05185
        joints[..., -2:] = (joints[..., -2:] - gripper_min) / \
            (gripper_max - gripper_min)

        B, N, _ = joints.shape
        state = torch.zeros(
            (B, N, self.args["model"]["state_token_dim"]), 
            device=joints.device, dtype=joints.dtype
        )
        # assemble the unifed state vector
        state[:, :, LIBERO_STATE_INDICES] = joints
        state_elem_mask = torch.zeros(
            (B, self.args["model"]["state_token_dim"]),
            device=joints.device, dtype=joints.dtype
        )
        state_elem_mask[:, LIBERO_STATE_INDICES] = 1
        return state, state_elem_mask

    def _unformat_action_to_joint(self, action):
        action_indices = LIBERO_ACTION_INDICES
        joints = action[:, :, action_indices]
        
        
        # 对 gripper 动作进行二值化（最后一个维度）
        # gripper_open: 负值表示闭合 (-1)，正值表示打开 (1)
        joints[..., -1] = torch.where(
            joints[..., -1] < 0, 
            torch.tensor(-1.0, device=joints.device), 
            torch.tensor(1.0, device=joints.device)
        )
        
        return joints

    @torch.no_grad()
    def step(self, proprio, images, text_embeds):
        """
        Args:
            proprio: proprioceptive states
            images: RGB images
            text_embeds: instruction embeddings

        Returns:
            action: predicted action
        """
        device = self.device
        dtype = self.dtype
        
        background_color = np.array([
            int(x*255) for x in self.image_processor.image_mean
        ], dtype=np.uint8).reshape(1, 1, 3)
        background_image = np.ones((
            self.image_processor.size["height"], 
            self.image_processor.size["width"], 3), dtype=np.uint8
        ) * background_color
        
        image_tensor_list = []
        for image in images:
            if image is None:
                # Replace it with the background image
                image = Image.fromarray(background_image)
            
            if self.image_size is not None:
                image = transforms.Resize(self.data_args.image_size)(image)
            
            if self.args["dataset"].get("auto_adjust_image_brightness", False):
                pixel_values = list(image.getdata())
                average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                if average_brightness <= 0.15:
                    image = transforms.ColorJitter(brightness=(1.75,1.75))(image)
                    
            if self.args["dataset"].get("image_aspect_ratio", "pad") == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_tensor_list.append(image)

        image_tensor = torch.stack(image_tensor_list, dim=0).to(device, dtype=dtype)

        image_embeds = self.vision_model(image_tensor).detach()
        image_embeds = image_embeds.reshape(-1, self.vision_model.hidden_size).unsqueeze(0)

        # history of actions
        joints = proprio.to(device).unsqueeze(0).unsqueeze(0)   # (1, 1, 9)
        states, state_elem_mask = self._format_joint_to_state(joints)    # (1, 1, 128), (1, 128)
        states, state_elem_mask = states.to(device, dtype=dtype), state_elem_mask.to(device, dtype=dtype)
        states = states[:, -1:, :]  # (1, 1, 128)
        ctrl_freqs = torch.tensor([self.control_frequency]).to(device)
        
        # 创建正确的 action_mask（7 维动作：6D velocity + gripper）
        action_elem_mask = torch.zeros((1, self.args["model"]["state_token_dim"]), device=device, dtype=dtype)
        action_elem_mask[:, LIBERO_ACTION_INDICES] = 1
        
        text_embeds = text_embeds.to(device, dtype=dtype)
        
        trajectory = self.policy.predict_action(
            lang_tokens=text_embeds,
            lang_attn_mask=torch.ones(
                text_embeds.shape[:2], dtype=torch.bool,
                device=text_embeds.device),
            img_tokens=image_embeds,
            state_tokens=states,
            action_mask=action_elem_mask.unsqueeze(1),  # 使用正确的 action mask
            ctrl_freqs=ctrl_freqs
        )
        trajectory = self._unformat_action_to_joint(trajectory).to(torch.float32)

        return trajectory
