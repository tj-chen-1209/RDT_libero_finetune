#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import json
import copy
import logging
import math
import os
from pathlib import Path

import diffusers
import torch
import torch.utils.checkpoint
import transformers
import yaml
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from safetensors.torch import load_model

from models.ema_model import EMAModel
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from models.rdt_runner import RDTRunner
from train.dataset_sft import DataCollatorForVLAConsumerDataset, VLAConsumerDataset
from train.sample import log_sample_res


if is_wandb_available():
    import wandb


def save_model_card(repo_id: str, base_model=str, repo_folder=None):
    yaml = f"""
---
license: mit
base_model: {base_model}
language:
- en
pipeline_tag: robotics
library_name: transformers
tags:
- robotics
- pytorch
- multimodal
- pretraining
- vla
- diffusion
- rdt
---
    """
    model_card = f"""
# RDT - {repo_id}

This is a RDT model derived from {base_model}. The weights were trained using [RDT](https://rdt-robotics.github.io/rdt-robotics/).
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def train(args, logger):
    # 读取配置文件，去读取configs/base.yaml文件中的配置
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit)
    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(
            hf_ds_config=args.deepspeed
        ) if args.deepspeed is not None else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if args.precomp_lang_embed:
        tokenizer, text_encoder = None, None
    else:
        # 使用T5编码器，去编码语言
        text_embedder = T5Embedder(from_pretrained=args.pretrained_text_encoder_name_or_path,
                                   model_max_length=config["dataset"]["tokenizer_max_length"], device=accelerator.device)
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    # 使用SigLIP编码器，去编码图像
    vision_encoder = SiglipVisionTower(
        vision_tower=args.pretrained_vision_encoder_name_or_path, args=None)
    image_processor = vision_encoder.image_processor

    # 加载预训练模型
    if (
        args.pretrained_model_name_or_path is not None
        and not os.path.isfile(args.pretrained_model_name_or_path)
    ):
        logger.info("Constructing model from pretrained checkpoint.")
        ckpt_path = args.pretrained_model_name_or_path
        cfg_file = os.path.join(ckpt_path, "config.json")
        with open(cfg_file, "r") as f:
            ckpt_cfg = json.load(f)

        model_kwargs = dict(
            action_dim=ckpt_cfg["action_dim"],
            pred_horizon=ckpt_cfg["pred_horizon"],
            config={
                "rdt": ckpt_cfg["rdt"],
                "lang_adaptor": ckpt_cfg["lang_adaptor"],
                "img_adaptor": ckpt_cfg["img_adaptor"],
                "state_adaptor": ckpt_cfg["state_adaptor"],
                "noise_scheduler": ckpt_cfg["noise_scheduler"],
            },
            lang_token_dim=ckpt_cfg["lang_token_dim"],
            img_token_dim=ckpt_cfg["img_token_dim"],
            state_token_dim=ckpt_cfg["state_token_dim"],
            max_lang_cond_len=ckpt_cfg["max_lang_cond_len"],
            img_cond_len=ckpt_cfg["img_cond_len"],
            lang_pos_embed_config=ckpt_cfg.get("lang_pos_embed_config"),
            img_pos_embed_config=ckpt_cfg.get("img_pos_embed_config"),
            dtype=weight_dtype,
        )
        rdt = RDTRunner.from_pretrained(ckpt_path, **model_kwargs)
        
        # 应用 LoRA (如果启用)
        if args.use_lora:
            from peft import LoraConfig, get_peft_model
            
            # 根据用户选择定义目标模块
            if args.lora_target_modules == "all":
                target_modules = [
                    "attn.qkv", "attn.proj",           # 自注意力
                    "cross_attn.q", "cross_attn.kv", "cross_attn.proj",  # 交叉注意力
                    "ffn.fc1", "ffn.fc2",               # FFN
                    "lang_adaptor.0","lang_adaptor.2",
                    "img_adaptor.0","img_adaptor.2",
                    "state_adaptor.0","state_adaptor.2","state_adaptor.4" # 条件适配器
                ]
            
            lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=args.lora_dropout,
                bias="none"
                # 不指定 task_type - RDT 是扩散变换器，不属于标准任务类型
            )
            
            rdt = get_peft_model(rdt, lora_config)
            rdt.print_trainable_parameters()
            logger.info(f"LoRA enabled: rank={args.lora_rank}, alpha={args.lora_alpha}, "
                       f"target_modules={args.lora_target_modules}")
            logger.info(f"Targeting these module patterns: {target_modules}")
    else:
        logger.info("Constructing model from provided config.")
        # 计算图像条件长度
        img_cond_len = (config["common"]["img_history_size"]
                        * config["common"]["num_cameras"]
                        * vision_encoder.num_patches)
        # 创建RDT模型
        rdt = RDTRunner(
            action_dim=config["common"]["state_dim"],
            pred_horizon=config["common"]["action_chunk_size"],
            config=config["model"],
            lang_token_dim=config["model"]["lang_token_dim"],
            img_token_dim=config["model"]["img_token_dim"],
            state_token_dim=config["model"]["state_token_dim"],
            max_lang_cond_len=config["dataset"]["tokenizer_max_length"],
            img_cond_len=img_cond_len,
            img_pos_embed_config=[
                # 在最后一个网格大小中没有初始位置嵌入
                # 因为我们已经在ViT中完成了
                ("image", (config["common"]["img_history_size"],
                           config["common"]["num_cameras"],
                           -vision_encoder.num_patches)),
            ],
            lang_pos_embed_config=[
                # 同样，语言没有初始位置嵌入
                ("lang", -config["dataset"]["tokenizer_max_length"]),
            ],
            dtype=weight_dtype,
        )

    # 复制RDT模型，用于EMA
    ema_rdt = copy.deepcopy(rdt)
    # 创建EMA模型
    ema_model = EMAModel(
        ema_rdt,
        update_after_step=config["model"]["ema"]["update_after_step"],
        inv_gamma=config["model"]["ema"]["inv_gamma"],
        power=config["model"]["ema"]["power"],
        min_value=config["model"]["ema"]["min_value"],
        max_value=config["model"]["ema"]["max_value"]
    )

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    # which ensure saving model in huggingface format (config.json + pytorch_model.bin)
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                model_to_save = model.module if hasattr(
                    model, "module") else model  # type: ignore
                if isinstance(model_to_save, type(accelerator.unwrap_model(rdt))):
                    if args.use_lora:
                        # 保存LoRA权重（轻量级，只有几MB）
                        logger.info(f"Saving LoRA weights to {output_dir}")
                        model_to_save.save_pretrained(output_dir)
                    else:
                        # 保存完整模型
                        model_to_save.save_pretrained(output_dir)

    accelerator.register_save_state_pre_hook(save_model_hook)

    if args.gradient_checkpointing:
        # TODO:
        raise NotImplementedError(
            "Gradient checkpointing is not yet implemented.")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps *
            args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # 创建优化器，优化器使用AdamW优化器
    params_to_optimize = rdt.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 创建训练数据集和数据加载器
    train_dataset = VLAConsumerDataset(
        config=config["dataset"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["img_history_size"],
        dataset_type=args.dataset_type,
        image_aug=args.image_aug,
        cond_mask_prob=args.cond_mask_prob,
        cam_ext_mask_prob=args.cam_ext_mask_prob,
        state_noise_snr=args.state_noise_snr,
        use_hdf5=args.load_from_hdf5,
        use_precomp_lang_embed=args.precomp_lang_embed,
    )
    sample_dataset = VLAConsumerDataset(
        config=config["dataset"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["img_history_size"],
        dataset_type=args.dataset_type,
        image_aug=False,
        cond_mask_prob=0,
        cam_ext_mask_prob=-1,
        state_noise_snr=None,
        use_hdf5=args.load_from_hdf5,
        use_precomp_lang_embed=args.precomp_lang_embed,
    )

    data_collator = DataCollatorForVLAConsumerDataset(tokenizer)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    sample_dataloader = torch.utils.data.DataLoader(
        sample_dataset,
        batch_size=args.sample_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    # 创建学习率调度器，学习率调度器使用常数学习率调度器
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # 准备所有内容，使用`accelerator`进行加速
    rdt, optimizer, train_dataloader, sample_dataloader, lr_scheduler = accelerator.prepare(
        rdt, optimizer, train_dataloader, sample_dataloader, lr_scheduler
    )

    ema_rdt.to(accelerator.device, dtype=weight_dtype)

    if text_encoder is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    if vision_encoder is not None:
        vision_encoder.vision_tower.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(
            "roboticDiffusionTransformer", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Load from a pretrained checkpoint
    if (
        args.resume_from_checkpoint is None
        and args.pretrained_model_name_or_path is not None
        and os.path.isfile(args.pretrained_model_name_or_path)
    ):
        # Since EMA is deprecated, we do not load EMA from the pretrained checkpoint
        logger.info("Loading from a pretrained checkpoint.")
        checkpoint = torch.load(args.pretrained_model_name_or_path)
        rdt.module.load_state_dict(checkpoint["module"])

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            try:
                # load_module_strict=False
                accelerator.load_state(os.path.join(args.output_dir, path))
            except:
                # load deepspeed's state_dict
                logger.info(
                    "Resuming training state failed. Attempting to only load from model checkpoint.")
                checkpoint = torch.load(os.path.join(
                    args.output_dir, path, "pytorch_model", "mp_rank_00_model_states.pt"))
                rdt.module.load_state_dict(checkpoint["module"])

            load_model(ema_rdt, os.path.join(
                args.output_dir, path, "ema", "model.safetensors"))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    loss_for_log = {}
    # 训练循环
    for epoch in range(first_epoch, args.num_train_epochs):
        # Pytorch模型设置为训练模式
        rdt.train()

        # 设置进度条到正确的位置
        if args.resume_from_checkpoint and epoch == first_epoch:
            progress_bar.update(
                resume_step // args.gradient_accumulation_steps)

        # ========== 训练循环：前向传播和反向传播 ==========
        for batch in train_dataloader:
            # 使用Accumulator进行梯度累积（支持多GPU和梯度累积）
            with accelerator.accumulate(rdt):
                # 步骤1：准备输入数据
                # (B, img_history*num_cam, C, H, W)
                images = batch["images"].to(dtype=weight_dtype)
                # (B, T, D_a) 可能包含多个时间步的状态
                states = batch["states"].to(dtype=weight_dtype)
                # 只使用最后一个状态作为输入（当前时刻的状态）
                states = states[:, -1:, :]  # (B, 1, D_a)
                actions = batch["actions"].to(
                    dtype=weight_dtype)  # (B, horizon, D_a) 未来动作序列
                state_elem_mask = batch["state_elem_mask"].to(
                    dtype=weight_dtype)  # (B, 1, D_a) 动作mask
                ctrl_freqs = batch["ctrl_freqs"]  # (B,) 控制频率

                # 步骤2：编码多模态输入（冻结编码器，不计算梯度）
                with torch.no_grad():
                    batch_size, _, C, H, W = images.shape
                    # 图像编码：使用SigLIP编码器
                    # 将所有图像展平后编码，再重塑为批次格式
                    image_embeds = vision_encoder(
                        images.reshape(-1, C, H, W)).detach()
                    image_embeds = image_embeds.reshape(
                        (batch_size, -1, vision_encoder.hidden_size))

                    lang_attn_mask = batch["lang_attn_mask"]
                    # 语言编码：使用T5编码器或预计算嵌入
                    text_embeds = batch["lang_embeds"].to(dtype=weight_dtype) \
                        if args.precomp_lang_embed \
                        else text_encoder(
                            input_ids=batch["input_ids"],
                            attention_mask=lang_attn_mask
                    )["last_hidden_state"].detach()

                # 步骤3：扩展状态元素mask的维度
                state_elem_mask = state_elem_mask.unsqueeze(
                    1)  # (B, 1, 1, D_a)

                # 步骤4：前向传播并计算损失
                # rdt.forward() 调用 compute_loss()
                # Pytorch模型调用forward方法，计算损失
                loss = rdt(
                    lang_tokens=text_embeds,      # (B, L_lang, lang_token_dim)
                    lang_attn_mask=lang_attn_mask,  # (B, L_lang)
                    img_tokens=image_embeds,      # (B, L_img, img_token_dim)
                    state_tokens=states,          # (B, 1, state_token_dim)
                    # (B, horizon, state_token_dim)
                    action_gt=actions,
                    action_mask=state_elem_mask,  # (B, 1, state_token_dim)
                    ctrl_freqs=ctrl_freqs         # (B,)
                )

                # 步骤5：反向传播
                accelerator.backward(loss)

                # 步骤6：梯度裁剪和优化器更新
                if accelerator.sync_gradients:
                    params_to_clip = rdt.parameters()
                    accelerator.clip_grad_norm_(
                        params_to_clip, args.max_grad_norm)  # 防止梯度爆炸
                optimizer.step()  # 更新参数
                lr_scheduler.step()  # 更新学习率
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)  # 清零梯度

            # 步骤7：更新EMA模型（指数移动平均，用于稳定推理）
            ema_model.step(accelerator.unwrap_model(rdt))

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_period == 0:
                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    ema_save_path = os.path.join(save_path, f"ema")
                    accelerator.save_model(ema_rdt, ema_save_path)
                    logger.info(f"Saved state to {save_path}")

                if args.sample_period > 0 and global_step % args.sample_period == 0:
                    sample_loss_for_log = log_sample_res(
                        text_encoder,
                        vision_encoder,
                        rdt,    # We do not use EMA currently
                        args,
                        accelerator,
                        weight_dtype,
                        sample_dataset.get_dataset_id2name(),
                        sample_dataloader,
                        logger,
                    )
                    logger.info(sample_loss_for_log)
                    accelerator.log(sample_loss_for_log, step=global_step)

            logs = {"loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            logs.update(loss_for_log)
            # logger.info(logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(rdt).save_pretrained(args.output_dir)
        ema_save_path = os.path.join(args.output_dir, f"ema")
        accelerator.save_model(ema_rdt, ema_save_path)

        logger.info(f"Saved Model to {args.output_dir}")

        if args.push_to_hub:
            save_model_card(
                repo_id,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                token=args.hub_token,
                allow_patterns=["pytorch_model.bin", "*.json", "*.md"],
                # ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()
