import os
import pickle
import math
import time
import numpy as np
import torch
import inspect
import dataclasses

from config.config import GPTConfig, TrainConfig, RunConfig
from model.base_gpt import GPT
from utils.logger import get_logger

import dataclasses


class Trainer:
    def __init__(self, model_config: GPTConfig, train_config: TrainConfig, run_config: RunConfig):
        self.model_config = model_config
        self.train_config = train_config
        self.run_config = run_config
        # 初始化日志
        log_path = os.path.join(run_config.out_dir, run_config.log_file)
        self.logger = get_logger("trainer", log_file=log_path)
        all_cfg = {}
        # 合并三类配置
        for field in dataclasses.fields(GPTConfig):
            all_cfg[field.name] = getattr(model_config, field.name)
        for field in dataclasses.fields(TrainConfig):
            all_cfg[field.name] = getattr(train_config, field.name)
        for field in dataclasses.fields(RunConfig):
            all_cfg[field.name] = getattr(run_config, field.name)
        show_keys = [
            "dataset", "n_layer", "n_head", "n_embd", "block_size", "batch_size",
            "learning_rate", "max_iters", "eval_interval", "eval_iters",
            "dropout", "bias", "device"
        ]
        for k in show_keys:
            if k in all_cfg:
                val = all_cfg[k]
                # 浮点数统一显示 0.001 而非 1e-3
                if isinstance(val, float):
                    self.logger.info(f"Overriding: {k} = {val:.3f}".rstrip("0").rstrip(".") if "." in f"{val:.3f}" else f"Overriding: {k} = {val:.3f}")
                else:
                    self.logger.info(f"Overriding: {k} = {val}")
      
        # 基础环境设置
        self.device = run_config.device
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        self._setup_seed_and_backend()

        # 加载数据集
        self._load_dataset()

        # 初始化模型与优化器
        self._init_model()
        self._init_optimizer()

        # 训练状态
        self.iter_num = 0
        self.best_val_loss = 1e9

    def _setup_seed_and_backend(self):
        """设置随机种子与CUDA后端，与原版完全一致"""
        torch.manual_seed(self.run_config.seed)
        torch.cuda.manual_seed(self.run_config.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def _load_dataset(self):
        """加载二进制数据集与词汇表元信息"""
        data_dir = os.path.join('data', self.run_config.dataset)
        self.train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        self.val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

        # 从meta.pkl读取词汇表大小
        meta_path = os.path.join(data_dir, 'meta.pkl')
        self.meta_vocab_size = None
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.meta_vocab_size = meta['vocab_size']
            self.model_config.vocab_size = self.meta_vocab_size
            self.logger.info(f"Loaded dataset vocab_size = {self.meta_vocab_size}")
        tokens_per_iter = self.train_config.batch_size * self.model_config.block_size * self.train_config.gradient_accumulation_steps
        self.logger.info(f"tokens per iteration will be: {tokens_per_iter:,}")

    def _init_model(self):
        self.logger.info("Initializing a new model from scratch")
        self.model = GPT(self.model_config)
        self.model.to(self.device)
        param_m = self.model.get_num_params() / 1e6
        self.logger.info(f"number of parameters: {param_m:.2f}M")

    def _init_optimizer(self):
        self.logger.info("Configuring AdamW optimizer...")
        self.optimizer = self.model.configure_optimizers(
            weight_decay=self.train_config.weight_decay,
            learning_rate=self.train_config.learning_rate,
            betas=(self.train_config.beta1, self.train_config.beta2),
            device_type=self.device_type
        )
        opt_groups = self.optimizer.param_groups
        decay_cnt = len(opt_groups[0]["params"])
        decay_params_num = sum(p.numel() for p in opt_groups[0]["params"])
        nodecay_cnt = len(opt_groups[1]["params"])
        nodecay_params_num = sum(p.numel() for p in opt_groups[1]["params"])
        self.logger.info(f"num decayed parameter tensors: {decay_cnt}, with {decay_params_num:,} parameters")
        self.logger.info(f"num non-decayed parameter tensors: {nodecay_cnt}, with {nodecay_params_num:,} parameters")
        # fused AdamW 判断打印
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and self.device_type == 'cuda'
        self.logger.info(f"using fused AdamW: {use_fused}")

    def get_batch(self, split):
        """获取一个batch的数据，与原版逻辑完全一致"""
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.model_config.block_size, (self.train_config.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self.model_config.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.model_config.block_size]).astype(np.int64)) for i in ix])

        if self.device_type == 'cuda':
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self):
        """评估训练集和验证集损失"""
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.train_config.eval_iters)
            for k in range(self.train_config.eval_iters):
                X, Y = self.get_batch(split)
                _, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def _get_lr(self, it):
        """cosine学习率调度，带warmup，与原版完全一致"""
        if not self.train_config.decay_lr:
            return self.train_config.learning_rate
        if it < self.train_config.warmup_iters:
            return self.train_config.learning_rate * (it + 1) / self.train_config.warmup_iters
        if it > self.train_config.lr_decay_iters:
            return self.train_config.min_lr
        decay_ratio = (it - self.train_config.warmup_iters) / (self.train_config.lr_decay_iters - self.train_config.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.train_config.min_lr + coeff * (self.train_config.learning_rate - self.train_config.min_lr)

    def _save_checkpoint(self):
        """保存模型检查点，包含所有配置与训练状态"""
        ckpt_path = os.path.join(self.run_config.out_dir, 'ckpt.pt')
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'model_config': self.model_config,
            'train_config': self.train_config,
            'run_config': self.run_config,
            'iter_num': self.iter_num,
            'best_val_loss': self.best_val_loss,
        }
        torch.save(checkpoint, ckpt_path)
        self.logger.info(f"Checkpoint saved to {ckpt_path}")

    def train(self):
        """主训练循环，与原版逻辑完全对齐"""
        self.logger.info("=" * 50)
        self.logger.info("Training started")
        self.logger.info("=" * 50)
        if torch.__version__ >= "2.0.0" and self.device_type == "cuda":
            self.logger.info("compiling the model... (takes a ~minute)")
            # self.model = torch.compile(self.model)
        self.model.train()

        X, Y = self.get_batch('train')
        t0 = time.time()
        local_iter = 0
        running_mfu = -1.0

        while True:
            # 更新学习率
            lr = self._get_lr(self.iter_num)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # 定期评估+保存检查点
            if self.iter_num % self.train_config.eval_interval == 0:
                losses = self.estimate_loss()
                self.logger.info(
                    f"step {self.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
                if losses['val'] < self.best_val_loss and self.iter_num > 0:
                    self.best_val_loss = losses['val']
                    self._save_checkpoint()

            # 梯度累积前向反向
            for micro_step in range(self.train_config.gradient_accumulation_steps):
                logits, loss = self.model(X, Y)
                loss = loss / self.train_config.gradient_accumulation_steps
                # 预取下一个batch，重叠计算与数据搬运
                X, Y = self.get_batch('train')
                loss.backward()

            # 梯度裁剪
            if self.train_config.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.grad_clip)

            # 优化器步进
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            # 计时与日志
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            lossf = loss.item() * self.train_config.gradient_accumulation_steps
            if local_iter >= 5:
                mfu = self.model.estimate_mfu(
                    self.train_config.batch_size * self.train_config.gradient_accumulation_steps,
                    dt
                )
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            self.logger.info(
                f"iter {self.iter_num:d}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
            )

            self.iter_num += 1
            local_iter += 1

            # 终止条件
            if self.iter_num > self.train_config.max_iters:
                self.logger.info("Training finished")
                break
