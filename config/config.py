from dataclasses import dataclass

@dataclass
class GPTConfig:
    """模型结构配置，对应原命令行 --n_layer、--n_head 等参数"""
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    vocab_size: int = 50304
    dropout: float = 0.0
    bias: bool = True

    # ========== 后续Phase扩展字段，默认关闭 ==========
    use_rope: bool = False
    n_kv_head: int | None = None  # None 表示与n_head一致（标准MHA）


@dataclass
class TrainConfig:
    """训练超参数配置，对应原命令行训练相关参数"""
    batch_size: int = 64
    learning_rate: float = 6e-4
    max_iters: int = 600000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # 评估配置
    eval_interval: int = 2000
    eval_iters: int = 200

    # 学习率调度
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5

    # 梯度累积
    gradient_accumulation_steps: int = 5 * 8

    # ========== 后续Phase扩展字段，默认关闭 ==========
    use_amp: bool = False
    use_grad_checkpoint: bool = False


@dataclass
class RunConfig:
    """运行时通用配置"""
    dataset: str = "shakespeare_char"
    device: str = "cuda"
    out_dir: str = "out"
    seed: int = 1337
    log_file: str = "run.log"
