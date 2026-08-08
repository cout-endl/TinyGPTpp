import os
import pickle
import torch
import tiktoken

from config.config import RunConfig
from model.base_gpt import GPT
from utils.logger import get_logger


class Generator:
    def __init__(self, run_config: RunConfig):
        self.run_config = run_config
        self.device = run_config.device
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'

        # 初始化日志
        log_path = os.path.join(run_config.out_dir, "inference.log")
        self.logger = get_logger("generator", log_file=log_path)

        # 加载模型与编码器
        self._load_checkpoint()
        self._init_encoder()

    def _load_checkpoint(self):
        """加载训练好的检查点，恢复模型与配置"""
        ckpt_path = os.path.join(self.run_config.out_dir, 'ckpt.pt')
        self.logger.info(f"Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self.model_config = checkpoint['model_config']
        self.model = GPT(self.model_config)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()
        self.logger.info("Model loaded successfully")

    def _init_encoder(self):
        """初始化编码器：优先用数据集字符编码，否则用GPT-2 BPE"""
        data_dir = os.path.join('data', self.run_config.dataset)
        meta_path = os.path.join(data_dir, 'meta.pkl')

        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            stoi, itos = meta['stoi'], meta['itos']
            self.encode = lambda s: [stoi[c] for c in s]
            self.decode = lambda l: ''.join([itos[i] for i in l])
            self.logger.info("Using character-level encoder from dataset")
        else:
            self.enc = tiktoken.get_encoding("gpt2")
            self.encode = lambda s: self.enc.encode(s, allowed_special={"<|endoftext|>"})
            self.decode = lambda l: self.enc.decode(l)
            self.logger.info("Using GPT-2 BPE encoder (tiktoken)")

    def generate(self, start: str = "\n", num_samples: int = 10,
                max_new_tokens: int = 500, temperature: float = 1.0,
                top_k: int | None = None, top_p: float | None = None):
        """批量生成文本，支持 temperature、top_k、top_p"""
        start_ids = self.encode(start)
        x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]

        self.logger.info(
            f"Generating {num_samples} samples | max_new_tokens={max_new_tokens} | "
            f"temperature={temperature} | top_k={top_k} | top_p={top_p}"
        )

        with torch.no_grad():
            for k in range(num_samples):
                y = self.model.generate(x, max_new_tokens,
                                        temperature=temperature,
                                        top_k=top_k,
                                        top_p=top_p)
                print(self.decode(y[0].tolist()))
                print('-' * 50)

        self.logger.info("Generation finished")
