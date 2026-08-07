import argparse
from config.config import GPTConfig, TrainConfig, RunConfig
from train.trainer import Trainer

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description="TinyGPT++ Training (Phase 0)")

    # 模型结构参数
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--bias', type=str2bool, default=True)

    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=6e-4)
    parser.add_argument('--max_iters', type=int, default=600000)
    parser.add_argument('--weight_decay', type=float, default=1e-1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # 学习率调度
    parser.add_argument('--decay_lr', type=bool, default=True)
    parser.add_argument('--warmup_iters', type=int, default=2000)
    parser.add_argument('--lr_decay_iters', type=int, default=600000)
    parser.add_argument('--min_lr', type=float, default=6e-5)

    # 评估与梯度累积
    parser.add_argument('--eval_interval', type=int, default=2000)
    parser.add_argument('--eval_iters', type=int, default=200)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)

    # 运行参数
    parser.add_argument('--dataset', type=str, default='shakespeare_char')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out_dir', type=str, default='out')
    parser.add_argument('--seed', type=int, default=1337)

    args = parser.parse_args()

    # 组装配置
    model_cfg = GPTConfig(
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        block_size=args.block_size, dropout=args.dropout, bias=args.bias
    )
    train_cfg = TrainConfig(
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        max_iters=args.max_iters, weight_decay=args.weight_decay,
        beta1=args.beta1, beta2=args.beta2, grad_clip=args.grad_clip,
        decay_lr=args.decay_lr, warmup_iters=args.warmup_iters,
        lr_decay_iters=args.lr_decay_iters, min_lr=args.min_lr,
        eval_interval=args.eval_interval, eval_iters=args.eval_iters,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    run_cfg = RunConfig(
        dataset=args.dataset, device=args.device,
        out_dir=args.out_dir, seed=args.seed
    )

    # 启动训练
    trainer = Trainer(model_cfg, train_cfg, run_cfg)
    trainer.train()


if __name__ == '__main__':
    main()
