import argparse
from config.config import RunConfig
from inference.generator import Generator


def main():
    parser = argparse.ArgumentParser(description="TinyGPT++ Inference (Phase 0)")

    parser.add_argument('--out_dir', type=str, default='out')
    parser.add_argument('--start', type=str, default='\n')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--max_new_tokens', type=int, default=500)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--dataset', type=str, default='shakespeare_char')

    args = parser.parse_args()

    run_cfg = RunConfig(
        out_dir=args.out_dir, device=args.device,
        seed=args.seed, dataset=args.dataset
    )

    generator = Generator(run_cfg)
    generator.generate(
        start=args.start,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )


if __name__ == '__main__':
    main()
