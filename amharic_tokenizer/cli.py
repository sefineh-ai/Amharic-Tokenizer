import argparse
from .pipeline import train_and_save


def main():
    parser = argparse.ArgumentParser(prog="amh-tokenizer", description="Amharic BPE-like tokenizer utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train tokenizer and save model")
    p_train.add_argument("corpus", help="Path to cleaned Amharic text corpus")
    p_train.add_argument("output_prefix", help="Output path prefix for saved model")
    p_train.add_argument("--num-merges", type=int, default=50000, help="Number of BPE merges to learn")
    p_train.add_argument("--verbose", action="store_true", help="Print progress during training")
    p_train.add_argument("--log-every", type=int, default=1000, help="Print status every N merges when verbose")

    args = parser.parse_args()
    if args.cmd == "train":
        print(f"[AMH-Tokenizer] Loading corpus: {args.corpus}")
        print(f"[AMH-Tokenizer] Training with num_merges={args.num_merges}")
        merges = train_and_save(
            args.corpus,
            args.output_prefix,
            num_merges=args.num_merges,
            verbose=args.verbose,
            log_every=args.log_every,
        )
        print(f"[AMH-Tokenizer] Saved model to {args.output_prefix}.json with {merges} merges")


if __name__ == "__main__":
    main()


