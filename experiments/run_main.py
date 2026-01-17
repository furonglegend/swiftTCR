import argparse
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--phase", type=str, choices=["1", "2", "eval"], required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.phase == "1":
        print("Running Phase 1: Adapter estimation")
    elif args.phase == "2":
        print("Running Phase 2: Memory construction and retrieval")
    else:
        print("Running evaluation")


if __name__ == "__main__":
    main()
