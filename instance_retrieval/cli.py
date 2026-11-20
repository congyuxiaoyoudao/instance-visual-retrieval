"""Small CLI; algorithms live in the package, not in shell scripts."""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    deep = sub.add_parser("deep")
    deep.add_argument("--methods", nargs="+")
    deep.add_argument("--datasets", nargs="+", default=["roxford5k", "rparis6k"])
    deep.add_argument("--device", default=None)
    local = sub.add_parser("local")
    local.add_argument("--sizes", nargs="+", type=int, default=[64, 256])
    local.add_argument("--top-n", type=int, default=100)
    local.add_argument("--workers", type=int, default=8)
    sub.add_parser("eval")
    sub.add_parser("tables")
    sub.add_parser("failures")
    demo = sub.add_parser("demo")
    demo.add_argument("--dataset", default="roxford5k")
    demo.add_argument("--method", default="dino_s14_518_cls")
    demo.add_argument("--port", type=int, default=17861)
    args = parser.parse_args()
    from .experiments import run_deep, run_local, reevaluate, tables
    if args.command == "deep":
        run_deep(args.root, args.methods, args.datasets, args.device)
    elif args.command == "local":
        run_local(args.root, args.sizes, args.top_n, args.workers)
    elif args.command == "eval":
        print(reevaluate(args.root).round(1).to_string(index=False))
    elif args.command == "tables":
        print(tables(args.root).round(1).to_string(index=False))
    elif args.command == "failures":
        from .analysis import prepare_failures, failure_summary
        prepare_failures(args.root)
        print(failure_summary(args.root))
    elif args.command == "demo":
        from .demo import RetrievalDemo
        RetrievalDemo(args.root, args.dataset, args.method).build().launch(server_name="127.0.0.1", server_port=args.port, share=False)

if __name__ == "__main__":
    main()
