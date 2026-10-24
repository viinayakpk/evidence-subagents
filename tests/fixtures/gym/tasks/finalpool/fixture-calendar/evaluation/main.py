import argparse
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_workspace", required=True)
    parser.add_argument("--groundtruth_workspace", required=True)
    parser.add_argument("--launch_time", default="")
    parser.add_argument("--res_log_file", default="")
    args = parser.parse_args()
    expected = (Path(args.groundtruth_workspace) / "answer.txt").read_text(encoding="utf-8").strip()
    agent = Path(args.agent_workspace) / "answer.txt"
    actual = agent.read_text(encoding="utf-8") if agent.is_file() else ""
    ok = expected in actual
    if args.res_log_file:
        Path(args.res_log_file).write_text("passed\n" if ok else "failed\n", encoding="utf-8")
    print("accuracy", 1.0 if ok else 0.0)
    raise SystemExit(0 if ok else 1)

if __name__ == "__main__":
    main()
