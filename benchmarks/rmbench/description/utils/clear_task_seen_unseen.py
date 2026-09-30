from argparse import ArgumentParser
import json

from benchmarks.rmbench.paths import description_root


def clear_seen_unseen(task_name):
    task_path = description_root() / "task_instruction" / f"{task_name}.json"
    with task_path.open("r") as f:
        task_info_json = f.read()
    # print(task_info_json)
    task_info = json.loads(task_info_json)
    task_info["seen"] = []
    task_info["unseen"] = []
    with task_path.open("w") as f:
        json.dump(task_info, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("task_name", type=str, default="beat_block_hammer")
    args = parser.parse_args()
    clear_seen_unseen(args.task_name)
