"""Preserve verl GSM8K messages/labels and select a fixed held-out subset.

Run inside the prepared container. Inputs are read-only existing parquet files;
output files are created exclusively under a new experiment input directory.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _convert(rows, split):
    converted = []
    for index, row in enumerate(rows):
        prompt = row["prompt"]
        label = row["reward_model"]["ground_truth"]
        assert row["data_source"] == "openai/gsm8k"
        assert len(prompt) == 1 and prompt[0]["role"] == "user"
        assert 'output the final answer after "####".' in prompt[0]["content"]
        expected = row["extra_info"]["answer"].rsplit("####", 1)[1].strip().replace(",", "")
        assert label == expected, (split, index, label, expected)
        converted.append({"prompt": prompt, "label": label,
                          "metadata": {"data_source": row["data_source"], "split": split,
                                       "source_index": index}})
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    source_paths = {split: args.source_dir / f"{split}.parquet" for split in ("train", "test")}
    rows = {split: pq.read_table(path, use_threads=False, pre_buffer=False).to_pylist()
            for split, path in source_paths.items()}
    assert (len(rows["train"]), len(rows["test"])) == (7473, 1319)
    train_questions = {row["extra_info"]["question"] for row in rows["train"]}
    test_questions = {row["extra_info"]["question"] for row in rows["test"]}
    assert not train_questions & test_questions
    converted = {split: _convert(records, split) for split, records in rows.items()}
    indices = sorted(random.Random(args.seed).sample(range(len(rows["test"])), args.eval_size))
    outputs = {"train.jsonl": converted["train"],
               f"test-fixed-{args.eval_size}.jsonl": [converted["test"][i] for i in indices]}
    report = {"source_sha256": {split: _sha256(path) for split, path in source_paths.items()},
              "source_paths": {split: str(path) for split, path in source_paths.items()},
              "model_path": str(args.model_path), "seed": args.seed,
              "test_source_indices": indices, "train_test_question_overlap": 0,
              "prompt_changes": "none; original single-user-message verl prompt preserved",
              "label_changes": "none", "chat_template_kwargs": {}, "outputs": {}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in outputs.items():
        lengths = [len(tokenizer.apply_chat_template(row["prompt"], tokenize=True,
                       add_generation_prompt=True, return_dict=False)) for row in records]
        assert max(lengths) <= 512, (name, max(lengths))
        path = args.output_dir / name
        with path.open("x") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=True) + "\n")
        report["outputs"][name] = {"rows": len(records), "sha256": _sha256(path),
                                  "min_prompt_tokens": min(lengths), "max_prompt_tokens": max(lengths)}
    with (args.output_dir / "manifest.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
