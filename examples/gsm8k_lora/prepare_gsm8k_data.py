import re

from datasets import load_dataset
import pyarrow.parquet as pq
from rllm.data.dataset import DatasetRegistry


# Adapted from verl/examples/data_preprocess/gsm8k.py
def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


import os
def save_as_parquet(dataset, path):
    table = dataset.data.table
    pq.write_table(table, path)

def prepare_gsm8k_data():
    gsm8k_dataset = load_dataset("openai/gsm8k", "main")
    train_dataset = gsm8k_dataset["train"]
    test_dataset = gsm8k_dataset["test"]

    def preprocess_fn(example, idx):
        return {
            "prompt": [
                {"role":"user", "content": example["question"]}
            ],
            "question": example["question"],
            "ground_truth": extract_solution(example["answer"]),
            "data_source": "gsm8k",
        }

    train_dataset = train_dataset.map(preprocess_fn, with_indices=True)
    test_dataset = test_dataset.map(preprocess_fn, with_indices=True)
    
    # TODO: replace (username) with your username
    save_dir = "/home/ray/rllm/~/data/rlhf/gsm8k"
    os.makedirs(save_dir, exist_ok=True)
    save_as_parquet(train_dataset, f"{save_dir}/train.parquet")
    save_as_parquet(test_dataset, f"{save_dir}/test.parquet")

    train_dataset = DatasetRegistry.register_dataset("gsm8k", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("gsm8k", test_dataset, "test")

    
    
     # 🔽 SAVE AS PARQUET
    # train_dataset.to_parquet(f"{save_dir}/train.parquet")
    # test_dataset.to_parquet(f"{save_dir}/test.parquet")
    

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_gsm8k_data()
    print(train_dataset)
    print(test_dataset)
    
    
