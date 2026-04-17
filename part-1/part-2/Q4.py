import os
from collections import Counter
from transformers import T5TokenizerFast

MODEL_NAME = "google-t5/t5-small"


def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def preprocess_nl(text: str) -> str:
    return text.strip()


def preprocess_sql(text: str) -> str:
    return text.strip()


def compute_split_stats(tokenizer, nl_lines, sql_lines=None):
    nl_token_lists = [
        tokenizer.encode(x, add_special_tokens=True) for x in nl_lines
    ]
    nl_vocab = set()
    for ids in nl_token_lists:
        nl_vocab.update(ids)

    stats = {
        "num_examples": len(nl_lines),
        "mean_nl_len": sum(len(ids) for ids in nl_token_lists) / len(nl_token_lists),
        "nl_vocab_size": len(nl_vocab),
    }

    if sql_lines is not None:
        sql_token_lists = [
            tokenizer.encode(x, add_special_tokens=True) for x in sql_lines
        ]
        sql_vocab = set()
        for ids in sql_token_lists:
            sql_vocab.update(ids)

        stats["mean_sql_len"] = sum(len(ids) for ids in sql_token_lists) / len(sql_token_lists)
        stats["sql_vocab_size"] = len(sql_vocab)

    return stats


def print_table(before_train, before_dev, after_train, after_dev):
    print("\n=== TABLE 1: BEFORE PREPROCESSING ===")
    print(f"{'Statistic':35s} {'Train':>12s} {'Dev':>12s}")
    print("-" * 62)
    print(f"{'Number of examples':35s} {before_train['num_examples']:12d} {before_dev['num_examples']:12d}")
    print(f"{'Mean sentence length':35s} {before_train['mean_nl_len']:12.2f} {before_dev['mean_nl_len']:12.2f}")
    print(f"{'Mean SQL query length':35s} {before_train['mean_sql_len']:12.2f} {before_dev['mean_sql_len']:12.2f}")
    print(f"{'Vocabulary size (natural language)':35s} {before_train['nl_vocab_size']:12d} {before_dev['nl_vocab_size']:12d}")
    print(f"{'Vocabulary size (SQL)':35s} {before_train['sql_vocab_size']:12d} {before_dev['sql_vocab_size']:12d}")

    print("\n=== TABLE 2: AFTER PREPROCESSING ===")
    print(f"Model name: {MODEL_NAME}")
    print(f"{'Statistic':35s} {'Train':>12s} {'Dev':>12s}")
    print("-" * 62)
    print(f"{'Mean sentence length':35s} {after_train['mean_nl_len']:12.2f} {after_dev['mean_nl_len']:12.2f}")
    print(f"{'Mean SQL query length':35s} {after_train['mean_sql_len']:12.2f} {after_dev['mean_sql_len']:12.2f}")
    print(f"{'Vocabulary size (natural language)':35s} {after_train['nl_vocab_size']:12d} {after_dev['nl_vocab_size']:12d}")
    print(f"{'Vocabulary size (SQL)':35s} {after_train['sql_vocab_size']:12d} {after_dev['sql_vocab_size']:12d}")


def main():
    data_dir = "data"
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)

    train_nl_raw = load_lines(os.path.join(data_dir, "train.nl"))
    train_sql_raw = load_lines(os.path.join(data_dir, "train.sql"))
    dev_nl_raw = load_lines(os.path.join(data_dir, "dev.nl"))
    dev_sql_raw = load_lines(os.path.join(data_dir, "dev.sql"))

    before_train = compute_split_stats(tokenizer, train_nl_raw, train_sql_raw)
    before_dev = compute_split_stats(tokenizer, dev_nl_raw, dev_sql_raw)

    train_nl_proc = [preprocess_nl(x) for x in train_nl_raw]
    train_sql_proc = [preprocess_sql(x) for x in train_sql_raw]
    dev_nl_proc = [preprocess_nl(x) for x in dev_nl_raw]
    dev_sql_proc = [preprocess_sql(x) for x in dev_sql_raw]

    after_train = compute_split_stats(tokenizer, train_nl_proc, train_sql_proc)
    after_dev = compute_split_stats(tokenizer, dev_nl_proc, dev_sql_proc)

    print_table(before_train, before_dev, after_train, after_dev)


if __name__ == "__main__":
    main()