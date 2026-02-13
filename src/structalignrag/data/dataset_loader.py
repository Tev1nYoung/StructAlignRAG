import json
from typing import Any, Dict, List, Optional, Set, Tuple


def load_corpus(corpus_path: str) -> List[Dict[str, Any]]:
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    if not isinstance(corpus, list):
        raise ValueError(f"Invalid corpus file: expected list, got {type(corpus)}")
    return corpus


def load_samples(samples_path: str) -> List[Dict[str, Any]]:
    with open(samples_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        raise ValueError(f"Invalid dataset file: expected list, got {type(samples)}")
    return samples


def corpus_to_docs(corpus: List[Dict[str, Any]]) -> List[str]:
    # Same formatting as HippoRAG for recall comparison.
    return [f"{doc['title']}\n{doc['text']}" for doc in corpus]


def get_gold_docs(samples: List[Dict[str, Any]], dataset_name: Optional[str] = None) -> List[List[str]]:
    """
    Copied (with minor normalization) from HippoRAG/main.py to keep Recall@k comparable.
    """
    gold_docs: List[List[str]] = []
    for sample in samples:
        if "supporting_facts" in sample:  # hotpotqa, 2wikimultihopqa
            gold_title = set([item[0] for item in sample["supporting_facts"]])
            gold_title_and_content_list = [item for item in sample["context"] if item[0] in gold_title]
            if dataset_name and "hotpotqa" in dataset_name:
                gold_doc = [item[0] + "\n" + "".join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_doc = [item[0] + "\n" + " ".join(item[1]) for item in gold_title_and_content_list]
        elif "contexts" in sample:
            gold_doc = [item["title"] + "\n" + item["text"] for item in sample["contexts"] if item.get("is_supporting")]
        else:
            assert "paragraphs" in sample, "`paragraphs` should be in sample, or consider the setting not to evaluate retrieval"
            gold_paragraphs = []
            for item in sample["paragraphs"]:
                if "is_supporting" in item and item["is_supporting"] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [item["title"] + "\n" + (item["text"] if "text" in item else item["paragraph_text"]) for item in gold_paragraphs]

        gold_docs.append(list(set(gold_doc)))
    return gold_docs


def get_gold_answers(samples: List[Dict[str, Any]]) -> List[Set[str]]:
    """
    Copied (with minor normalization) from HippoRAG/main.py to keep EM/F1 comparable.
    """
    gold_answers: List[Set[str]] = []
    for sample in samples:
        gold_ans = None
        if "answer" in sample or "gold_ans" in sample:
            gold_ans = sample["answer"] if "answer" in sample else sample["gold_ans"]
        elif "reference" in sample:
            gold_ans = sample["reference"]
        elif "obj" in sample:
            gold_ans = set([sample["obj"]] + [sample["possible_answers"]] + [sample["o_wiki_title"]] + [sample["o_aliases"]])
            gold_ans = list(gold_ans)
        if gold_ans is None:
            raise ValueError("Cannot find gold answer in sample.")

        if isinstance(gold_ans, str):
            gold_list = [gold_ans]
        else:
            gold_list = list(gold_ans)

        gold_set = set(gold_list)
        if "answer_aliases" in sample and isinstance(sample["answer_aliases"], list):
            gold_set.update(sample["answer_aliases"])

        gold_answers.append(gold_set)

    return gold_answers


def cap_samples(
    samples: List[Dict[str, Any]],
    max_queries: Optional[int] = None,
    query_offset: int = 0,
    shuffle_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if query_offset < 0:
        raise ValueError("query_offset must be >= 0")
    if max_queries is not None and max_queries < 0:
        raise ValueError("max_queries must be >= 0")

    if shuffle_seed is not None:
        import random
        rng = random.Random(shuffle_seed)
        rng.shuffle(samples)

    start = min(query_offset, len(samples))
    end = len(samples) if max_queries is None else min(start + max_queries, len(samples))
    return samples[start:end]

