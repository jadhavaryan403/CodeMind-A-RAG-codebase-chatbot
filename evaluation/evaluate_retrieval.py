import json
import os
import time
from eval_utils import run_pipeline_once


BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_DIR, "eval_dataset.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "results_pipeline.json")

USER_ID = 3
PROJECT_ID = 22


def evaluate_retrieval(expected, retrieved):
    hit = any(chunk in retrieved for chunk in expected) # hit if any expected chunk is retrieved

    correct = sum(1 for chunk in expected if chunk in retrieved)  # recall numerator

    rank = 0 # rank of the first relevant chunk in retrieved list
    for i, r in enumerate(retrieved):
        if r in expected:
            rank = i + 1
            break

    recall = correct / len(expected) if expected else 0

    rr = round(1 / rank if rank > 0 else 0 ,4)
    return recall ,hit ,rr


def main():
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    results = []

    for i in range(0 ,len(dataset)):
        item = dataset[i]
        query = item["query"]
        expected = item["relevant_chunks"]

        print(f"Running query {i+1}/{len(dataset)}")

        try:
            result = run_pipeline_once(query, USER_ID, PROJECT_ID)

            time.sleep(5)  # to avoid overwhelming the system during testing

            retrieved = result["retrieved_chunks"]
            answer = result["answer"]
            usage = result["usage"]

            recall, hit, rr = evaluate_retrieval(expected, retrieved)

            results.append({
                "query": query,
                "expected_chunks": expected,
                "retrieved_chunks": retrieved,
                "answer": answer,
                "hit": hit,
                "recall": recall,
                "reciprocal_rank": rr,
                "usage": usage
            })

        except Exception as e:
            print("Error:", e)

    # summary
    total = len(results)
    recall = sum(r["recall"] for r in results) / total
    mrr = sum(r["reciprocal_rank"] for r in results) / total

    output = {
        "summary": {
            "total_queries": total,
            "recall": round(recall, 4),
            "mrr": round(mrr, 4)
        },
        "results": results
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print("\nEvaluation complete!")
    print(output["summary"])


if __name__ == "__main__":
    main()