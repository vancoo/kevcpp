import base64
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


if os.environ.get("JEVBENCH_DIR"):
    sys.path.insert(0, os.environ["JEVBENCH_DIR"])

from jevbench.metrics import latency_summary
from jevbench.scoring import InvalidDistribution, score_task, validate_probs
from jevbench.tasks import Task


ENDPOINT = os.environ.get("KEV_ENDPOINT", "http://127.0.0.1:18901/v1/systemone")
TIMEOUT = float(os.environ.get("KEV_TIMEOUT", "3600"))

# Prompt-variant switches used for accuracy experiments. Unset leaves the
# server default (candidates included, original criteria order).
INCLUDE_CANDIDATES = os.environ.get("KEV_INCLUDE_CANDIDATES")
ORDER_ENSEMBLE = os.environ.get("KEV_ORDER_ENSEMBLE") == "1"
REVERSE_CRITERIA = os.environ.get("KEV_REVERSE_CRITERIA") == "1"


def order_criteria(criteria, reverse):
    """Reverse candidate order while keeping the mapping labels -> text."""
    if not reverse or criteria is None:
        return criteria
    if isinstance(criteria, dict):
        return {k: criteria[k] for k in reversed(list(criteria))}
    if isinstance(criteria, list):
        return list(reversed(criteria))
    return criteria


def build_body(task, reverse=False):
    question = task.question
    criteria = order_criteria(question.get("criteria"), reverse)
    body_question = {
        "type": question["type"],
        "instructions": question["instructions"],
    }
    if criteria is not None:
        body_question["criteria"] = criteria
    body = {"state": task.state, "questions": {"decision": body_question}}
    if INCLUDE_CANDIDATES is not None:
        body["include_choice_candidates"] = INCLUDE_CANDIDATES == "1"
    return body


def merge_responses(first, second, task):
    """Average two answer probability maps keyed by label."""
    answers = first["answers"]["decision"]
    other = second["answers"]["decision"]
    if task.question["type"] == "noul":
        answers["noul"] = (answers["noul"] + other["noul"]) / 2.0
        return
    merged = {}
    for label, value in answers["probabilities"].items():
        merged[label] = (value + other["probabilities"][label]) / 2.0
    total = sum(merged.values())
    if total > 0:
        merged = {k: v / total for k, v in merged.items()}
    answers["probabilities"] = merged
    if task.question["type"] == "choice":
        answers["choice"] = max(merged, key=merged.get)
        labels = list(merged)
        count = len(labels)
        answers["confidence"] = (
            (count * merged[answers["choice"]] - 1) / (count - 1) if count > 1 else 1.0
        )
    else:
        answers["score"] = sum(int(k) * v for k, v in merged.items())
        count = len(merged)
        answers["confidence"] = (
            (count * max(merged.values()) - 1) / (count - 1) if count > 1 else 1.0
        )


def request_task(task):
    """POST one task, optionally averaging over candidate orders."""
    body = build_body(task)
    raw, latency, error, status = post(body)
    if not ORDER_ENSEMBLE or error or status != 200:
        return raw, latency, error, status
    raw2, latency2, error2, status2 = post(build_body(task, reverse=True))
    if error2 or status2 != 200:
        return raw, latency, error, status
    try:
        merged = parse_json(raw)
        merge_responses(merged, parse_json(raw2), task)
        return (
            json.dumps(merged, ensure_ascii=False).encode("utf-8"),
            latency + latency2,
            None,
            200,
        )
    except Exception as exc:
        return None, latency + latency2, f"{type(exc).__name__}: {exc}", None


def post(body):
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
            raw = response.read()
            return raw, time.perf_counter() - started, None, response.status
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read()
            error = f"HTTP {exc.code}"
        except Exception as read_error:
            raw = None
            error = f"HTTP {exc.code}; {type(read_error).__name__}: {read_error}"
        return raw, time.perf_counter() - started, error, exc.code
    except Exception as exc:
        return None, time.perf_counter() - started, f"{type(exc).__name__}: {exc}", None


def load_tasks(paths):
    records = []
    for path in paths:
        with open(path, encoding="utf-8") as source:
            records.extend(json.loads(line) for line in source if line.strip())
    digest = hashlib.sha256()
    for record in sorted(
        json.dumps(t, ensure_ascii=False, sort_keys=True) for t in records
    ):
        digest.update(record.encode("utf-8"))
        digest.update(b"\n")
    return [Task.from_dict(record) for record in records], digest.hexdigest()


def evaluate(task, answer):
    if not isinstance(answer, dict):
        raise ValueError("missing or invalid answers.decision")
    qtype = task.question["type"]
    if answer.get("type") != qtype:
        raise ValueError("native answer type mismatch")
    if qtype == "noul":
        p = answer.get("noul")
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise ValueError("noul must be numeric")
        if not 0 <= p <= 1:
            raise ValueError(f"noul out of range: {p}")
        probs = {"yes": p, "no": 1.0 - p}
    else:
        if qtype == "choice" and answer.get("choice") not in task.labels:
            raise ValueError("invalid native choice")
        probs = answer.get("probabilities")
        if not isinstance(probs, dict):
            raise ValueError(f"{qtype} answer missing probabilities")

    try:
        validate_probs(probs, task.labels)
        strict_error = None
    except InvalidDistribution as exc:
        strict_error = str(exc)
    scored = score_task(probs, task)
    scored["probs_as_returned"] = probs
    scored["strict_error"] = strict_error
    return scored


def parse_json(raw):
    def reject_constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"non-finite JSON number: {value}")
        return result

    return json.loads(
        raw.decode("utf-8"),
        parse_constant=reject_constant,
        parse_float=finite_float,
        object_pairs_hook=unique_object,
    )


def summarize(results, planned, dataset_hash):
    scorable = [row for row in results if row["scorable"]]
    correct = sum(row["correct"] is True for row in scorable)
    families = {}
    for family in sorted({row["family"] for row in results}):
        rows = [row for row in results if row["family"] == family]
        eligible = [row for row in rows if row["scorable"]]
        hits = sum(row["correct"] is True for row in eligible)
        families[family] = {
            "n": len(rows),
            "scorable": len(eligible),
            "correct": hits,
            "accuracy": hits / len(eligible) if eligible else None,
        }
    latency = latency_summary([row["latency_s"] for row in results])
    return {
        "endpoint": ENDPOINT,
        "dataset_hash": dataset_hash,
        "planned": planned,
        "total": len(results),
        "ok": sum(row["ok"] for row in results),
        "valid": sum(row["valid"] for row in results),
        "strict_valid": sum(row["strict_valid"] for row in results),
        "scorable": len(scorable),
        "correct": correct,
        "strict_correct": sum(row["strict_correct"] is True for row in scorable),
        "accuracy": correct / len(scorable) if scorable else None,
        "p50_s": latency["p50_s"],
        "p95_s": latency["p95_s"],
        "latency_n": latency["n"],
        "by_family": families,
        "details": results,
    }


def save_results(path, summary):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as output:
            temp_path = output.name
            json.dump(summary, output, ensure_ascii=True, indent=2, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def main():
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit("usage: python run_jevbench_kevcpp.py DATASET.jsonl [...]")
    tasks, digest = load_tasks(paths)
    print(f"tasks={len(tasks)} hash={digest}", flush=True)
    print(
        f"variants: include_candidates={INCLUDE_CANDIDATES} "
        f"order_ensemble={int(ORDER_ENSEMBLE)} reverse_criteria={int(REVERSE_CRITERIA)}",
        flush=True,
    )
    results_path = os.environ.get("KEV_RESULTS", "build/jevbench-kevcpp-results.json")
    results = []
    save_results(results_path, summarize(results, len(tasks), digest))
    for index, task in enumerate(tasks, 1):
        raw, latency, error, status = request_task(task)
        row = {
            "id": task.id,
            "family": task.family,
            "type": task.question["type"],
            "expected": task.expected,
            "scorable": task.expected is not None
            and not task.provenance.get("exclude_reason"),
            "latency_s": latency,
            "status": status,
            "ok": False,
            "valid": False,
            "strict_valid": False,
            "renormalized": False,
            "correct": False,
            "strict_correct": False,
            "predicted": None,
            "error": error,
            "strict_error": None,
            "response": None,
            "raw_response": raw.decode("utf-8", errors="replace")
            if raw is not None
            else None,
            "raw_response_base64": base64.b64encode(raw).decode("ascii")
            if raw is not None
            else None,
        }
        try:
            if raw is not None:
                row["response"] = parse_json(raw)
            if error:
                pass
            elif status != 200:
                row["error"] = f"HTTP {status}"
            elif not isinstance(row["response"], dict):
                raise ValueError("response is not an object")
            else:
                answers = row["response"].get("answers")
                if not isinstance(answers, dict):
                    raise ValueError("missing or invalid answers")
                scored = evaluate(task, answers.get("decision"))
                row.update(scored)
                row["ok"] = True
                row["strict_correct"] = (
                    scored["strict_valid"] and scored["correct"]
                    if scored["correct"] is not None
                    else None
                )
                row["error"] = scored.get("error")
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            row["error"] = f"{error}; {detail}" if error else detail
        results.append(row)
        summary = summarize(results, len(tasks), digest)
        save_results(results_path, summary)
        if index % 10 == 0 or index == len(tasks):
            print(
                f"{index}/{len(tasks)} ok={summary['ok']} strict_valid={summary['strict_valid']} "
                f"strict_correct={summary['strict_correct']} accuracy={summary['accuracy']} "
                f"p50_s={summary['p50_s']} p95_s={summary['p95_s']}",
                flush=True,
            )
    if not tasks:
        print(
            "0/0 ok=0 strict_valid=0 strict_correct=0 accuracy=None p50_s=None p95_s=None",
            flush=True,
        )
    print(f"wrote {results_path}", flush=True)


if __name__ == "__main__":
    main()
