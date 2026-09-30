#!/usr/bin/env python
"""Conformance: re-run kev/tests/test_api.py's non-SDK assertions against the C++ server using httpx.

proxy env lacks pytest/typesafe_sdk, so we execute the same assertions directly.
Run with the server up:  build\\bin\\Release\\main_server.exe --model <gguf> --head head.bin --port 8008
"""
import math, os, sys
import httpx

BASE = os.environ.get("KEV_BASE_URL", "http://127.0.0.1:8008")
cl = httpx.Client(timeout=180)
passing = 0
failing = []

def post(body, expect_id_header=True):
    r = cl.post(f"{BASE}/v1/systemone", json=body)
    if expect_id_header and not r.headers.get("x-typesafe-request-id"):
        raise AssertionError("missing x-typesafe-request-id header")
    return r.status_code, r.json()

def check(name, fn):
    global passing
    try:
        fn(); passing += 1; print(f"  ok   {name}")
    except Exception as e:
        failing.append(name); print(f"  FAIL {name}: {e}")

DEPARTMENT = {"returns": "Exchanges, refunds, wrong or damaged items",
              "shipping": "Delivery status, delays, lost packages",
              "billing": "Charges, invoices, payment problems"}

def test_choice_basic():
    code, r = post({"state": "My running shoes arrived in the wrong size. Can I swap them for a size 10?",
                    "model": "jev-latest", "questions": {"department": {"type": "choice", "instructions": "Which team should handle this?", "criteria": DEPARTMENT}}})
    assert code == 200 and r["model"] == "jev-latest"
    a = r["answers"]["department"]
    assert a["type"] == "choice" and a["choice"] in DEPARTMENT and set(a["probabilities"]) == set(DEPARTMENT)
    assert math.isclose(sum(a["probabilities"].values()), 1.0, abs_tol=0.03) and 0 <= a["confidence"] <= 1
    assert a["choice"] == max(a["probabilities"], key=a["probabilities"].get)
    assert set(r["usage"]) == {"input_tokens", "output_tokens"}

def test_five_questions_and_null_descriptions():
    code, r = post({"state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card. What are you going to do about this?",
                    "questions": {
                        "department": {"type": "choice", "instructions": "Which team should handle this?", "criteria": DEPARTMENT},
                        "return_reason": {"type": "choice", "instructions": "If the customer wants to return something, why?",
                                          "criteria": {"wrong_size": "The item doesn't fit", "wrong_item": "A different product was delivered",
                                                       "damaged": "The item arrived broken or faulty", "changed_mind": "The item is fine, the customer no longer wants it",
                                                       "other": "A return reason that fits none of the above"}},
                        "tone": {"type": "choice", "instructions": "What is the customer's tone?", "criteria": {"calm": None, "frustrated": None, "angry": None}}}})
    assert code == 200 and set(r["answers"]) == {"department", "return_reason", "tone"}
    assert r["answers"]["tone"]["choice"] in {"calm", "frustrated", "angry"}

def test_structured_instructions_and_criteria():
    code, r = post({"state": "I sent the shoes back a week ago. When do I get my money?",
                    "questions": {"return_topic": {"type": "choice",
                                                   "instructions": {"question": "Which returns topic is the customer asking about?", "focus": "Classify the information the customer wants."},
                                                   "criteria": {"return_policy": {"what": "Whether and how an item can be returned", "not_for": "Progress of a return already sent",
                                                                                 "examples": ["Can I return shoes I've worn once?", "How long do I have to return an order?"]},
                                                                "return_status": {"what": "Progress of a return already sent", "not_for": "Whether and how an item can be returned",
                                                                                  "examples": ["Has my return arrived yet?", "When will my refund be paid?"]}}}}})
    assert code == 200 and r["answers"]["return_topic"]["choice"] in {"return_policy", "return_status"}

def test_noul_score_and_object_state():
    code, r = post({"state": {"document": "I was charged twice. Please fix this ASAP."},
                    "questions": {"billing": {"type": "noul", "instructions": "Is this ticket about billing?", "criteria": {"true": "Explicitly about charges", "false": "Not about charges"}},
                                  "urgency": {"type": "score", "instructions": "How urgent is this ticket?", "criteria": ["can wait", "this week", "today"]}}})
    assert code == 200
    n, s = r["answers"]["billing"], r["answers"]["urgency"]
    assert n == {"type": "noul", "noul": n["noul"]} and 0 <= n["noul"] <= 1
    assert s["type"] == "score" and 0 <= s["score"] <= 2 and s["legend"] == {"0": "can wait", "1": "this week", "2": "today"}
    assert set(s["probabilities"]) == {"0", "1", "2"} and 0 <= s["confidence"] <= 1
    assert math.isclose(s["score"], sum(int(k) * v for k, v in s["probabilities"].items()), abs_tol=0.05)

def test_optional_instructions_and_one_level_score():
    code, r = post({"state": "I was charged twice.",
                    "questions": {"billing": {"type": "noul", "criteria": {"true": "About charges", "false": "Not about charges"}},
                                  "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["today"]}}})
    assert code == 200 and 0 <= r["answers"]["billing"]["noul"] <= 1
    a = r["answers"]["urgency"]
    assert a["type"] == "score" and a["score"] == 0.0 and a["legend"] == {"0": "today"}
    assert a["probabilities"] == {"0": 1.0} and a["confidence"] == 1.0

def test_validation_422():
    assert post({"state": "x", "model": "m", "questions": {"q": {"type": "score", "instructions": "i", "criteria": []}}})[0] == 422
    assert post({"state": "x", "model": "m", "questions": {"q": {"type": "bogus", "instructions": "i"}}})[0] == 422
    assert post({"state": "x", "model": "m", "questions": {}})[0] == 422
    assert post({"state": "x", "model": "m", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {f"o{i}": None for i in range(256)}}}})[0] == 422

def test_packed_equals_separate():
    qs = {"a": {"type": "noul", "instructions": "Is the weather described as nice?"},
          "b": {"type": "choice", "instructions": "Which season is it most likely?", "criteria": {"summer": None, "winter": None, "unknown": None}}}
    state = "The weather is nice today and the park is full of people."
    both = post({"state": state, "model": "m", "questions": qs})[1]["answers"]
    alone = post({"state": state, "model": "m", "questions": {"b": qs["b"]}})[1]["answers"]
    for k in both["b"]["probabilities"]:
        assert abs(both["b"]["probabilities"][k] - alone["b"]["probabilities"][k]) <= 0.011

def test_models():
    r = cl.get(f"{BASE}/v1/models")
    assert r.status_code == 200
    cards = {m["name"] for m in r.json()["models"]}
    assert {"kev-latest", "jev-latest"} <= cards
    for m in r.json()["models"]:
        assert m.get("description") and m.get("release_date")

print("== kev test_api.py conformance (non-SDK subset, via httpx) ==")
check("choice_basic", test_choice_basic)
check("five_questions_and_null_descriptions", test_five_questions_and_null_descriptions)
check("structured_instructions_and_criteria", test_structured_instructions_and_criteria)
check("noul_score_and_object_state", test_noul_score_and_object_state)
check("optional_instructions_and_one_level_score", test_optional_instructions_and_one_level_score)
check("validation_422", test_validation_422)
check("packed_equals_separate", test_packed_equals_separate)
check("models", test_models)
print(f"RESULT: {passing} passed, {len(failing)} failed" + ((" -> " + ", ".join(failing)) if failing else " -> ALL PASS"))
sys.exit(0 if not failing else 1)