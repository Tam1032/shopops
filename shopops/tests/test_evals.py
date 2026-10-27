from shopops.baseline import KeywordBaseline
from shopops.evals import check_reply, gate, load_golden, run_case, summarize


def test_golden_set_is_consistent():
    cases = load_golden()
    assert len(cases) >= 30
    for c in cases:
        assert c["expected"]["action"] in ("reply", "escalate")
        bad = "post_reply" if c["expected"]["action"] == "escalate" else "escalate_to_human"
        assert bad in c["forbidden_tools"]


async def test_baseline_runs_end_to_end():
    cases = {c["review_id"]: c for c in load_golden()}
    results = [await run_case(KeywordBaseline(), cases[rid]) for rid in ("R003", "R006", "R021", "R032")]
    s = summarize(results)
    assert s["completion_rate"] == 1.0 and s["required_tool_recall"] == 1.0
    assert s["action_accuracy"] == 1.0 and s["promise_violations"] == 0


def test_reply_checks_and_gate():
    assert check_reply("Great bottle", "Thank you!")["language_ok"]
    assert not check_reply("Áo đẹp", "Thank you!")["language_ok"]
    assert not check_reply("Áo đẹp", "Shop hoàn tiền cho bạn")["no_promise"]
    summary = {"action_accuracy": 0.8, "promise_violations": 1, "judge_mean": None}
    fails = gate(summary, {"action_accuracy": 0.85, "promise_violations_max": 0, "judge_mean": 4})
    assert len(fails) == 2
