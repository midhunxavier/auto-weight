"""Human-readable views of the saved evaluation records."""
from html import escape

from .checkpoint import hf_text
from .contracts import TEXT_METRICS
from .runner import verify


def rate_metric(metric, adapter):
    return ((hf_text(adapter) and metric in TEXT_METRICS)
            or (adapter == "tiny-transformer-v1" and metric == "accuracy"))


def score(value, rate):
    return f"{value * 100:.6g}%" if rate else f"{value:.6g}"


def change(value, rate):
    return f"{value * 100:+.6g} percentage points" if rate else f"{value:+.6g} score units"


def acceptance_rules(gates, adapter):
    rules = []
    for gate in gates:
        rate = rate_metric(gate["metric"], adapter)
        parts = []
        if "min_delta" in gate:
            delta = gate["min_delta"]
            if gate["kind"] == "improve":
                parts.append(f"improve by at least {change(delta, rate).lstrip('+')}")
            elif delta == 0:
                parts.append("allow no decline")
            else:
                parts.append(f"allow a decline of at most {change(-delta, rate).lstrip('+')}")
        if "min_candidate" in gate:
            parts.append(f"reach an observed score of at least {score(gate['min_candidate'], rate)}")
        rules.append(f"{gate['group']} ({gate['metric']}): {' and '.join(parts)}.")
    if any("min_delta" in gate for gate in gates):
        rules.append("Change requirements use the paired lower bound in the direction of improvement.")
    return rules


def setup_feedback(baseline):
    return [f"{group}: The current model already meets a repair score floor on development examples. "
            "Check that these examples capture the reported failure and that the target is appropriate."
            for group in dict.fromkeys(baseline["baseline_meets_target"])]


def trial_lines(report):
    if report["status"] == "error":
        return [report["error"] or "Evaluation failed."]
    evaluation = report["evaluation"]
    adapter = report["model"]["adapter"]
    lines = []
    for gate in evaluation["gates"]:
        group, metric = gate["group"], gate["metric"]
        data = evaluation["groups"][group]
        rate = rate_metric(metric, adapter)
        after = data["candidate"][metric]
        scores = score(after, rate)
        if "baseline" in data:
            scores = f"{score(data['baseline'][metric], rate)} → {scores}"
        status = "Pass" if gate["passed"] else "Fail"
        reasons = []
        if "lower_bound" in gate and gate["lower_bound"] < gate["min_delta"]:
            reasons.append(f"lower bound {change(gate['lower_bound'], rate)} is below "
                           f"the required {change(gate['min_delta'], rate)}")
        if "min_candidate" in gate and after < gate["min_candidate"]:
            reasons.append(f"observed score is below the {score(gate['min_candidate'], rate)} minimum")
        detail = f" — {' and '.join(reasons)}" if reasons else ""
        lines.append(f"{status} · {group} ({metric}, {data['examples']} examples): {scores}{detail}")
    return lines


def render_report(workspace):
    run = verify(workspace, include_details=True)
    issue = run["issue"]
    confirmation = run["confirmation_report"]
    status = {
        None: "Awaiting confirmation" if "eligible" in run["attempts"].values() else "No confirmed candidate",
        "accepted": "Passed the configured confirmation checks",
        "discarded": "Did not pass the configured confirmation checks",
        "error": "Confirmation evaluation failed",
    }[run["confirmation"]]
    lines = ["AutoWeight repair report", issue["issue"], "", status]
    if run["checkpoint"]:
        lines.append(f"Exported checkpoint: {run['checkpoint']}")
    else:
        lines.append("No checkpoint exported.")
    if "backbone" in run:
        lines.append(f"Pinned base model: {run['backbone']}")
    if confirmation:
        lines.extend(["", f"Confirmation · {confirmation['attempt']} · {confirmation['seconds']:.1f}s",
                      *trial_lines(confirmation)])
    lines.extend(["", "Acceptance rules", *acceptance_rules(issue["gates"], issue["model"]["adapter"])])
    if run["baseline"]:
        baseline = run["baseline"]
        lines.extend(["", "Development baseline"])
        for gate in issue["gates"]:
            group, metric = gate["group"], gate["metric"]
            data = baseline["groups"][group]
            value = score(data["baseline"][metric], rate_metric(metric, issue["model"]["adapter"]))
            lines.append(f"{group} ({metric}, {data['examples']} examples): {value}")
        lines.extend(setup_feedback(baseline))
    reports = run["attempt_reports"]
    lines.extend(["", f"Development attempts · {run['attempts_used']}/{run['attempt_limit']}"])
    attempt_status = {"eligible": "Ready for confirmation", "discarded": "Failed development checks",
                      "error": "Evaluation failed"}
    for attempt_id, report in reports.items():
        lines.extend([f"{attempt_id} · {attempt_status[report['status']]} · {report['seconds']:.1f}s",
                      *trial_lines(report)])
    selected = reports.get(confirmation["attempt"]) if confirmation else next(reversed(reports.values()), None)
    if selected and selected["evaluation"]:
        samples = selected["evaluation"].get("development_samples", {})
        if samples:
            lines.extend(["", f"Development examples · attempt {selected['attempt']}"])
        for group, rows in samples.items():
            for row in rows:
                lines.extend([f"{group} · Prompt: {row['prompt']!r}",
                              f"  Before: {row['baseline']!r}", f"  After:  {row['candidate']!r}"])
    lines.extend(["", "Results apply to the examples and metrics evaluated in this run."])
    return "\n".join(lines)


def write_html(path, report):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(
            '<!doctype html>\n<html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>AutoWeight repair report</title>'
            '<style>body{max-width:80ch;margin:2rem auto;padding:0 1rem}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.5}</style>'
            f'<pre>{escape(report)}</pre></html>\n'
        )
