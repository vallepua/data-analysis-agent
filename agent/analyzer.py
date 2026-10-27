# agent/analyzer.py
"""
The AI writes the words (summary, recommendations, limits).
It may only use numbers from the verified fact sheet; everything it writes is then checked again.
"""
import re

from agent import llm
from agent.stats_engine import facts_to_text
from agent.verifier import verify_and_clean

SYSTEM = (
    "You are a careful senior data analyst writing for business decision makers. "
    "You NEVER invent or calculate numbers. You only use numbers that appear in the FACT SHEET, "
    "copied exactly, including minus signs. You describe associations, never proven causes."
)


def _prompt(fact_text, goal, decision_hint):
    return f"""FACT SHEET (all numbers were computed by software and are correct):
{fact_text}

TASK: Write exactly three sections in Markdown, using these exact headings:

## Executive Summary
3 to 4 plain-English sentences. Start with what the data says about the user's goal, mention how far the data
can be trusted (data quality verdict), and name the single most important finding (cite its ID, e.g. F1).

## Recommendations
3 to 5 bullets. Each bullet: **the action** - why (cite finding IDs like F1) - what direction of change to expect -
**Confidence: High/Medium/Low** (copy the confidence of the cited finding). {decision_hint}
Never recommend something that contradicts the sign of a finding. Use words like "is associated with", "tends to".

## What This Data Cannot Tell You
2 to 3 bullets on limits: missing variables, sample size, causation, anything in the CAVEATS line.

RULES: do NOT guess what a column means beyond its name (for example do not call a column "acquisition cost" or
"price per sq ft" unless the user said so); refer to columns by their exact names. Frame actions as where to focus,
test or investigate, never as guaranteed outcomes of changing a variable. Copy numbers exactly from the fact sheet; do not compute new numbers (no averages, ratios, percentages of your
own); if you need a number that is not in the fact sheet, describe it in words instead. Keep it under 330 words."""


def _repair_prompt(draft, issues):
    bad = "; ".join(f"'{i['token']}' ({i['type']})" for i in issues[:8])
    return ("Your draft contained claims that do not match the fact sheet: " + bad + ".\n"
            "Rewrite the full three-section answer. Use only numbers exactly as written in the fact sheet, keep minus "
            "signs, and describe things in words when unsure.\n\nDRAFT:\n" + draft)


def split_sections(text):
    parts = {"summary": "", "recommendations": "", "limits": ""}
    current = None
    for line in text.split("\n"):
        m = re.match(r"^#{1,3}\s+(.*)$", line.strip())
        if m:
            h = m.group(1).lower()
            if "executive" in h or "summary" in h:
                current = "summary"
            elif "recommend" in h:
                current = "recommendations"
            elif "cannot" in h or "limit" in h or "can't" in h:
                current = "limits"
            else:
                current = None
            continue
        if current:
            parts[current] += line + "\n"
    return {k: v.strip() for k, v in parts.items()}


def fallback_sections(facts):
    """Used when the AI is unavailable: purely rule-based, still correct."""
    t = facts.get("target")
    q = facts["profile"]
    if t and facts["findings"]:
        f = facts["findings"][0]
        summary = (f"Data quality is rated '{q['verdict']}' (score {q['score']}/100). "
                   f"The strongest signal for {t['name']} is: {f['text']} ({f['confidence']} confidence).")
        recs = "\n".join(
            f"- Review **{x['variable']}** as a lever for {t['name']} - see {x['id']} "
            f"- **Confidence: {x['confidence']}**" for x in facts["findings"][:3])
    else:
        summary = (f"Data quality is rated '{q['verdict']}' (score {q['score']}/100). "
                   "Choose an outcome column to get driver analysis.")
        recs = "- Pick the column that represents the result you care about and run the analysis again."
    limits = "\n".join(f"- {c}" for c in facts["caveats"][:3])
    return {"summary": summary, "recommendations": recs, "limits": limits}


def generate_narrative(facts, goal="", decision_hint=""):
    """Return (sections dict, verification dict, fact_text)."""
    fact_text = facts_to_text(facts, goal)
    columns = [c for c in facts["kinds"].keys()]
    pairs = facts.get("_pairs")
    status = {"checked": 0, "verified": 0, "removed": 0, "issues": [], "ai_used": False,
              "repaired": False, "error": ""}
    try:
        raw = llm.call_llm(_prompt(fact_text, goal, decision_hint), max_tokens=1800, system=SYSTEM)
    except Exception as e:  # noqa
        status["error"] = str(e)
        return fallback_sections(facts), status, fact_text

    clean, rep = verify_and_clean(raw, fact_text, goal, pairs, columns)
    if rep["removed"] > 0:
        try:
            raw2 = llm.call_llm(_repair_prompt(raw, rep["issues"]), max_tokens=1800, system=SYSTEM)
            clean2, rep2 = verify_and_clean(raw2, fact_text, goal, pairs, columns)
            if rep2["removed"] <= rep["removed"]:
                clean, rep = clean2, rep2
                status["repaired"] = True
        except Exception:
            pass
    status.update(rep)
    status["ai_used"] = True

    sections = split_sections(clean)
    fb = fallback_sections(facts)
    for k in sections:
        if len(sections[k]) < 20:          # section was empty or removed by verification
            sections[k] = fb[k]
    return sections, status, fact_text
