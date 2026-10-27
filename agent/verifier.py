# agent/verifier.py
"""
Checks every number and direction claim in AI-written text against the computed fact sheet.
Sentences that contain a figure we cannot find (or a sign that contradicts the data) are removed.
This is what catches errors like 'r = 0.25' when the real value is -0.25.
"""
import re

NUM_RE = re.compile(r"(?<![\w.])([-+]?)(\d[\d,]*(?:\.\d+)?)(%?)")
SIGN_CONTEXT = ("r =", "r=", "rho", "correl", "beta", "effect", "coefficient")
POS_WORDS = re.compile(r"\b(positive|positively)\b", re.I)
NEG_WORDS = re.compile(r"\b(negative|negatively|inverse|inversely)\b", re.I)


def _parse(text):
    out = []
    t = text.replace("\u2212", "-")          # real minus sign -> '-'
    for m in NUM_RE.finditer(t):
        sign, num, pct = m.groups()
        s = num.replace(",", "")
        try:
            v = float(s)
        except ValueError:
            continue
        out.append({"value": v, "neg": sign == "-", "dec": len(s.split(".")[1]) if "." in s else 0,
                    "pct": pct == "%", "start": m.start(), "raw": m.group(0)})
    return out


def _fact_numbers(text):
    return [{"value": n["value"], "neg": n["neg"]} for n in _parse(text)]


def _needs_check(n):
    if n["dec"] == 0 and not n["pct"]:
        if n["value"] <= 10:
            return False                       # small counts / list numbers
        if 1900 <= n["value"] <= 2100:
            return False                       # years
    return True


def _candidates(n, facts):
    tol = 0.5 * 10 ** (-n["dec"]) + 1e-9
    v = n["value"]
    out = []
    for f in facts:
        fv = abs(f["value"])
        if abs(v - fv) <= tol:
            out.append(f)
        elif (n["pct"] or v >= 1) and fv <= 1.5 and abs(v - fv * 100) <= max(tol, 0.5):
            out.append(f)                      # 0.54 quoted as 54% (only for percent-style numbers)
    return out


def _check_numbers(sentence, facts_nums):
    problems, checked, verified = [], 0, 0
    for n in _parse(sentence):
        if not _needs_check(n):
            continue
        checked += 1
        cands = _candidates(n, facts_nums)
        if not cands:
            problems.append({"type": "unverified figure", "token": n["raw"]})
            continue
        ctx = sentence[max(0, n["start"] - 25): n["start"]].lower().replace("\u2212", "-")
        if n["neg"] and not any(c["neg"] for c in cands):
            problems.append({"type": "sign mismatch (text negative, data positive)", "token": n["raw"]})
        elif (not n["neg"]) and all(c["neg"] for c in cands) and any(k in ctx for k in SIGN_CONTEXT):
            problems.append({"type": "sign dropped (data is negative)", "token": n["raw"]})
        else:
            verified += 1
    return problems, checked, verified


def _check_direction(sentence, pairs, columns):
    """'A is positively related to B' must agree with the sign of r(A, B)."""
    if not pairs or not columns:
        return []
    low = sentence.lower()
    present = [c for c in columns if re.search(r"(?<![\w])" + re.escape(c.lower()) + r"(?![\w])", low)]
    if len(present) != 2:
        return []
    r = pairs.get(f"{present[0]}||{present[1]}")
    if r is None:
        return []
    pos, neg = bool(POS_WORDS.search(sentence)), bool(NEG_WORDS.search(sentence))
    if pos and not neg and r < -0.1:
        return [{"type": f"direction wrong ({present[0]} & {present[1]} are negatively related)", "token": "positive"}]
    if neg and not pos and r > 0.1:
        return [{"type": f"direction wrong ({present[0]} & {present[1]} are positively related)", "token": "negative"}]
    return []


def verify_and_clean(text, fact_text, extra_allowed="", pairs=None, columns=None):
    """Return (clean_text, report). Unverifiable sentences are dropped."""
    facts_nums = _fact_numbers(fact_text + "\n" + (extra_allowed or ""))
    kept_lines, issues = [], []
    checked = verified = removed = 0
    for line in text.split("\n"):
        if not line.strip() or line.lstrip().startswith("#"):
            kept_lines.append(line)
            continue
        m = re.match(r"^(\s*(?:[-*\u2022]|\d+\.)\s+)?", line)
        prefix = m.group(0) if m else ""
        body = line[len(prefix):]
        kept = []
        for sent in re.split(r"(?<=[.!?])\s+", body):
            probs, c, v = _check_numbers(sent, facts_nums)
            probs += _check_direction(sent, pairs, columns)
            checked += c
            if probs:
                removed += 1
                issues += [dict(p, sentence=sent.strip()[:140]) for p in probs]
            else:
                verified += v
                kept.append(sent)
        if kept:
            kept_lines.append(prefix + " ".join(kept))
    return "\n".join(kept_lines), {"checked": checked, "verified": verified,
                                   "removed": removed, "issues": issues}
