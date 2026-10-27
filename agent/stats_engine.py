# agent/stats_engine.py
"""
All numbers in the report are computed HERE, in Python.
The AI only writes words around this verified fact sheet.
Works for any dataset (any columns, any size).
"""
import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

ID_NAME = re.compile(r"(^|[_\s\-])(id|uuid|guid|index|idx|key|sku|code|no|num)([_\s\-]|$)", re.I)
TARGET_HINTS = ["target", "label", "outcome", "revenue", "sales", "profit", "income", "salary", "rent",
                "amount", "total", "score", "churn", "rating", "price", "value", "cost"]
MAX_NUMERIC_COLS = 25


# --------------------------------------------------------------------------- formatting
def g(v):
    """Compact number format used everywhere (so the AI text can be verified)."""
    if v is None:
        return "n/a"
    try:
        v = float(v)
    except Exception:
        return str(v)
    if np.isnan(v):
        return "n/a"
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    if a >= 0.01:
        return f"{v:.3f}"
    return np.format_float_positional(v, precision=2, fractional=False, trim="-")


def pc(v):
    """Percent with one decimal: 20.6%"""
    return f"{float(v):.1f}%"


def sg(v):
    """Signed 2-decimal format for correlations: +0.54 / -0.25."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{float(v):+.2f}"


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "p=n/a"
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def strength_label(r):
    a = abs(r)
    if a >= 0.7:
        return "very strong"
    if a >= 0.5:
        return "strong"
    if a >= 0.3:
        return "moderate"
    if a >= 0.1:
        return "weak"
    return "negligible"


# --------------------------------------------------------------------------- column types
def _is_integer_like(s):
    x = s.dropna()
    if x.empty:
        return False
    try:
        return bool(np.all(np.mod(x.astype(float).values, 1) == 0))
    except Exception:
        return False


def column_kinds(df):
    """numeric | categorical | datetime | id | high_cardinality | constant"""
    kinds = {}
    for c in df.columns:
        s = df[c]
        nn = int(s.notna().sum())
        nun = int(s.nunique(dropna=True))
        if nn == 0 or nun <= 1:
            kinds[c] = "constant"
        elif pd.api.types.is_bool_dtype(s):
            kinds[c] = "categorical"
        elif pd.api.types.is_datetime64_any_dtype(s):
            kinds[c] = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            ratio = nun / nn
            x = s.dropna()
            if (_is_integer_like(s) and ratio >= 0.98 and nn >= 20
                    and (ID_NAME.search(str(c)) or x.is_monotonic_increasing
                         or x.is_monotonic_decreasing)):
                kinds[c] = "id"
            else:
                kinds[c] = "numeric"
        else:
            ratio = nun / nn
            if ratio >= 0.9 and nn >= 20:
                kinds[c] = "id"
            elif nun > 20:
                kinds[c] = "high_cardinality"
            else:
                kinds[c] = "categorical"
    return kinds


def _numeric_cols(df, kinds):
    cols = [c for c in df.columns if kinds[c] == "numeric"]
    if len(cols) > MAX_NUMERIC_COLS:
        cols = sorted(cols, key=lambda c: df[c].isna().sum())[:MAX_NUMERIC_COLS]
    return cols


def _segment_cols(df, kinds, exclude=None):
    n = len(df)
    out = []
    for c in df.columns:
        if c == exclude:
            continue
        nun = int(df[c].nunique(dropna=True))
        if kinds[c] == "categorical" and 2 <= nun <= max(2, min(12, n // 3)):
            out.append(c)
        elif kinds[c] == "numeric" and 2 <= nun <= 6 and _is_integer_like(df[c]):
            out.append(c)
    return out


def suggest_target(df, kinds=None):
    kinds = kinds or column_kinds(df)
    nums = [c for c in df.columns if kinds[c] == "numeric"]
    if not nums:
        return None
    low = {c: str(c).lower() for c in nums}
    for hint in TARGET_HINTS:
        for c in nums:
            if hint in low[c]:
                return c
    for c in nums:
        if low[c] == "y":
            return c
    if len(nums) == 1:
        return nums[0]
    corr = df[nums].corr().abs()
    np.fill_diagonal(corr.values, np.nan)
    return corr.mean().idxmax()


# --------------------------------------------------------------------------- correlation helpers
def _pearson(x, y):
    m = x.notna() & y.notna()
    xv, yv = x[m].astype(float).values, y[m].astype(float).values
    n = len(xv)
    if n < 4 or np.std(xv) == 0 or np.std(yv) == 0:
        return np.nan, np.nan, n
    r, p = stats.pearsonr(xv, yv)
    return float(r), float(p), n


def _spearman(x, y):
    m = x.notna() & y.notna()
    xv, yv = x[m].astype(float).values, y[m].astype(float).values
    if len(xv) < 4 or np.std(xv) == 0 or np.std(yv) == 0:
        return np.nan
    rho, _ = stats.spearmanr(xv, yv)
    return float(rho)


# --------------------------------------------------------------------------- data quality
def profile_data(df, kinds):
    n = len(df)
    cols = []
    issues = []

    def add(sev, text):
        issues.append({"severity": sev, "text": text})

    for c in df.columns:
        s = df[c]
        miss = int(s.isna().sum())
        info = {"name": c, "kind": kinds[c], "missing": miss,
                "missing_pct": round(100 * miss / n, 1) if n else 0.0,
                "unique": int(s.nunique(dropna=True)),
                "outliers": 0, "outlier_pct": 0.0}
        if kinds[c] == "numeric":
            x = s.dropna().astype(float)
            if len(x) >= 4:
                q1, q3 = x.quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr > 0:
                    out = int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum())
                    info["outliers"] = out
                    info["outlier_pct"] = round(100 * out / len(x), 1)
        cols.append(info)

    # sample size
    if n < 30:
        add("high", f"Very small sample ({n} rows): results are unstable and should be treated as indicative only.")
    elif n < 100:
        add("medium", f"Small sample ({n} rows): relationships can look stronger or weaker than they really are.")

    # missing values
    for info in cols:
        if info["missing"] == 0:
            continue
        pct = info["missing_pct"]
        word = "value" if info["missing"] == 1 else "values"
        text = f"{info['missing']} missing {word} in '{info['name']}' ({pct}% of rows)."
        add("high" if pct > 30 else "medium" if pct > 5 else "low", text)

    # duplicates
    dups = int(df.duplicated().sum())
    if dups:
        add("medium", f"{dups} fully duplicated rows (may double-count records).")

    # outliers
    heavy = [f"'{i['name']}' ({i['outliers']} rows, {i['outlier_pct']}%)"
             for i in cols if i["outlier_pct"] >= 5]
    if heavy:
        add("medium", "Many extreme values (beyond 1.5 x IQR) in: " + ", ".join(heavy) + ".")

    const = [c for c in df.columns if kinds[c] == "constant"]
    if const:
        add("low", "Columns with no variation (ignored): " + ", ".join(f"'{c}'" for c in const) + ".")
    ids = [c for c in df.columns if kinds[c] == "id"]
    if ids:
        add("low", "Identifier-like columns (ignored in analysis): " + ", ".join(f"'{c}'" for c in ids) + ".")
    hc = [c for c in df.columns if kinds[c] == "high_cardinality"]
    if hc:
        add("low", "Text columns with too many distinct values to group (ignored): "
            + ", ".join(f"'{c}'" for c in hc) + ".")
    add("info", "Units and definitions are not stored in the file (for example $/sq ft vs $/year). "
                "Please confirm them before acting on the numbers.")

    score = 100 - sum({"high": 25, "medium": 10, "low": 3}.get(i["severity"], 0) for i in issues)
    score = max(0, score)
    verdict = "Good" if score >= 85 else "Usable with caveats" if score >= 60 else "Weak - fix data issues first"
    return {"columns": cols, "duplicates": dups, "issues": issues, "score": score, "verdict": verdict}


def numeric_summary(df, cols):
    out = {}
    for c in cols:
        x = df[c].dropna().astype(float)
        if len(x) < 2:
            continue
        out[c] = {"mean": float(x.mean()), "median": float(x.median()), "std": float(x.std()),
                  "min": float(x.min()), "max": float(x.max()),
                  "q1": float(x.quantile(0.25)), "q3": float(x.quantile(0.75)),
                  "skew": float(x.skew()) if len(x) > 2 else 0.0}
    return out


def pair_table(df, num_cols):
    pairs, lookup = [], {}
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            a, b = num_cols[i], num_cols[j]
            r, p, n = _pearson(df[a], df[b])
            if np.isnan(r):
                continue
            pairs.append({"a": a, "b": b, "r": r, "p": p, "n": n})
            lookup[f"{a}||{b}"] = r
            lookup[f"{b}||{a}"] = r
    pairs.sort(key=lambda d: -abs(d["r"]))
    return pairs, lookup


# --------------------------------------------------------------------------- regression pieces
def standardized_ols(X, y):
    """Standardised multiple regression with p-values and VIF. X: DataFrame, y: Series (complete cases)."""
    n, k = X.shape
    if k == 0 or n <= k + 2:
        return None
    sx = X.std(ddof=0)
    keep = sx[sx > 0].index.tolist()
    X = X[keep]
    k = len(keep)
    if k == 0 or y.std(ddof=0) == 0:
        return None
    A = ((X - X.mean()) / X.std(ddof=0)).values
    b = ((y - y.mean()) / y.std(ddof=0)).values
    beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = b - A @ beta
    rss, tss = float(resid @ resid), float(b @ b)
    dof = n - k - 1
    r2 = 1 - rss / tss if tss > 0 else np.nan
    adj = 1 - (1 - r2) * (n - 1) / dof if dof > 0 else np.nan
    pvals = [np.nan] * k
    try:
        cov = (rss / dof) * np.linalg.inv(A.T @ A)
        se = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        pvals = list(2 * stats.t.sf(np.abs(beta / se), dof))
    except Exception:
        pass
    vif = [np.nan] * k
    try:
        if k == 1:
            vif = [1.0]
        else:
            vif = list(np.diag(np.linalg.inv(np.corrcoef(A, rowvar=False))))
    except Exception:
        pass
    return {"features": keep, "beta": list(beta), "p": pvals, "vif": vif,
            "r2": float(r2), "adj_r2": float(adj), "n": int(n)}


def forest_importance(df, target, num_feats, cat_feats):
    """Cross-validated random forest + permutation importance (handles numbers and categories)."""
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import r2_score
        from sklearn.model_selection import KFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except Exception:
        return None

    feats = list(num_feats) + list(cat_feats)
    data = df.dropna(subset=[target])
    if len(data) < 30 or not feats:
        return None
    if len(data) > 3000:
        data = data.sample(3000, random_state=0)
    X = data[feats].copy()
    for c in cat_feats:
        X[c] = X[c].astype(str)
    y = data[target].astype(float).values

    transformers = []
    if num_feats:
        transformers.append(("num", SimpleImputer(strategy="median"), list(num_feats)))
    if cat_feats:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                             list(cat_feats)))
    folds = 5 if len(data) >= 60 else 3
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    r2s, imps = [], []
    try:
        for tr, te in kf.split(X):
            pipe = Pipeline([
                ("pre", ColumnTransformer(transformers)),
                ("rf", RandomForestRegressor(n_estimators=150, min_samples_leaf=3,
                                             random_state=0, n_jobs=1)),
            ])
            pipe.fit(X.iloc[tr], y[tr])
            pred = pipe.predict(X.iloc[te])
            r2s.append(r2_score(y[te], pred))
            pi = permutation_importance(pipe, X.iloc[te], y[te], n_repeats=3,
                                        random_state=0, scoring="r2")
            imps.append(pi.importances_mean)
    except Exception:
        return None
    mean_imp = np.mean(imps, axis=0)
    return {"r2_cv": float(np.mean(r2s)), "n": int(len(data)),
            "importance": {f: float(v) for f, v in zip(feats, mean_imp)}}


# --------------------------------------------------------------------------- target analysis
def analyze_target(df, kinds, target, pair_lookup):
    n_target = int(df[target].notna().sum())
    num_cols = [c for c in _numeric_cols(df, kinds) if c != target]
    seg_cols = _segment_cols(df, kinds, exclude=target)
    cat_cols = [c for c in seg_cols if kinds[c] == "categorical"]

    y = df[target].astype(float)

    # one-by-one relationships
    uni = {}
    for c in num_cols:
        r, p, n = _pearson(df[c], y)
        if np.isnan(r):
            continue
        uni[c] = {"r": r, "p": p, "n": n, "rho": _spearman(df[c], y)}

    # all numeric together (controls for overlap between variables)
    ols = None
    feat_for_ols = [c for c in uni.keys()]
    if feat_for_ols:
        cc = df[feat_for_ols + [target]].dropna()
        if len(cc) >= max(15, 4 * (len(feat_for_ols) + 1)):
            ols = standardized_ols(cc[feat_for_ols], cc[target].astype(float))
        elif len(cc) >= 12 and len(feat_for_ols) > 3:
            # too many variables for the rows we have: use only the 3 strongest
            top3 = sorted(feat_for_ols, key=lambda c: -abs(uni[c]["r"]))[:3]
            cc = df[top3 + [target]].dropna()
            if len(cc) >= 12:
                ols = standardized_ols(cc[top3], cc[target].astype(float))

    beta = {}
    if ols:
        sd_y = float(df[target].std())
        for f, b, p, v in zip(ols["features"], ols["beta"], ols["p"], ols["vif"]):
            sd_x = float(df[f].std())
            beta[f] = {"beta": float(b), "p": float(p) if not np.isnan(p) else np.nan,
                       "vif": float(v) if not np.isnan(v) else np.nan,
                       "slope": float(b) * sd_y / sd_x if sd_x else np.nan}

    forest = forest_importance(df, target, num_cols, cat_cols)
    rf_ok = bool(forest and forest["r2_cv"] >= 0.05)
    shares = {}
    if forest:
        pos = {k: max(0.0, v) for k, v in forest["importance"].items()}
        tot = sum(pos.values())
        shares = {k: (v / tot if tot > 0 else 0.0) for k, v in pos.items()}

    # categorical effects
    segs = analyze_segments(df, kinds, target)

    drivers = []
    for c, u in uni.items():
        b = beta.get(c, {})
        d = {"name": c, "kind": "numeric", "r": u["r"], "p": u["p"], "n": u["n"], "rho": u["rho"],
             "direction": "positive" if u["r"] > 0 else "negative",
             "strength": strength_label(u["r"]),
             "beta": b.get("beta"), "beta_p": b.get("p"), "vif": b.get("vif"), "slope": b.get("slope"),
             "share": shares.get(c) if rf_ok else None}
        d["rank_key"] = (d["share"] if (rf_ok and d["share"] is not None) else abs(u["r"]))
        drivers.append(d)
    for s in segs["categorical"]:
        d = {"name": s["name"], "kind": "categorical", "r": None, "p": s["p"], "n": s["n"],
             "eta2": s["eta2"], "direction": None,
             "strength": "large" if s["eta2"] >= 0.14 else "medium" if s["eta2"] >= 0.06 else "small",
             "share": shares.get(s["name"]) if rf_ok else None}
        d["rank_key"] = (d["share"] if (rf_ok and d["share"] is not None) else float(np.sqrt(max(s["eta2"], 0))))
        drivers.append(d)
    drivers.sort(key=lambda d: -d["rank_key"])

    model = {"r2_cv": forest["r2_cv"] if forest else None,
             "rf_reliable": rf_ok,
             "r2_linear": ols["r2"] if ols else None,
             "adj_r2_linear": ols["adj_r2"] if ols else None,
             "n_linear": ols["n"] if ols else None,
             "ranking_basis": ("cross-validated permutation importance" if rf_ok
                               else "absolute correlation (the predictive model was too weak or too small to rank reliably)")}

    # tiers of the top numeric driver
    top_num = next((d for d in drivers if d["kind"] == "numeric"), None)
    tiers = tier_analysis(df, top_num["name"], target) if top_num else None

    performers = high_vs_low(df, kinds, target, num_cols)

    return {"name": target, "n": n_target, "drivers": drivers, "model": model,
            "segments": segs, "tiers": tiers, "performers": performers}


def analyze_segments(df, kinds, target):
    out = {"categorical": []}
    seg_cols = _segment_cols(df, kinds, exclude=target)
    for c in seg_cols:
        sub = df[[c, target]].dropna()
        groups = {str(k): v[target].astype(float).values for k, v in sub.groupby(c)}
        groups = {k: v for k, v in groups.items() if len(v) >= 3}
        if len(groups) < 2:
            continue
        samples = list(groups.values())
        try:
            _, p = stats.kruskal(*samples)
            p = float(p)
        except Exception:
            p = np.nan
        allv = np.concatenate(samples)
        grand = allv.mean()
        sst = ((allv - grand) ** 2).sum()
        ssb = sum(len(s) * (s.mean() - grand) ** 2 for s in samples)
        eta2 = float(ssb / sst) if sst > 0 else 0.0
        levels = sorted(
            [{"level": k, "n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v))}
             for k, v in groups.items()], key=lambda d: -d["mean"])
        total_groups = int(sub[c].nunique())
        out["categorical"].append({"name": c, "n": int(sum(len(s) for s in samples)), "p": p,
                                   "eta2": eta2, "levels": levels,
                                   "small_groups": total_groups - len(groups)})
    out["categorical"].sort(key=lambda d: -d["eta2"])
    out["categorical"] = out["categorical"][:4]
    return out


def tier_analysis(df, feat, target):
    sub = df[[feat, target]].dropna()
    if len(sub) < 15 or sub[feat].nunique() < 3:
        return None
    try:
        q = pd.qcut(sub[feat], 3, labels=False, duplicates="drop")
    except Exception:
        return None
    nb = int(q.max()) + 1
    if nb < 2:
        return None
    names = ["Low", "Mid", "High"] if nb == 3 else ["Low", "High"]
    rows = []
    for i in range(nb):
        part = sub[q == i]
        rows.append({"tier": names[i], "n": int(len(part)),
                     "x_min": float(part[feat].min()), "x_max": float(part[feat].max()),
                     "mean": float(part[target].mean())})
    lo = sub[q == 0][target].values
    hi = sub[q == nb - 1][target].values
    try:
        _, p = stats.mannwhitneyu(hi, lo)
        p = float(p)
    except Exception:
        p = np.nan
    return {"feature": feat, "rows": rows, "gap": rows[-1]["mean"] - rows[0]["mean"], "p": p}


def high_vs_low(df, kinds, target, num_cols):
    sub = df.dropna(subset=[target])
    if len(sub) < 20:
        return []
    q1, q3 = sub[target].quantile([0.25, 0.75])
    lo, hi = sub[sub[target] <= q1], sub[sub[target] >= q3]
    if len(lo) < 5 or len(hi) < 5:
        return []
    out = []
    for c in num_cols:
        a, b = hi[c].dropna(), lo[c].dropna()
        if len(a) < 3 or len(b) < 3:
            continue
        pooled = np.sqrt((a.var() + b.var()) / 2)
        if not pooled or np.isnan(pooled):
            continue
        d = float((a.mean() - b.mean()) / pooled)
        out.append({"name": c, "d": d, "hi_mean": float(a.mean()), "lo_mean": float(b.mean())})
    out = [o for o in out if abs(o["d"]) >= 0.3]
    out.sort(key=lambda o: -abs(o["d"]))
    return out[:3]


def time_trend(df, kinds, target):
    dcols = [c for c in df.columns if kinds[c] == "datetime"]
    if not dcols or not target:
        return None
    c = dcols[0]
    sub = df[[c, target]].dropna().sort_values(c)
    if len(sub) < 12:
        return None
    span = (sub[c].max() - sub[c].min()).days
    if span <= 0:
        return None
    freq, label = ("MS", "month") if span > 730 else ("W", "week") if span > 60 else ("D", "day")
    ser = sub.set_index(c)[target].astype(float).resample(freq).mean().dropna()
    if len(ser) < 6:
        return None
    x = np.arange(len(ser))
    res = stats.linregress(x, ser.values)
    return {"date_col": c, "period": label, "n_periods": int(len(ser)),
            "first": float(ser.iloc[0]), "last": float(ser.iloc[-1]),
            "slope": float(res.slope), "p": float(res.pvalue),
            "direction": "rising" if res.slope > 0 else "falling",
            "start": str(ser.index[0].date()), "end": str(ser.index[-1].date())}


# --------------------------------------------------------------------------- findings (plain-language, rule based)
def _confidence(points):
    return "High" if points >= 4 else "Medium" if points >= 2 else "Low"


def build_findings(facts, max_findings=5):
    t = facts.get("target")
    if not t:
        return []
    target = t["name"]
    model = t["model"]
    findings = []
    for i, d in enumerate(t["drivers"][:max_findings], start=1):
        fid = f"F{i}"
        name = d["name"]
        if d["kind"] == "numeric":
            r, p, n = d["r"], d["p"], d["n"]
            pts = 2 if (p is not None and p < 0.01) else 1 if (p is not None and p < 0.05) else -1
            pts += 1 if n >= 100 else -1 if n < 30 else 0
            pts += 1 if abs(r) >= 0.3 else 0
            caveats = []
            if d.get("beta") is not None:
                if np.sign(d["beta"]) == np.sign(r):
                    pts += 1
                else:
                    pts -= 1
                    caveats.append("the direction reverses once other variables are held constant, "
                                   "so this link is probably confounded and unreliable")
            if (d.get("beta") is not None and np.sign(d["beta"]) == np.sign(r)
                    and abs(d["beta"]) - abs(r) > 0.25):
                caveats.append(f"its link is much stronger once other variables are held constant "
                               f"(standardized effect={sg(d['beta'])} vs r={sg(r)}), which usually means "
                               "other variables partly hide it in the raw correlation")
            if d.get("vif") is not None and not np.isnan(d["vif"]) and d["vif"] > 5:
                pts -= 1
                caveats.append(f"it overlaps heavily with other variables (VIF={g(d['vif'])}), "
                               "so its separate effect is hard to isolate")
            if d.get("share") is not None and d["share"] >= 0.15:
                pts += 1
            if 0.1 <= abs(r) < 0.3:
                caveats.append(f"alone it explains only {pc(100 * r * r)} of the variation in {target}")
            if abs((d.get("rho") or 0) - r) > 0.15:
                caveats.append(f"Spearman rho={sg(d['rho'])} differs from r, so the pattern may be "
                               "non-linear or driven by a few extreme rows")
            r2pct = 100 * r * r
            if d["strength"] == "negligible":
                text = (f"{name} shows no meaningful association with {target} "
                        f"(r={sg(r)}, {fmt_p(p)}, n={n}).")
                pts = min(pts, 1)
                caveats = [c for c in caveats if "reverses" not in c]
            else:
                word = "higher" if r > 0 else "lower"
                text = (f"{name} has a {d['strength']} {d['direction']} association with {target} "
                        f"(r={sg(r)}, {fmt_p(p)}, n={n}): rows with higher {name} tend to have {word} {target}.")
            ev = [f"Spearman rho={sg(d['rho'])}", f"on its own it explains {pc(r2pct)} of the variation in {target}"]
            if d.get("beta") is not None:
                ev.append(f"holding the other numeric variables constant: standardized effect={sg(d['beta'])} "
                          f"({fmt_p(d['beta_p'])}), about {g(d['slope'])} {target} per +1 {name}")
            if d.get("share") is not None:
                ev.append(f"predictive importance share={pc(100 * d['share'])} (cross-validated)")
            ev_text = "; ".join(ev) + "."
            findings.append({"id": fid, "variable": name, "kind": "numeric", "text": text,
                             "evidence": ev_text, "confidence": _confidence(pts),
                             "caveat": ("; ".join(caveats) + ".") if caveats else "",
                             "direction": d["direction"]})
        else:
            seg = next((s for s in t["segments"]["categorical"] if s["name"] == name), None)
            if not seg:
                continue
            hi, lo = seg["levels"][0], seg["levels"][-1]
            p = seg["p"]
            pts = 2 if (p is not None and p < 0.01) else 1 if (p is not None and p < 0.05) else -1
            pts += 1 if seg["eta2"] >= 0.14 else 0
            pts += 1 if min(l["n"] for l in seg["levels"]) >= 10 else 0
            pts -= 1 if seg["small_groups"] else 0
            pts -= 1 if seg["n"] < 30 else 0
            caveats = []
            if seg["small_groups"]:
                caveats.append(f"{seg['small_groups']} very small group(s) with fewer than 3 rows were left out")
            text = (f"{target} differs across {name} groups: '{hi['level']}' averages {g(hi['mean'])} "
                    f"(n={hi['n']}) versus '{lo['level']}' at {g(lo['mean'])} (n={lo['n']}) "
                    f"(eta-squared={g(seg['eta2'])}, {fmt_p(p)}).")
            ev = f"Group differences explain about {pc(100 * seg['eta2'])} of the variation in {target}."
            if d.get("share") is not None:
                ev += f" Predictive importance share={pc(100 * d['share'])} (cross-validated)."
            findings.append({"id": fid, "variable": name, "kind": "categorical", "text": text,
                             "evidence": ev, "confidence": _confidence(pts),
                             "caveat": ("; ".join(caveats) + ".") if caveats else "",
                             "direction": None})
    return findings


def build_caveats(facts):
    out = ["These are associations in the data you supplied, not proof of cause and effect.",
           "Only the columns in the file are considered; factors that were not recorded can change the picture."]
    n = facts["dataset"]["rows"]
    if n < 100:
        out.append(f"With only {n} rows, treat results as directional and re-check as more data arrives.")
    t = facts.get("target")
    if t:
        m = t["model"]
        r2 = m.get("r2_cv")
        if r2 is not None and r2 < 0.3:
            out.append(f"The variables available explain only a limited part of {t['name']} "
                       f"(cross-validated R-squared={g(r2)}); expect large unexplained variation.")
        if m.get("r2_linear") is not None and m["r2_linear"] < 0.3 and (r2 is None or r2 >= 0.3):
            out.append(f"A straight-line model explains only {pc(100 * m['r2_linear'])} of {t['name']}.")
    return out


# --------------------------------------------------------------------------- master function
def compute_facts(df, dataset_name, target=None, target_source="chosen by user"):
    kinds = column_kinds(df)
    if target is not None and (target not in df.columns or kinds.get(target) != "numeric"):
        target = None
    num_cols = _numeric_cols(df, kinds)
    pairs, lookup = pair_table(df, num_cols)

    facts = {
        "dataset": {"name": dataset_name, "rows": int(len(df)), "cols": int(df.shape[1])},
        "kinds": kinds,
        "profile": profile_data(df, kinds),
        "numeric_summary": {},
        "top_pairs": pairs[:8],
        "_pairs": lookup,
        "target": None,
        "time_trend": None,
    }

    if target:
        t = analyze_target(df, kinds, target, lookup)
        t["source"] = target_source
        facts["target"] = t
        facts["time_trend"] = time_trend(df, kinds, target)
        focus = [target] + [d["name"] for d in t["drivers"] if d["kind"] == "numeric"][:6]
    else:
        focus = num_cols[:8]
    facts["numeric_summary"] = numeric_summary(df, focus)

    facts["findings"] = build_findings(facts)
    facts["caveats"] = build_caveats(facts)
    facts["driver_table"] = driver_table(facts)
    return facts


def driver_table(facts):
    t = facts.get("target")
    if not t:
        return pd.DataFrame()
    rows = []
    for d in t["drivers"]:
        rows.append({
            "Variable": d["name"], "Type": d["kind"],
            "Direction": d["direction"] or "-",
            "Correlation r": None if d["r"] is None else round(d["r"], 3),
            "p-value": None if d["p"] is None or np.isnan(d["p"]) else round(d["p"], 4),
            "Strength": d["strength"],
            "Adjusted effect (std. beta)": None if d.get("beta") is None else round(d["beta"], 3),
            "Importance share %": None if d.get("share") is None else round(100 * d["share"], 1),
            "Rows used": d["n"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- fact sheet text for the AI
def facts_to_text(facts, goal=""):
    L = []
    ds = facts["dataset"]
    L.append(f"DATASET: {ds['name']} | rows={ds['rows']} | columns={ds['cols']}")
    if goal:
        L.append(f"USER GOAL: {goal}")
    q = facts["profile"]
    L.append(f"DATA QUALITY: {q['verdict']} (score {q['score']}/100)")
    for i in q["issues"]:
        L.append(f"  - [{i['severity']}] {i['text']}")

    t = facts.get("target")
    ns = facts["numeric_summary"]
    if t:
        L.append(f"OUTCOME VARIABLE: {t['name']} ({t['source']}), rows with a value={t['n']}")
    for c, s in ns.items():
        L.append(f"SUMMARY {c}: mean={g(s['mean'])}, median={g(s['median'])}, sd={g(s['std'])}, "
                 f"min={g(s['min'])}, max={g(s['max'])}, Q1={g(s['q1'])}, Q3={g(s['q3'])}")

    if t:
        L.append(f"RANKING BASIS: {t['model']['ranking_basis']}")
        if t["model"]["r2_cv"] is not None:
            L.append(f"MODEL: cross-validated R-squared={g(t['model']['r2_cv'])}")
        if t["model"]["r2_linear"] is not None:
            L.append(f"LINEAR MODEL: R-squared={g(t['model']['r2_linear'])}, "
                     f"adjusted R-squared={g(t['model']['adj_r2_linear'])}, rows={t['model']['n_linear']}")
        L.append(f"KEY FINDINGS about what is associated with {t['name']}:")
        for f in facts["findings"]:
            L.append(f"{f['id']} [{f['confidence']} confidence] {f['text']} Evidence: {f['evidence']}"
                     + (f" Caveat: {f['caveat']}" if f["caveat"] else ""))
        tiers = t.get("tiers")
        if tiers:
            parts = [f"{r['tier']} (range {g(r['x_min'])} to {g(r['x_max'])}): mean {t['name']}={g(r['mean'])}, n={r['n']}"
                     for r in tiers["rows"]]
            L.append(f"TIERS of {tiers['feature']}: " + "; ".join(parts)
                     + f"; High minus Low gap={g(tiers['gap'])} ({fmt_p(tiers['p'])})")
        for p in t["performers"]:
            L.append(f"TOP-VS-BOTTOM QUARTILE of {t['name']}: {p['name']} averages {g(p['hi_mean'])} in the top "
                     f"quartile vs {g(p['lo_mean'])} in the bottom quartile (effect size d={sg(p['d'])})")
        tt = facts.get("time_trend")
        if tt:
            L.append(f"TREND over time ({tt['date_col']}, by {tt['period']}): {tt['direction']}, "
                     f"from {g(tt['first'])} to {g(tt['last'])} between {tt['start']} and {tt['end']} ({fmt_p(tt['p'])})")

    pairs = facts["top_pairs"]
    tname = t["name"] if t else None
    others = [p for p in pairs if tname not in (p["a"], p["b"])][:5]
    if others:
        L.append("OTHER RELATIONSHIPS BETWEEN VARIABLES:")
        for i, p in enumerate(others, 1):
            L.append(f"P{i} {p['a']} vs {p['b']}: r={sg(p['r'])}, {fmt_p(p['p'])}, n={p['n']}")
    elif not t:
        for i, p in enumerate(pairs[:5], 1):
            L.append(f"P{i} {p['a']} vs {p['b']}: r={sg(p['r'])}, {fmt_p(p['p'])}, n={p['n']}")
    L.append("CAVEATS: " + " | ".join(facts["caveats"]))
    return "\n".join(L)


def schema_text(df, kinds, facts=None):
    """Compact schema for the 'chat with your data' feature."""
    L = [f"Rows: {len(df)}"]
    for c in df.columns:
        s = df[c]
        ex = ", ".join(str(v) for v in s.dropna().unique()[:4])
        L.append(f"- {c} [{kinds[c]}, dtype={s.dtype}] e.g. {ex[:60]}")
    return "\n".join(L)
