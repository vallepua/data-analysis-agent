# agent/visualizer.py
"""
Charts are chosen from the computed facts (top drivers first).
Every chart returns: path, title, aim (what it shows), so_what (what to take from it).
No AI-written plotting code, so there are no syntax errors and no 'filler' charts.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from agent.stats_engine import fmt_p, g, sg, _numeric_cols

BLUE, RED, GREEN, GREY, ORANGE = "#2C6FAC", "#C0392B", "#2E8B57", "#8C8C8C", "#E67E22"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, out_dir, idx):
    path = os.path.join(out_dir, f"chart_{idx}.png")
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _chart(path, title, aim, so_what):
    return {"path": path, "title": title, "aim": aim, "so_what": so_what}


# --------------------------------------------------------------------------- with an outcome variable
def _target_distribution(df, target, facts, out_dir, idx):
    y = df[target].dropna().astype(float)
    s = facts["numeric_summary"].get(target)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bins = int(min(30, max(8, np.sqrt(len(y)) * 1.5)))
    ax.hist(y, bins=bins, color=BLUE, edgecolor="white")
    ax.axvline(y.mean(), color=RED, ls="--", lw=1.8, label=f"Mean = {g(y.mean())}")
    ax.axvline(y.median(), color=GREEN, lw=1.8, label=f"Median = {g(y.median())}")
    ax.set_xlabel(target)
    ax.set_ylabel("Number of rows")
    ax.set_title(f"Distribution of {target}")
    ax.legend(frameon=False)
    path = _save(fig, out_dir, idx)
    so = (f"The typical {target} is about {g(y.median())} (median); the middle 50% of rows fall between "
          f"{g(y.quantile(.25))} and {g(y.quantile(.75))}.")
    if s and abs(s["skew"]) > 1:
        so += (f" The distribution is skewed (skew={g(s['skew'])}), so the mean is pulled toward the long "
               f"tail: use the median as the 'typical' value.")
    return _chart(path, f"How {target} is distributed",
                  f"Shows how the outcome '{target}' is spread across all rows: its typical value, how much it "
                  f"varies, and whether a few extreme rows pull the average.", so)


def _driver_ranking(df, target, facts, out_dir, idx):
    drivers = facts["target"]["drivers"][:8]
    if not drivers:
        return None
    nums = [d for d in drivers if d["kind"] == "numeric"]
    has_share = any(d.get("share") is not None for d in drivers)
    ncols = 2 if (nums and has_share) else 1
    fig, axes = plt.subplots(1, ncols, figsize=(11 if ncols == 2 else 7, 0.55 * max(len(drivers), 3) + 1.8))
    axes = np.atleast_1d(axes)
    ai = 0
    if nums:
        ax = axes[ai]
        ai += 1
        nd = sorted(nums, key=lambda d: abs(d["r"]))
        colors = [GREEN if d["r"] > 0 else RED for d in nd]
        bars = ax.barh([d["name"] for d in nd], [d["r"] for d in nd], color=colors)
        for b, d in zip(bars, nd):
            if d["p"] is not None and d["p"] >= 0.05:
                b.set_alpha(0.35)
                b.set_hatch("//")
            ax.text(d["r"] + (0.02 if d["r"] >= 0 else -0.02), b.get_y() + b.get_height() / 2,
                    sg(d["r"]), va="center", ha="left" if d["r"] >= 0 else "right", fontsize=9)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlim(-1.05, 1.05)
        ax.set_xlabel(f"Correlation with {target}  (green = higher together, red = opposite)")
        ax.set_title("Direction and strength")
    if has_share:
        ax = axes[ai]
        sd = sorted([d for d in drivers if d.get("share") is not None], key=lambda d: d["share"])
        ax.barh([d["name"] for d in sd], [100 * d["share"] for d in sd], color=BLUE)
        for i, d in enumerate(sd):
            ax.text(100 * d["share"] + 0.5, i, f"{100 * d['share']:.1f}%", va="center", fontsize=9)
        ax.set_xlabel("Share of predictive importance (%)")
        ax.set_title("Predictive importance (cross-validated)")
    fig.suptitle(f"What is associated with {target}?", fontweight="bold", y=1.02)
    fig.tight_layout()
    path = _save(fig, out_dir, idx)
    top = drivers[0]
    if top["kind"] == "numeric":
        so = (f"{top['name']} ranks first ({top['direction']} link, r={sg(top['r'])}). Faded hatched bars are not "
              f"statistically significant at the 5% level, so do not act on them alone.")
    else:
        so = f"The grouping '{top['name']}' ranks first (it separates {target} levels most clearly)."
    return _chart(path, f"Ranked drivers of {target}",
                  f"Ranks every available variable by how strongly it moves with '{target}'. Left: direction and "
                  f"strength of the straight-line link. Right: how much each variable improves prediction "
                  f"(also covers category columns).", so)


def _scatter(df, target, driver, facts, out_dir, idx):
    x_col = driver["name"]
    sub = df[[x_col, target]].dropna()
    if len(sub) < 5:
        return None
    x, y = sub[x_col].astype(float), sub[target].astype(float)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    pts = sub.sample(2000, random_state=0) if len(sub) > 2000 else sub
    ax.scatter(pts[x_col], pts[target], s=34 if len(pts) < 300 else 12, alpha=0.65,
               color=BLUE, edgecolor="white", linewidth=0.5)
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, m * xs + b, color=RED, lw=2, label="Trend line")
    ax.set_xlabel(x_col)
    ax.set_ylabel(target)
    ax.set_title(f"{target} vs {x_col}")
    ax.text(0.02, 0.96, f"r = {sg(driver['r'])}   {fmt_p(driver['p'])}   n = {driver['n']}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec=GREY, alpha=0.9))
    ax.legend(frameon=False, loc="best")
    path = _save(fig, out_dir, idx)
    so = (f"On average, each +1 in {x_col} goes with {g(m)} change in {target} (r={sg(driver['r'])}, "
          f"{fmt_p(driver['p'])}). This is an association, not proof that changing {x_col} will change {target}.")
    if driver.get("rho") is not None and abs(driver["rho"] - driver["r"]) > 0.15:
        so += f" Spearman rho={sg(driver['rho'])} differs from r, so the pattern may be curved or driven by outliers."
    return _chart(path, f"{target} vs {x_col}",
                  f"Each dot is one row. Shows whether '{target}' tends to rise or fall as '{x_col}' increases, "
                  f"and how tightly the dots follow the trend line.", so)


def _segment_chart(df, target, facts, out_dir, idx):
    t = facts["target"]
    segs = t["segments"]["categorical"]
    if segs:
        seg = segs[0]
        col = seg["name"]
        levels = [l["level"] for l in sorted(seg["levels"], key=lambda l: -l["median"])][:10]
        sub = df[[col, target]].dropna()
        sub = sub.assign(_k=sub[col].astype(str))
        data = [sub.loc[sub["_k"] == lv, target].astype(float).values for lv in levels]
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.boxplot(data, vert=True, patch_artist=True, showmeans=True,
                   boxprops=dict(facecolor="#DCE9F5"), medianprops=dict(color=RED, lw=2))
        ax.set_xticks(range(1, len(levels) + 1))
        ax.set_xticklabels([f"{lv}\n(n={len(d)})" for lv, d in zip(levels, data)], fontsize=9)
        ax.set_ylabel(target)
        ax.set_title(f"{target} by {col}")
        path = _save(fig, out_dir, idx)
        hi, lo = seg["levels"][0], seg["levels"][-1]
        so = (f"'{hi['level']}' averages {g(hi['mean'])} vs '{lo['level']}' at {g(lo['mean'])} "
              f"({fmt_p(seg['p'])}). Groups explain about {100 * seg['eta2']:.1f}% of the variation in {target}.")
        return _chart(path, f"{target} compared across {col} groups",
                      f"Compares '{target}' between the groups in '{col}'. The box shows the middle 50% of each "
                      f"group, the red line is the median and the green triangle the mean.", so)
    tiers = t.get("tiers")
    if tiers:
        fig, ax = plt.subplots(figsize=(7, 4.4))
        names = [f"{r['tier']}\n({g(r['x_min'])} to {g(r['x_max'])})" for r in tiers["rows"]]
        means = [r["mean"] for r in tiers["rows"]]
        bars = ax.bar(names, means, color=[BLUE, "#6FA3D1", GREEN][: len(means)])
        for bar, r in zip(bars, tiers["rows"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{g(r['mean'])}\n(n={r['n']})",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylabel(f"Average {target}")
        ax.set_xlabel(f"{tiers['feature']} tier")
        ax.set_title(f"Average {target} by {tiers['feature']} tier")
        ax.set_ylim(0, max(means) * 1.25 if max(means) > 0 else 1)
        path = _save(fig, out_dir, idx)
        so = (f"Moving from the Low to the High {tiers['feature']} tier, average {target} changes by "
              f"{g(tiers['gap'])} ({fmt_p(tiers['p'])}).")
        return _chart(path, f"{target} across {tiers['feature']} tiers",
                      f"Splits rows into low / mid / high thirds of '{tiers['feature']}' and compares the average "
                      f"'{target}' in each, so you can see which tier performs best.", so)
    return None


def _heatmap(df, facts, out_dir, idx):
    kinds = facts["kinds"]
    target = facts["target"]["name"] if facts.get("target") else None
    nums = _numeric_cols(df, kinds)
    if len(nums) < 3:
        return None
    if target and target in nums:
        others = sorted([c for c in nums if c != target],
                        key=lambda c: -abs(df[c].corr(df[target]) if df[c].nunique() > 1 else 0))[:11]
        cols = [target] + others
    else:
        cols = nums[:12]
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    size = max(5.5, 0.75 * len(cols) + 2)
    fig, ax = plt.subplots(figsize=(size + 1, size))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Correlation (r)"})
    ax.set_title("Correlation matrix")
    path = _save(fig, out_dir, idx)
    pairs = [p for p in facts["top_pairs"] if p["a"] in cols and p["b"] in cols][:2]
    so = "Strongest pairs: " + "; ".join(f"{p['a']} & {p['b']} (r={sg(p['r'])})" for p in pairs) + "." if pairs \
        else "No strong pairwise relationships."
    so += " Variables that move together (|r| above about 0.7) carry overlapping information."
    return _chart(path, "How all numeric variables relate to each other",
                  "Each cell is the correlation between two columns: +1 means they rise together, -1 means one "
                  "rises as the other falls, 0 means no straight-line link. Red/blue = stronger.", so)


def _time_chart(df, target, facts, out_dir, idx):
    tt = facts.get("time_trend")
    if not tt:
        return None
    c = tt["date_col"]
    freq = {"month": "MS", "week": "W", "day": "D"}[tt["period"]]
    sub = df[[c, target]].dropna().sort_values(c)
    ser = sub.set_index(c)[target].astype(float).resample(freq).mean().dropna()
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(ser.index, ser.values, marker="o", color=BLUE, lw=2)
    x = np.arange(len(ser))
    m, b = np.polyfit(x, ser.values, 1)
    ax.plot(ser.index, m * x + b, color=RED, ls="--", label="Trend")
    ax.set_title(f"Average {target} per {tt['period']}")
    ax.set_ylabel(target)
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    path = _save(fig, out_dir, idx)
    so = (f"{target} is {tt['direction']} over time: from {g(tt['first'])} to {g(tt['last'])} "
          f"({tt['start']} to {tt['end']}, {fmt_p(tt['p'])}).")
    return _chart(path, f"{target} over time",
                  f"Shows how the average '{target}' changes across time, using the date column '{c}'.", so)


# --------------------------------------------------------------------------- general charts (any dataset)
def _outlier_chart(df, facts, out_dir, idx):
    kinds = facts["kinds"]
    nums = _numeric_cols(df, kinds)[:10]
    if not nums:
        return None
    z = {}
    for c in nums:
        x = df[c].dropna().astype(float)
        if x.std() > 0:
            z[c] = ((x - x.mean()) / x.std()).values
    if not z:
        return None
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(z) + 3), 4.6))
    ax.boxplot(list(z.values()), patch_artist=True, boxprops=dict(facecolor="#DCE9F5"),
               medianprops=dict(color=RED, lw=2), flierprops=dict(marker="o", markerfacecolor=ORANGE, markersize=4))
    ax.set_xticks(range(1, len(z) + 1))
    ax.set_xticklabels(list(z.keys()), rotation=30, ha="right")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_ylabel("Standard deviations from the column average")
    ax.set_title("Outlier check (all columns on the same scale)")
    path = _save(fig, out_dir, idx)
    prof = {c["name"]: c for c in facts["profile"]["columns"]}
    flagged = [f"{c} ({prof[c]['outliers']} row{'s' if prof[c]['outliers'] != 1 else ''})"
               for c in z if prof[c]["outliers"] > 0]
    so = ("Rows beyond 1.5 x IQR: " + ", ".join(flagged) + ". Check these for entry errors before relying on averages."
          if flagged else "No column has extreme values beyond 1.5 x IQR, so averages are not distorted by outliers.")
    return _chart(path, "Do any columns contain extreme values?",
                  "Puts every numeric column on the same standardized scale so extreme values (orange dots) stand "
                  "out, even in columns with small numbers.", so)


def _missing_chart(df, facts, out_dir, idx):
    cols = [c for c in facts["profile"]["columns"] if c["missing"] > 0 and c["missing_pct"] >= 5]
    if not cols:
        return None
    cols = sorted(cols, key=lambda c: c["missing_pct"])
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(cols) + 2))
    ax.barh([c["name"] for c in cols], [c["missing_pct"] for c in cols], color=ORANGE)
    for i, c in enumerate(cols):
        ax.text(c["missing_pct"] + 0.3, i, f"{c['missing_pct']}%", va="center", fontsize=9)
    ax.set_xlabel("% of rows missing")
    ax.set_title("Missing values by column")
    path = _save(fig, out_dir, idx)
    worst = cols[-1]
    return _chart(path, "Where is data missing?",
                  "Shows how much of each column is blank. Columns with many blanks give less reliable results.",
                  f"'{worst['name']}' is the most incomplete column ({worst['missing_pct']}% missing).")


def _distribution_grid(df, facts, out_dir, idx):
    nums = _numeric_cols(df, facts["kinds"])[:6]
    if not nums:
        return None
    n = len(nums)
    cols = 3 if n > 4 else 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, nums):
        ax.hist(df[c].dropna().astype(float), bins=15, color=BLUE, edgecolor="white")
        ax.set_title(c, fontsize=10)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    path = _save(fig, out_dir, idx)
    sk = [(c, facts["numeric_summary"].get(c, {}).get("skew", 0)) for c in nums]
    sk = [f"{c} (skew={g(s)})" for c, s in sk if abs(s) > 1]
    so = ("Strongly skewed columns: " + ", ".join(sk) + ". Medians describe these better than averages." if sk
          else "Columns are reasonably balanced, so averages are a fair summary.")
    return _chart(path, "Shape of each numeric column",
                  "One small histogram per numeric column: where most values sit and how widely they spread.", so)


def _pair_scatter(df, pair, out_dir, idx):
    a, b = pair["a"], pair["b"]
    sub = df[[a, b]].dropna()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    pts = sub.sample(2000, random_state=0) if len(sub) > 2000 else sub
    ax.scatter(pts[a], pts[b], s=30 if len(pts) < 300 else 10, alpha=0.65, color=BLUE, edgecolor="white")
    m, c0 = np.polyfit(sub[a].astype(float), sub[b].astype(float), 1)
    xs = np.linspace(sub[a].min(), sub[a].max(), 50)
    ax.plot(xs, m * xs + c0, color=RED, lw=2)
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.set_title(f"{b} vs {a}")
    ax.text(0.02, 0.96, f"r = {sg(pair['r'])}   {fmt_p(pair['p'])}   n = {pair['n']}", transform=ax.transAxes,
            va="top", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec=GREY, alpha=0.9))
    path = _save(fig, out_dir, idx)
    return _chart(path, f"Strongest relationship: {a} and {b}",
                  f"The pair of numeric columns with the strongest straight-line link in this dataset.",
                  f"r={sg(pair['r'])} ({fmt_p(pair['p'])}, n={pair['n']}): they tend to move "
                  f"{'together' if pair['r'] > 0 else 'in opposite directions'}. Association only.")


def _category_counts(df, facts, out_dir, idx):
    kinds = facts["kinds"]
    cats = [c for c in df.columns if kinds[c] == "categorical"]
    if not cats:
        return None
    col = cats[0]
    vc = df[col].astype(str).value_counts().head(10).sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(vc) + 2))
    ax.barh(vc.index, vc.values, color=BLUE)
    for i, v in enumerate(vc.values):
        ax.text(v, i, f" {v}", va="center", fontsize=9)
    ax.set_xlabel("Number of rows")
    ax.set_title(f"Rows per {col}")
    path = _save(fig, out_dir, idx)
    top = vc.index[-1]
    return _chart(path, f"Breakdown of '{col}'",
                  f"Counts how many rows fall in each '{col}' category.",
                  f"'{top}' is the largest group with {int(vc.iloc[-1])} of {int(df[col].notna().sum())} rows.")


# --------------------------------------------------------------------------- master function
def create_charts(df, facts, out_dir, max_charts=9):
    os.makedirs(out_dir, exist_ok=True)
    charts = []

    def add(fn, *args):
        if len(charts) >= max_charts:
            return
        try:
            c = fn(*args, out_dir, len(charts) + 1)
        except Exception as e:  # never let one bad chart stop the report
            print(f"Chart skipped ({fn.__name__}): {e}")
            plt.close("all")
            return
        if c:
            charts.append(c)

    t = facts.get("target")
    if t:
        target = t["name"]
        add(_target_distribution, df, target, facts)
        add(_driver_ranking, df, target, facts)
        for d in [d for d in t["drivers"] if d["kind"] == "numeric"][:3]:
            add(_scatter, df, target, d, facts)
        add(_segment_chart, df, target, facts)
        add(_time_chart, df, target, facts)
        add(_heatmap, df, facts)
        for fn in (_missing_chart, _outlier_chart):
            add(fn, df, facts)
    else:
        add(_distribution_grid, df, facts)
        add(_heatmap, df, facts)
        for p in facts["top_pairs"][:2]:
            add(_pair_scatter, df, p)
        add(_category_counts, df, facts)
        for fn in (_missing_chart, _outlier_chart):
            add(fn, df, facts)
    return charts
