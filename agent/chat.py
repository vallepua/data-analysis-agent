# agent/chat.py
"""
'Chat with your data': the AI writes pandas code, we run it, and if it fails the error is sent
back so the AI can fix its own code (up to 3 tries). The final explanation is verified too.

NOTE: this is a restricted runner (no imports, no file/network names, limited builtins), good for
local/student use. If you ever expose the app publicly, run this code in a real sandbox.
"""
import builtins
import io
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from agent import llm
from agent.stats_engine import g, schema_text
from agent.verifier import verify_and_clean

BLOCKED = re.compile(
    r"\b(import|exec|eval|open|compile|globals|locals|getattr|setattr|delattr|input|breakpoint|"
    r"os|sys|subprocess|shutil|pathlib|socket|requests|urllib|pickle|to_pickle|read_pickle|to_csv|"
    r"to_excel|to_parquet|read_csv|read_excel|read_json|system|popen)\b|__\w+__")
STRING_LITERAL = re.compile(r"(\"\"\".*?\"\"\"|'''.*?'''|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')", re.S)

SAFE_BUILTINS = {n: getattr(builtins, n)
                 for n in ["abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len", "list",
                           "max", "min", "range", "round", "set", "sorted", "str", "sum", "tuple", "zip",
                           "isinstance", "map", "filter", "reversed", "pow", "divmod", "print"]}


def _plain(answer):
    if answer is None:
        return "See the table or chart below."
    if isinstance(answer, (int, float, np.integer, np.floating)):
        return g(answer)
    return str(answer)


def _is_safe(code):
    stripped = STRING_LITERAL.sub('""', code)
    m = BLOCKED.search(stripped)
    return (m is None), (m.group(0) if m else "")


def _clean_code(text):
    text = re.sub(r"^```(?:python)?", "", text.strip(), flags=re.I).strip()
    return re.sub(r"```$", "", text).strip()


def _code_prompt(question, df, kinds, facts, goal, previous=None, error=None):
    ctx = f"User goal: {goal}\n" if goal else ""
    t = facts.get("target") if facts else None
    if t:
        ctx += f"Main outcome column: {t['name']}\n"
    base = f"""You write pandas code to answer a question about a dataframe called df.
{ctx}COLUMNS (use these exact names):
{schema_text(df, kinds)}

SAMPLE ROWS:
{df.head(3).to_string()[:900]}

RULES:
- Available names: df, pd, np, plt, sns, stats (scipy.stats). Do NOT write import statements.
- No file access, no network, no input().
- Put the final answer in a variable `answer` (a number, short string or small dict).
- If a table helps, put a DataFrame in `table` (max 30 rows).
- If a chart is requested, draw it with matplotlib/seaborn (plt.figure(...) first, give it a title and axis labels). Do not call plt.show().
- Handle missing values (dropna) where needed. Do not modify df in place.
- Return ONLY Python code, no explanations, no markdown fences.

QUESTION: {question}"""
    if previous and error:
        base += f"\n\nYour previous code failed:\n{previous}\n\nERROR: {error}\nFix it and return the full corrected code."
    return base


def _run(code, df):
    ok, word = _is_safe(code)
    if not ok:
        raise ValueError(f"Blocked for safety: '{word}' is not allowed")
    plt.close("all")
    ns = {"__builtins__": SAFE_BUILTINS, "df": df.copy(), "pd": pd, "np": np, "plt": plt,
          "sns": sns, "stats": stats}
    exec(code, ns)
    image = None
    if plt.get_fignums():
        buf = io.BytesIO()
        plt.gcf().savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
        image = buf.getvalue()
    plt.close("all")
    table = ns.get("table")
    if table is not None and not isinstance(table, (pd.DataFrame, pd.Series)):
        table = None
    if isinstance(table, pd.Series):
        table = table.to_frame()
    if isinstance(table, pd.DataFrame):
        table = table.head(30).reset_index() if table.index.name or not isinstance(table.index, pd.RangeIndex) \
            else table.head(30)
    return ns.get("answer"), table, image


def answer_question(question, df, kinds, facts=None, goal=""):
    """Return dict: text, table, image, code, error, attempts."""
    result = {"text": "", "table": None, "image": None, "code": "", "error": "", "attempts": 0}
    code, err = None, None
    for attempt in range(1, 4):
        result["attempts"] = attempt
        try:
            raw = llm.call_llm(_code_prompt(question, df, kinds, facts, goal, code, err),
                               max_tokens=1200,
                               system="You are an expert pandas programmer. Output only runnable Python code.")
        except Exception as e:  # noqa
            result["error"] = str(e)
            return result
        code = _clean_code(raw)
        result["code"] = code
        try:
            answer, table, image = _run(code, df)
            result["table"], result["image"], result["error"] = table, image, ""
            break
        except Exception as e:  # noqa  -> send the error back so the AI can fix its own code
            err = f"{type(e).__name__}: {e}"
            result["error"] = err
    else:
        result["text"] = "I could not produce working code for that question after 3 tries. Try rephrasing it."
        return result

    preview = ""
    if answer is not None:
        preview += f"ANSWER VALUE: {str(answer)[:800]}\n"
    if result["table"] is not None:
        preview += "TABLE:\n" + result["table"].head(12).to_string()[:1200]
    if not preview.strip() and result["image"] is not None:
        result["text"] = "Here is the chart you asked for."
        return result
    try:
        expl = llm.call_llm(
            f"Question: {question}\n\nRESULT COMPUTED BY CODE:\n{preview}\n\n"
            "In at most 3 sentences, answer the question for a business reader using ONLY numbers that appear in "
            "the result above (copy them exactly, keep minus signs). Do not add new calculations.",
            max_tokens=500, system="You explain results accurately and never invent numbers.")
        clean, rep = verify_and_clean(expl, preview, question)
        result["text"] = clean.strip() or _plain(answer)
        if rep["removed"]:
            result["text"] += "\n\n_(Some wording was removed because it contained figures not in the result.)_"
    except Exception:
        result["text"] = _plain(answer)
    return result
