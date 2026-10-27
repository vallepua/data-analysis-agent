# agent/data_loader.py
"""Load and clean any CSV / Excel file."""
import warnings

import pandas as pd


def _blank_to_na(s):
    s2 = s.astype("string").str.strip()
    return s2.mask(s2 == "")


def _try_numeric(s):
    """Turn '$1,200', '12%', '(5)' style text columns into numbers."""
    nn = int(s.notna().sum())
    if nn == 0:
        return s, False
    s2 = _blank_to_na(s)
    nn = int(s2.notna().sum())
    if nn == 0:
        return s, False
    cleaned = s2.str.replace(r"[,$€£¥%\s]", "", regex=True)
    cleaned = cleaned.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    conv = pd.to_numeric(cleaned, errors="coerce")
    if int(conv.notna().sum()) >= 0.9 * nn:
        return conv.astype(float), True
    return s, False


def _try_datetime(s):
    nn = int(s.notna().sum())
    if nn < 5:
        return s, False
    sample = s.dropna().astype(str)
    if sample.str.contains(r"\d").mean() < 0.9:
        return s, False
    if sample.str.contains(r"[-/:]|[A-Za-z]{3}").mean() < 0.9:
        return s, False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            conv = pd.to_datetime(s, errors="coerce", format="mixed")
        except Exception:
            conv = pd.to_datetime(s, errors="coerce")
    if int(conv.notna().sum()) >= 0.9 * nn:
        return conv, True
    return s, False


def clean_dataframe(df):
    df = df.copy()
    names, seen = [], {}
    for i, c in enumerate(df.columns):
        name = str(c).strip() or f"column_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)
    df.columns = names

    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)

    for c in list(df.columns):
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s) \
                or pd.api.types.is_bool_dtype(s):
            continue
        s_num, ok = _try_numeric(s)
        if ok:
            df[c] = s_num
            continue
        s_dt, ok = _try_datetime(s)
        if ok:
            df[c] = s_dt
    return df


def load_data(uploaded_file):
    """Return (dataframe, error_message)."""
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin-1")
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported format. Please upload a CSV or Excel file."

        df = clean_dataframe(df)
        if df.empty or df.shape[1] == 0:
            return None, "The file has no usable data."
        return df, None
    except Exception as e:  # noqa
        return None, f"Error loading file: {e}"
