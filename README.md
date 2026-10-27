# 📊 Data Analysis Agent

Upload any CSV or Excel file, say what you are trying to decide, and get a **verified, decision-oriented report** with charts, plus a downloadable PDF.

## What makes it different
Most "AI data analysts" let the language model do the maths and then trust what it writes. This one doesn't:

1. **Statistics are computed in Python** (pandas, SciPy, scikit-learn), not by the AI.
2. The AI only writes the summary and recommendations from a **verified fact sheet**.
3. A **verifier re-checks every figure and direction claim** the AI writes. Sentences with an unverifiable number, a dropped minus sign, or the wrong direction are removed.

## Features
- 🎯 Asks what you are deciding and which column is the outcome
- 🧪 Data-quality check (missing values, duplicates, outliers, sample size) with a trust score
- 📈 Ranked drivers: correlation, regression that controls for overlap, cross-validated importance
- 🧩 Segment and tier comparisons; trend over time when a date column exists
- 📊 Charts chosen from the top findings, each with a title, "what it shows" and "so what"
- 🧾 Findings with evidence, confidence (High / Medium / Low) and caveats
- 💬 Optional "Ask your data" chat: the AI writes pandas code, runs it, and fixes its own errors
- 📥 PDF report download
- Works on any tabular dataset (any columns, any size up to 10 MB)

## Run locally
```bash
git clone https://github.com/vallepua/data-analysis-agent.git
cd data-analysis-agent
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your free Groq API key
streamlit run app.py
```
Get a free key at [console.groq.com](https://console.groq.com). The default model is `openai/gpt-oss-120b`; change it with `GROQ_MODEL`.

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (the `.env` file is git-ignored, so your key stays private).
2. On [share.streamlit.io](https://share.streamlit.io) create an app from the repo, main file `app.py`, Python 3.12.
3. In **Advanced settings → Secrets** add:
```toml
GROQ_API_KEY = "your_key_here"
ENABLE_CHAT = "0"
```
`ENABLE_CHAT = "0"` turns off the chat tab, which executes AI-written code and should not be exposed publicly without a real sandbox.

## Project structure
```
app.py                 Streamlit interface
agent/
  data_loader.py       Load and clean CSV / Excel
  stats_engine.py      All statistics, findings and the fact sheet
  visualizer.py        Charts with titles and explanations
  analyzer.py          AI summary written from the fact sheet
  verifier.py          Re-checks every figure and direction claim
  reporter.py          Report assembly and PDF
  pipeline.py          Runs everything end to end
  chat.py              "Ask your data" (AI-written pandas, self-correcting)
  llm.py               Groq client with rate-limit handling
```

## Limits
- **Decision support, not proof.** Results are associations in your file, not causes.
- Small samples (under about 100 rows) give unstable results; the report says so.
- Column units and meanings are not in the file, so the agent never assumes them.
- The free Groq tier limits tokens per minute, so heavy shared use may be slow.
- Your file's summary statistics (not raw rows) are sent to the AI service. The chat tab also sends 3 sample rows.

## Tech
Python, Streamlit, pandas, SciPy, scikit-learn, matplotlib, seaborn, fpdf2, Groq API.
