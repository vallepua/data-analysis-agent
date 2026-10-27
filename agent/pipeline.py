# agent/pipeline.py
"""One function that runs the whole analysis. The Streamlit app just calls this."""
import os
import tempfile

from agent.analyzer import generate_narrative
from agent.reporter import assemble_report, save_report_as_pdf
from agent.stats_engine import compute_facts
from agent.visualizer import create_charts


def run_analysis(df, dataset_name, target=None, target_source="chosen by user",
                 goal="", progress=None):
    def step(msg):
        if progress:
            progress(msg)

    out_dir = tempfile.mkdtemp(prefix="dag_")

    step("Profiling data quality and computing statistics...")
    facts = compute_facts(df, dataset_name, target, target_source)

    step("Choosing and drawing charts from the top findings...")
    charts = create_charts(df, facts, out_dir)

    step("Writing the report and checking every figure...")
    hint = ("Tailor the actions to the user's stated decision." if goal
            else "No decision was stated, so keep the actions general.")
    sections, verification, fact_text = generate_narrative(facts, goal, hint)
    report_md = assemble_report(facts, sections, verification, goal)

    step("Building the PDF...")
    pdf_path = save_report_as_pdf(report_md, charts, dataset_name, out_dir)
    pdf_bytes = None
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

    for c in charts:                      # keep chart bytes so the app survives temp-file cleanup
        with open(c["path"], "rb") as f:
            c["bytes"] = f.read()

    return {"facts": facts, "charts": charts, "report_md": report_md, "pdf_bytes": pdf_bytes,
            "verification": verification, "fact_text": fact_text, "out_dir": out_dir}
