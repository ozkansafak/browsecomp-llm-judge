"""Generates browsecomp_eval.ipynb in the current directory."""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# ── Cell sources ──────────────────────────────────────────────────────────────

CELL_HEADER = """\
# BrowseComp LLM Eval Harness

Evaluate two OpenAI models on 20 BrowseComp tasks using LLM-as-a-judge.
Models: `gpt-4o-mini` vs `gpt-4o`
Sampling: 3 runs per task per model → 120 total inference calls
"""

CELL_SETUP = """\
from openai import OpenAI
import pandas as pd
import json, time, os, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

client = OpenAI()   # reads OPENAI_API_KEY from env
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
print("OpenAI client ready.")
"""

CELL_DATA = """\
# ── Load tasks from CSV ───────────────────────────────────────────────────────
CSV_PATH = "n=20 Browsecomp data - BrowseComp n=20.csv"

df_raw = pd.read_csv(CSV_PATH)
df_tasks = df_raw.rename(columns={
    "Task ID":    "task_id",
    "Domain":     "domain",
    "Workflow":   "workflow",
    "Prompt":     "prompt",
    "Rubric JSON":"rubric_json",
})[["task_id", "domain", "prompt", "rubric_json"]]

print(f"Loaded {len(df_tasks)} tasks")
df_tasks[["task_id", "domain"]].assign(
    prompt_preview=df_tasks["prompt"].str[:80] + "..."
)
"""

CELL_CONFIG = """\
# ── Configuration ────────────────────────────────────────────────────────────
MODELS = {
    "gpt-4o-mini":  "gpt-4o-mini",   # small, fast, cheap
    "gpt-4.1-mini": "gpt-4.1-mini",  # small, newer
}
JUDGE_MODEL = "gpt-4o"  # stronger than both test models; rubric has ground truth so no deep reasoning needed
N_SAMPLES    = 3          # samples per task per model
MAX_TOOL_STEPS = 5        # max agentic steps
MAX_TOKENS   = 1500
USE_WEB_SEARCH = True     # set False to skip tool calls (faster but dumber)
N_WORKERS    = 10         # parallel threads

WEIGHT_MAP = {
    "Primary objective(s)":  1.0,
    "Supporting detail(s)":  0.5,
}
print("Config OK")
"""

CELL_TOOLS = """\
# ── Tool implementations ──────────────────────────────────────────────────────
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

def web_search(query: str, max_results: int = 5) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Search error: {e}"

def fetch_url(url: str, max_chars: int = 5000) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (research-bot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "lxml")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\\n", strip=True)
        return text[:max_chars]
    except Exception as e:
        return f"Fetch error: {e}"

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web. Use to find documents, statistics, or URLs.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch and read text from a URL. Use after finding a relevant URL.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "URL to fetch"}},
                "required": ["url"],
            },
        },
    },
]
TOOL_EXECUTORS = {"web_search": web_search, "fetch_url": fetch_url}
print("Tools ready.")
"""

CELL_INFERENCE = """\
# ── Inference (agentic loop) ──────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert research analyst with web search capabilities. "
    "For each question, search the web to find precise data from primary sources. "
    "Follow URLs to retrieve exact numbers from documents. "
    "End with a concise, direct answer."
)

def run_inference(model: str, prompt: str, use_tools: bool = True) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    tools = TOOL_SCHEMAS if use_tools else []
    tool_log = []

    for step in range(MAX_TOOL_STEPS):
        kwargs = dict(model=model, max_tokens=MAX_TOKENS, temperature=0.7, messages=messages)
        if tools:
            kwargs["tools"] = tools

        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        if msg.tool_calls:
            messages.append(msg)   # assistant turn with tool_calls
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = TOOL_EXECUTORS[tc.function.name](**args)
                tool_log.append({"tool": tc.function.name, "input": args,
                                  "result_preview": result[:300]})
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result,
                })
        else:
            text = msg.content or ""
            return {"response": text, "tool_log": tool_log, "steps": step + 1}

    # Hit MAX_TOOL_STEPS — force a final answer with no tools available
    messages.append({"role": "user", "content": "Based on your research above, give your final answer now."})
    final = client.chat.completions.create(
        model=model, max_tokens=MAX_TOKENS, temperature=0.7, messages=messages
    )
    text = final.choices[0].message.content or ""
    return {"response": text, "tool_log": tool_log, "steps": MAX_TOOL_STEPS}

# Quick smoke test
# test = run_inference(MODELS["gpt-4o-mini"], "What is 2+2?", use_tools=False)
# print(test["response"])
"""

CELL_JUDGE = """\
# ── LLM-as-a-Judge ───────────────────────────────────────────────────────────
JUDGE_SYSTEM = (
    "You are a strict factual evaluator for a research Q&A benchmark. "
    "Each rubric criterion states the exact expected answer or source. "
    "Score 1 only if the model response precisely matches the expected value — "
    "correct number, percentage, entity, or URL. Approximate or vague answers score 0. "
    "Respond ONLY with a valid JSON array, no prose before or after."
)

def parse_rubric(rubric_json_str: str) -> list[tuple[int, dict]]:
    raw = json.loads(rubric_json_str)
    result = []
    for item in raw:
        for key, val in item.items():
            num = int(re.search(r"\\d+", key).group())
            result.append((num, val))
    return result

def judge_response(model_response: str, prompt: str, rubric_json_str: str) -> dict:
    criteria = parse_rubric(rubric_json_str)
    if not criteria:
        return {"score": 0.0, "error": "bad rubric"}

    criteria_block = "\\n\\n".join(
        f"Criterion {num}:\\n"
        f"  Check: {c['description']}\\n"
        f"  Correct answer: {c.get('justification', '')}"
        for num, c in criteria
    )

    judge_prompt = (
        f"Question asked to the model:\\n{prompt}\\n\\n"
        f"Model response:\\n{model_response}\\n\\n"
        f"Evaluate each criterion — score 1 if satisfied, 0 if not:\\n{criteria_block}\\n\\n"
        f"Return a JSON array, one object per criterion:\\n"
        f'[{{"criterion": 1, "score": 0, "reasoning": "..."}}, {{"criterion": 2, "score": 1, "reasoning": "..."}}]'
    )

    resp = client.chat.completions.create(
        model=JUDGE_MODEL, max_tokens=512, temperature=0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": judge_prompt},
        ],
    )
    raw_text = resp.choices[0].message.content.strip()

    match = re.search(r"\\[.*?\\]", raw_text, re.DOTALL)
    if not match:
        return {"score": 0.0, "error": "no JSON", "raw": raw_text}
    try:
        scores_list = json.loads(match.group())
    except Exception:
        return {"score": 0.0, "error": "invalid JSON", "raw": raw_text}

    crit_scores = {s["criterion"]: s["score"] for s in scores_list}

    total_w = sum(WEIGHT_MAP.get(c["weight"], 0.5) for _, c in criteria)
    weighted = sum(
        crit_scores.get(n, 0) * WEIGHT_MAP.get(c["weight"], 0.5)
        for n, c in criteria
    )
    final_score = weighted / total_w if total_w > 0 else 0.0

    return {
        "score": final_score,
        "criterion_scores": crit_scores,
        "details": scores_list,
        "raw": raw_text,
    }

print("Judge ready.")
"""

CELL_RUN = """\
# ── Run Eval ─────────────────────────────────────────────────────────────────
def eval_one(task: dict, model_name: str, model_id: str, sample_idx: int) -> dict:
    t0 = time.time()
    inf = run_inference(model_id, task["prompt"], use_tools=USE_WEB_SEARCH)
    latency = round(time.time() - t0, 2)
    jdg = judge_response(inf["response"], task["prompt"], task["rubric_json"])
    return {
        "task_id":         task["task_id"],
        "domain":          task["domain"],
        "model_name":      model_name,
        "model_id":        model_id,
        "sample_idx":      sample_idx,
        "response":        inf["response"][:500],
        "steps":           inf["steps"],
        "latency_s":       latency,
        "score":           jdg.get("score", 0.0),
        "criterion_scores": json.dumps(jdg.get("criterion_scores", {})),
        "judge_details":   json.dumps(jdg.get("details", [])),
    }

tasks_list = df_tasks.to_dict("records")
jobs = [
    (task, mname, mid, sidx)
    for task in tasks_list
    for mname, mid in MODELS.items()
    for sidx in range(N_SAMPLES)
]
print(f"{len(jobs)} jobs: {len(tasks_list)} tasks × {len(MODELS)} models × {N_SAMPLES} samples")

results = []
with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futs = {ex.submit(eval_one, *j): j for j in jobs}
    for i, fut in enumerate(as_completed(futs)):
        try:
            r = fut.result()
            results.append(r)
            print(f"[{i+1:3d}/{len(jobs)}] {r['task_id'][:20]}  {r['model_name']:7s} s{r['sample_idx']}  score={r['score']:.2f}  {r['latency_s']}s")
        except Exception as e:
            j = futs[fut]
            print(f"  ERROR {j[1]} s{j[3]}: {e}")

df_results = pd.DataFrame(results)
print(f"\\nDone. {len(df_results)} rows collected.")
"""

CELL_SAVE = """\
# ── Save ─────────────────────────────────────────────────────────────────────
out = RESULTS_DIR / "browsecomp_eval_results.csv"
df_results.to_csv(out, index=False)
print(f"Saved → {out}")
df_results[["task_id", "model_name", "sample_idx", "score", "latency_s", "steps"]].head(12)
"""

# Analysis cells
CELL_B_HEADER = """\
---
## Analysis
"""

CELL_B_LOAD = """\
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# Reload saved results to run the analysis standalone
# df_results = pd.read_csv(RESULTS_DIR / "browsecomp_eval_results.csv")
print(df_results.shape)
df_results.dtypes
"""

CELL_B_DIST = """\
# ── 1. Score distributions per model ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, (mname, grp) in zip(axes, df_results.groupby("model_name")):
    ax.hist(grp["score"], bins=10, range=(0, 1), edgecolor="black", alpha=0.8)
    ax.axvline(grp["score"].mean(), color="red", linestyle="--", label=f"mean={grp['score'].mean():.2f}")
    ax.set_title(mname)
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.legend()
plt.suptitle("Score Distributions by Model", fontsize=14)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "dist.png", dpi=120)
plt.show()
"""

CELL_B_VARIANCE = """\
# ── 2. Run-to-run variance per task per model ─────────────────────────────────
variance_df = (
    df_results.groupby(["task_id", "model_name"])["score"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(columns={"mean": "avg_score", "std": "std_score"})
)
variance_df["std_score"] = variance_df["std_score"].fillna(0)

task_idx = {t: i + 1 for i, t in enumerate(sorted(variance_df["task_id"].unique()))}

fig, ax = plt.subplots(figsize=(14, 4))
for mname, grp in variance_df.groupby("model_name"):
    ax.bar(
        [f"T{task_idx[r]}" for r in grp["task_id"]],
        grp["std_score"], alpha=0.7, label=mname
    )
ax.set_title("Run-to-Run Score Std Dev per Task")
ax.set_ylabel("Std Dev of Score")
ax.tick_params(axis="x", rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "variance.png", dpi=120)
plt.show()

print("\\nMean std dev by model:")
print(variance_df.groupby("model_name")["std_score"].mean())
"""

CELL_B_CI = """\
# ── 3. Confidence intervals (bootstrap) ──────────────────────────────────────
def bootstrap_ci(scores, n_boot=2000, ci=0.95):
    means = [np.mean(np.random.choice(scores, len(scores))) for _ in range(n_boot)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return np.mean(scores), lo, hi

ci_rows = []
for mname, grp in df_results.groupby("model_name"):
    mean, lo, hi = bootstrap_ci(grp["score"].values)
    ci_rows.append({"model": mname, "mean": mean, "ci_lo": lo, "ci_hi": hi})
ci_df = pd.DataFrame(ci_rows)
print(ci_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 4))
for _, row in ci_df.iterrows():
    ax.errorbar(row["model"], row["mean"],
                yerr=[[row["mean"] - row["ci_lo"]], [row["ci_hi"] - row["mean"]]],
                fmt="o", capsize=8, markersize=10)
ax.set_ylim(0, 1)
ax.set_ylabel("Mean Score")
ax.set_title("95% Bootstrap CI by Model")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "ci.png", dpi=120)
plt.show()
"""

CELL_B_AGREEMENT = """\
# ── 4. Inter-model agreement per task ────────────────────────────────────────
# Pivot to task × model average score
pivot = (
    df_results.groupby(["task_id", "model_name"])["score"]
    .mean()
    .unstack("model_name")
)
model_cols = list(pivot.columns)

# Correlation
corr = pivot.corr()
print("Score correlation between models:")
print(corr)

if len(model_cols) == 2:
    m1, m2 = model_cols
    # Agreement = tasks where both models agree (both pass or both fail, threshold=0.5)
    pivot["m1_pass"] = pivot[m1] >= 0.5
    pivot["m2_pass"] = pivot[m2] >= 0.5
    agreement = (pivot["m1_pass"] == pivot["m2_pass"]).mean()
    print(f"\\nBinary agreement (threshold 0.5): {agreement:.0%}")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pivot[m1], pivot[m2], alpha=0.7, s=80)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel(f"{m1} avg score")
    ax.set_ylabel(f"{m2} avg score")
    ax.set_title(f"Task Scores: {m1} vs {m2}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "agreement.png", dpi=120)
    plt.show()
"""

CELL_B_EXTRA = """\
# ── 5. Latency vs score trade-off ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for mname, grp in df_results.groupby("model_name"):
    ax.scatter(grp["latency_s"], grp["score"], alpha=0.6, s=60, label=mname)
ax.set_xlabel("Latency (s)")
ax.set_ylabel("Score")
ax.set_title("Latency vs Score")
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "latency_vs_score.png", dpi=120)
plt.show()

# ── 6. Score by domain ────────────────────────────────────────────────────────
domain_df = df_results.groupby(["domain", "model_name"])["score"].mean().unstack()
print("\\nAvg score by domain:")
print(domain_df)

# ── 7. Summary table ──────────────────────────────────────────────────────────
summary = df_results.groupby("model_name").agg(
    n_evals=("score", "count"),
    mean_score=("score", "mean"),
    median_score=("score", "median"),
    std_score=("score", "std"),
    pct_perfect=("score", lambda x: (x == 1.0).mean()),
    pct_zero=("score", lambda x: (x == 0.0).mean()),
    avg_latency=("latency_s", "mean"),
    avg_steps=("steps", "mean"),
)
print("\\n=== FINAL SUMMARY ===")
print(summary.T)
"""

# ── Build notebook ────────────────────────────────────────────────────────────
cells = [
    new_markdown_cell(CELL_HEADER),
    new_code_cell(CELL_SETUP),
    new_code_cell(CELL_DATA),
    new_code_cell(CELL_CONFIG),
    new_code_cell(CELL_TOOLS),
    new_code_cell(CELL_INFERENCE),
    new_code_cell(CELL_JUDGE),
    new_code_cell(CELL_RUN),
    new_code_cell(CELL_SAVE),
    new_markdown_cell(CELL_B_HEADER),
    new_code_cell(CELL_B_LOAD),
    new_code_cell(CELL_B_DIST),
    new_code_cell(CELL_B_VARIANCE),
    new_code_cell(CELL_B_CI),
    new_code_cell(CELL_B_AGREEMENT),
    new_code_cell(CELL_B_EXTRA),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {"name": "python", "version": "3.11"}

out = "browsecomp_eval.ipynb"
with open(out, "w") as f:
    nbformat.write(nb, f)

print(f"Created {out}  ({len(cells)} cells)")
