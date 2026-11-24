"""Generate every result table and number used in the manuscript from the JSON outputs.

Run from ManuscriptR2/paper:  ../.venv/bin/python make_results.py
(in the released artifact: python paper/make_results.py; standard library only).
Writes generated/*.tex. Nothing in the paper's results is typed by hand.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Manuscript tree: the artifact is ../code. Released artifact: this script sits in paper/ of the artifact itself.
CODE = ROOT.parent / "code" if (ROOT.parent / "code" / "output").is_dir() else ROOT.parent
OUT = CODE / "output"
GEN = ROOT / "generated"
MACROS: dict = {}

sys.path.insert(0, str(CODE))
from experiments.common.stats import wilson  # noqa: E402  (the helper the evaluation scripts use)


def load(rel: str):
    p = OUT / rel
    return json.loads(p.read_text()) if p.exists() else None


def pct(x, d=1):
    return "--" if x is None else f"{100 * x:.{d}f}"


def macro(name: str, value) -> None:
    MACROS[name] = value


def write(name: str, text: str) -> None:
    (GEN / name).write_text(text)


# ---------------------------------------------------------------------- #
def detector_table() -> None:
    d = load("calibration/baselines.json")
    if d is None:
        return
    sets = ["InjecAgent-base", "InjecAgent-enhanced", "BIPIA e-mail", "AgentDojo tool output", "R1 in-study (280)"]
    short = ["IA-b", "IA-e", "BIPIA", "ADojo", "R1"]
    order = ["regex", "keyword", "semantic", "classifier", "deepset", "fmops", "testsavant", "nemo"]
    names = {"regex": "StruQ-like rules", "keyword": "Keyword count", "semantic": "Semantic prototypes",
             "classifier": "LLM Guard (ProtectAI)~\\cite{protectai2024deberta}",
             "deepset": "deepset DeBERTa-v3", "fmops": "fmops DistilBERT", "testsavant": "TestSavantAI tiny",
             "nemo": "NeMo rules~\\cite{rebedea2023nemo}"}
    rows = []
    for k in order:
        v = d[k]
        b, df = v["fpr_0.01"], v["default"]
        tprs = [b["tpr_mean"][s] for s in sets]
        rows.append((k, b, df, tprs))
    ntab = 4  # the paper shows the four benchmark sets; the previous submission's set is in the supplement
    best = [max(r[3][j] for r in rows) for j in range(ntab)]
    lines = []
    for k, b, df, tprs in rows:
        cells = []
        for j, t in enumerate(tprs[:ntab]):
            s = pct(t)
            cells.append(f"\\best{{{s}}}" if abs(t - best[j]) < 1e-9 and t > 0 else s)
        fpr = f"{100 * b['fpr_mean']:.1f}$\\pm${100 * b['fpr_sd']:.1f}"
        lines.append(f"{names[k]} & {fpr} & " + " & ".join(cells) + f" & {pct(df['fpr_mean'])} \\\\")
    meta = d["_meta"]
    tex = r"""\begin{table}[t]
\centering
\caption{Content detectors under one calibration budget. Thresholds are set on benign
development text only (FPR $\le$1\%); held-out benign FPR is mean$\pm$SD over 10 splits
($n_{\mathrm{test}}=@@NTEST@@$ per split); TPR (\%) on four attack sets never used for tuning
(IA-b/IA-e: InjecAgent base/enhanced; BIPIA: poisoned e-mails; ADojo: AgentDojo injected tool
outputs); over the splits the TPR standard deviation is at most @@TPRSD@@ points. For the NeMo rules the calibrated
``threshold'' is the most permissive rule-family subset within budget. Last column: benign FPR at
each tool's default setting. Best TPR per column in bold.}
\label{tab:detectors}
\scriptsize
\setlength{\tabcolsep}{1.3pt}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Detector & FPR\,\down & IA-b\,\up & IA-e\,\up & BIPIA\,\up & ADojo\,\up & FPR$_{\mathrm{def}}$\,\down \\
\midrule
@@ROWS@@
\bottomrule
\end{tabular}
\end{table}
""".replace("@@ROWS@@", "\n".join(lines)).replace("@@NTEST@@", str(meta["n_benign"] // 2)).replace(
        "@@TPRSD@@", pct(max(d[k]["fpr_0.01"]["tpr_sd"][s] for k in order for s in sets[:ntab])))
    write("tab_detectors.tex", tex)
    macro("DetTPRSDMax", pct(max(d[k]["fpr_0.01"]["tpr_sd"][s] for k in order for s in sets[:ntab])))
    macro("DetNBenign", f"{meta['n_benign']:,}".replace(",", "{,}"))
    # earlier in-study attack documents: scored for comparison with the earlier version only (not in the paper)
    rows = [f"{names[k].split('~')[0]} & {pct(d[k]['fpr_0.01']['tpr_mean'][sets[-1]])} $\\pm$ "
            f"{pct(d[k]['fpr_0.01']['tpr_sd'][sets[-1]])} & {pct(d[k]['default']['tpr_mean'][sets[-1]])} \\\\"
            for k in order]
    write("letter_prev_detectors.tex", r"""\begin{table}[h]\centering\small
\caption{Detectors scored document by document on the 280 in-study attack documents of the previous
submission: true-positive rate (\%) at the 1\% benign-FPR budget (mean $\pm$ SD over the ten
calibration splits) and at each tool's default setting.}\label{tab:letter-prev}
\begin{tabular}{@{}lrr@{}}
\toprule
Detector & TPR, 1\% budget & TPR, default \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    for k in order:
        b = d[k]["fpr_0.01"]
        key = {"regex": "Regex", "keyword": "Keyword", "semantic": "Sem", "classifier": "PAI",
               "deepset": "Deepset", "fmops": "Fmops", "testsavant": "TSav", "nemo": "Nemo"}[k]
        for s, sh in zip(sets, ["IAb", "IAe", "BIPIA", "ADojo", "Rone"]):
            macro(f"Det{key}{sh}", pct(b["tpr_mean"][s]))
            macro(f"Det{key}{sh}Five", pct(d[k]["fpr_0.05"]["tpr_mean"][s]))
        macro(f"Det{key}FPR", pct(b["fpr_mean"]))
        macro(f"Det{key}FPRdef", pct(d[k]["default"]["fpr_mean"]))
        macro(f"Det{key}EmailFPR", pct(b["fpr_cat_mean"]["email"]))


# ---------------------------------------------------------------------- #
IA_MODELS = [("qwen2.5-7b", "Qwen-7B"), ("llama3.1-8b", "Llama-8B"), ("qwen2.5-32b", "Qwen-32B$^\\ast$"),
             ("qwen2.5-72b", "Qwen-72B$^{\\ast\\ast}$")]
IA_ROWS = [
    ("no_defense", "No defense", None),
    ("S5 filter [regex]", "$\\sigma_5$ filter: rules", "S5"),
    ("S5 filter [classifier]", "$\\sigma_5$ filter: classifier", "S5"),
    ("S5 filter [regex|classifier]", "$\\sigma_5$ filter: rules $\\cup$ classifier", "S5"),
    ("S7 args-only [regex]", "$\\sigma_7$ argument filter (no provenance)", "S7"),
    ("S7 max-taint (R1 rule) [regex]", "$\\sigma_7$ max-taint (session-level)", "S7"),
    ("S7 re-execution check on every side-effecting call (no S5, no S6)", "$\\sigma_7$ re-execution check of every call", "S7"),
    ("S5 filter [regex|classifier] + re-execution check (no S6)", "$\\sigma_5$ filter $+$ re-execution check", "S7"),
    ("S6+S7 [regex|classifier] review=allow", "\\sys, review$=$allow", "RG"),
    ("S6+S7 [regex|classifier] review=verify", "\\sys, review$=$verify (default)", "RG"),
    ("S6+S7 [regex|classifier] review=block", "\\sys, review$=$block", "RG"),
    ("S6+S7 [none] review=verify", "\\sys without evidence ($\\sigma_6{+}\\sigma_7$ only)", "RG"),
]


def verify_ok(e) -> bool:
    """Verify mode and the re-execution baselines need the masked (task-neutral) re-runs; without them the
    replayed numbers would silently equal allow mode."""
    return bool((e or {}).get("_meta", {}).get("masked_available"))


def needs_masked(key: str) -> bool:
    return "verify" in key or "re-execution" in key


def injecagent_table() -> None:
    evals = {}
    for tag, _ in IA_MODELS:
        e = load(f"injecagent/{tag}/eval_fpr_0.01.json")
        if e is not None:
            evals[tag] = e
    if not evals:
        return
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{lab}}}" for tag, lab in IA_MODELS)
    cm = " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(IA_MODELS)))
    sub = " & ".join("ASR\\,\\down & CR\\,\\up" for _ in IA_MODELS)
    lines = []
    ncase = []
    for tag, _ in IA_MODELS:
        e = evals.get(tag)
        ncase.append(f"\\multicolumn{{2}}{{c}}{{{e['no_defense']['n']} / {e['no_defense']['n_succ_nodef']}}}" if e else "\\multicolumn{2}{c}{--}")
    lines.append("Cases / successful attacks without defense & " + " & ".join(ncase) + " \\\\")
    lines.append("\\midrule")
    for key, label, grp in IA_ROWS:
        cells = []
        for tag, _ in IA_MODELS:
            e = evals.get(tag)
            if e is None or key not in e or (needs_masked(key) and not verify_ok(e)):
                cells += ["--", "--"]
                continue
            r = e[key]
            asr = pct(r["asr_all"])
            cr = "--" if key == "no_defense" else pct(r["containment"])
            if key != "no_defense" and r.get("mcnemar_vs_nodef_p") is not None and r["mcnemar_vs_nodef_p"] >= 0.05:
                cr += "$^{\\dagger}$"
            cells += [asr, cr]
        row = f"{label} & " + " & ".join(cells) + " \\\\"
        if key == "S6+S7 [regex|classifier] review=verify":
            row = "\\hlrow " + row
        if key in ("S5 filter [regex]", "S7 args-only [regex]", "S6+S7 [regex|classifier] review=allow"):
            lines.append("\\midrule")
        lines.append(row)
    tex = r"""\begin{table*}[t]
\centering
\caption{InjecAgent: attack success (ASR, \%) over all cases and attributable containment (CR, \%;
Eq.~\eqref{eq:containment}) of each configuration, replayed on the same model outputs. Base and
enhanced settings pooled; detectors calibrated at 1\% benign FPR. $^\ast$Random half of the cases.
$^{\ast\ast}$Random 20\% of the cases (a subset of that half). The re-execution rows re-run every
side-effecting call without the user's task (MELON-style), with no attribution.
$^{\dagger}$Not significant against no defense (exact McNemar, $p\ge0.05$); all other differences
have $p<0.05$. 95\% intervals: Table~\ref{tab:app-ia}.}
\label{tab:injecagent}
\footnotesize
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{@{}l""" + "rr" * len(IA_MODELS) + r"""@{}}
\toprule
 & """ + head + r""" \\
""" + cm + r"""
Configuration & """ + sub + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write("tab_injecagent.tex", tex)
    for tag, lab in IA_MODELS:
        e = evals.get(tag)
        if e is None:
            continue
        m = {"qwen2.5-7b": "QSeven", "llama3.1-8b": "LEight", "qwen2.5-32b": "QThirtyTwo", "qwen2.5-72b": "QSeventyTwo"}[tag]
        macro(f"IA{m}N", e["no_defense"]["n"])
        macro(f"IA{m}NSucc", e["no_defense"]["n_succ_nodef"])
        macro(f"IA{m}ASR", pct(e["no_defense"]["asr_all"]))
        macro(f"IA{m}ASRbase", pct(e["no_defense"]["by_setting"]["base"]["asr_all"]))
        macro(f"IA{m}ASRenh", pct(e["no_defense"]["by_setting"]["enhanced"]["asr_all"]))
        for key, short in (("S5 filter [regex|classifier]", "SFive"), ("S7 args-only [regex]", "Args"),
                           ("S7 max-taint (R1 rule) [regex]", "MaxT"),
                           ("S6+S7 [regex|classifier] review=allow", "Allow"),
                           ("S6+S7 [regex|classifier] review=verify", "Verify"),
                           ("S6+S7 [regex|classifier] review=block", "Block"),
                           ("S6+S7 [none] review=verify", "NoEvVerify"),
                           ("S7 re-execution check on every side-effecting call (no S5, no S6)", "Melon"), ("S5 filter [regex|classifier] + re-execution check (no S6)", "FilterMelon")):
            if key in e and not (needs_masked(key) and not verify_ok(e)):
                macro(f"IA{m}{short}CR", pct(e[key]["containment"]))
                macro(f"IA{m}{short}ASR", pct(e[key]["asr_all"]))
        cov = e.get("_attribution_coverage", {})
        if cov.get("n"):
            macro(f"IA{m}CovEither", pct(cov["either"] / cov["n"]))
            macro(f"IA{m}CovValue", pct(cov["tainted_value"] / cov["n"]))
            macro(f"IA{m}CovAction", pct(cov["tainted_action"] / cov["n"]))
            macro(f"IA{m}CovEvidence", pct(cov["evidence_ens"] / cov["n"]))
        if "_gate_latency_ms" in e:
            macro(f"IA{m}GateMs", f"{e['_gate_latency_ms']['mean']:.2f}")
            macro(f"IA{m}GatePNinetyFive", f"{e['_gate_latency_ms']['p95']:.2f}")
        ia_extra_macros(m, e)



IA_KEYS = (("S5 filter [regex|classifier]", "SFive"), ("S7 args-only [regex]", "Args"),
           ("S7 max-taint (R1 rule) [regex]", "MaxT"),
           ("S6+S7 [regex|classifier] review=allow", "Allow"),
           ("S6+S7 [regex|classifier] review=verify", "Verify"),
           ("S6+S7 [regex|classifier] review=block", "Block"),
           ("S6+S7 [none] review=verify", "NoEvVerify"),
           ("S7 re-execution check on every side-effecting call (no S5, no S6)", "Melon"), ("S5 filter [regex|classifier] + re-execution check (no S6)", "FilterMelon"))


def ia_extra_macros(m: str, e: dict) -> None:
    """By-setting and strict containment, compromised sources, label noise, sensitivity, adaptive attack."""
    vok = verify_ok(e)
    for key, short in IA_KEYS:
        if key not in e or (needs_masked(key) and not vok):
            continue
        bs = e[key].get("by_setting", {})
        macro(f"IA{m}{short}CRbase", pct(bs.get("base", {}).get("containment")))
        macro(f"IA{m}{short}CRenh", pct(bs.get("enhanced", {}).get("containment")))
        macro(f"IA{m}{short}CRstrict", pct(e[key].get("containment_strict")))
    nd = e["no_defense"]
    macro(f"IA{m}StrictASR", pct(nd.get("asr_strict")))
    macro(f"IA{m}NSuccStrict", nd.get("n_succ_strict"))
    macro(f"IA{m}ASRvalid", pct(nd.get("asr_valid")))
    comp = e.get("_compromised", {})
    for lab, sh in (("compromised source labelled HIGH", "CompHigh"), ("compromised source, zero-trust mode", "CompZero")):
        for rv, rs in (("allow", "Allow"), ("verify", "Verify"), ("block", "Block")):
            r = comp.get(f"{lab} | review={rv}")
            if r and not (rv == "verify" and not vok):
                macro(f"IA{m}{sh}{rs}CR", pct(r["containment"]))
    ln = e.get("_label_noise", {})
    for rv, rs in (("verify", "Verify"), ("block", "Block")):
        if rv in ln and not (rv == "verify" and not vok):
            macro(f"IA{m}Noise{rs}Half", pct(ln[rv]["0.5"]["containment_mean"]))
            macro(f"IA{m}Noise{rs}One", pct(ln[rv]["1.0"]["containment_mean"]))
    sens = e.get("_sensitivity", {})
    if sens:
        fz = [v["containment"] for v in sens["fuzzy_min"].values()]
        macro(f"IA{m}FuzzyMin", pct(min(fz)))
        macro(f"IA{m}FuzzyMax", pct(max(fz)))
        al = [v["containment"] for k, v in sens["action_min"].items() if float(k) <= 1.0]
        macro(f"IA{m}AlphaMin", pct(min(al)))
        macro(f"IA{m}AlphaMax", pct(max(al)))
        macro(f"IA{m}NoActionBlockCR", pct(sens["action_min"]["1.01"]["containment"]))
        mt = [v["containment"] for v in sens["max_taint_theta_block"].values()]
        macro(f"IA{m}MaxTMin", pct(min(mt)))
        macro(f"IA{m}MaxTMax", pct(max(mt)))
    ad = e.get("_adaptive")
    if ad and ad.get("no_defense", {}).get("n"):
        macro(f"IA{m}AdN", ad["no_defense"]["n"])
        macro(f"IA{m}AdNSucc", ad["no_defense"]["n_succ_nodef"])
        macro(f"IA{m}AdASR", pct(ad["no_defense"]["asr_all"]))
        for key, short in IA_KEYS:
            if key in ad:
                macro(f"IA{m}Ad{short}CR", pct(ad[key]["containment"]))
        cov = ad.get("_attribution_coverage", {})
        if cov.get("n"):
            macro(f"IA{m}AdCovEither", pct(cov["either"] / cov["n"]))
            macro(f"IA{m}AdCovValue", pct(cov["tainted_value"] / cov["n"]))
            macro(f"IA{m}AdCovAction", pct(cov["tainted_action"] / cov["n"]))
        macro(f"IA{m}AdMaskedAvail", ad.get("_masked_available"))
    macro(f"IA{m}MaskedAvail", e.get("_meta", {}).get("masked_available"))
    ci = e.get("_clean_intervention")
    if ci and ci.get("n"):
        macro(f"IA{m}CleanN", ci["n"])
        macro(f"IA{m}CleanUndone", pct(ci["undone"] / ci["n"]))
        macro(f"IA{m}CleanUndoneLo", pct(ci["undone_ci"][0]))
        macro(f"IA{m}CleanTaintedUndone", pct(ci["undone_and_tainted"] / max(1, ci["tainted"])))
        macro(f"IA{m}CleanNotTaintedUndone", ci["undone_not_tainted"])

# ---------------------------------------------------------------------- #
AD_ROWS = [
    ("none", "No defense"),
    ("spotlight", "Spotlighting~\\cite{hines2024spotlighting}"),
    ("pi_detector", "AgentDojo detector, default threshold"),
    ("s5_filter_t05", "$\\sigma_5$ filter, classifier at 0.5"),
    ("s5_filter", "$\\sigma_5$ filter, calibrated (1\\% FPR)"),
    ("rg_maxtaint", "$\\sigma_7$ max-taint (session-level)"),
    ("rg_attr_allow", "\\sys, review$=$allow"),
    ("rg_attr_verify", "\\sys, review$=$verify (default)"),
    ("rg_attr_verify_t05", "\\sys, verify, classifier at 0.5"),
    ("rg_attr_block", "\\sys, review$=$block"),
]


def fb_entry(r: dict) -> dict:
    """FB counts benign runs in which the defense blocked (gate) or withheld (input filter) something."""
    b, w = r["benign_episodes_with_block"], r["benign_episodes_with_redaction"]
    return b if (b["rate"] or 0) >= (w["rate"] or 0) else w


def agentdojo_table(tag: str = "qwen2.5-7b") -> None:
    e = load(f"agentdojo/{tag}/eval.json")
    if e is None:
        return
    lines = []
    show_cr = (e["none"]["asr"]["k"] or 0) >= 5  # containment needs successful attacks without defense
    for key, label in AD_ROWS:
        if key not in e:
            continue
        r = e[key]
        cr = "--" if key == "none" else pct(r["containment_acted"]["rate"])
        if key != "none" and (r.get("mcnemar_asr_vs_none_p") or 0) >= 0.05:
            cr += "$^{\\dagger}$"
        ub = pct(r["benign_utility"]["rate"])
        if key != "none" and (r.get("mcnemar_benign_utility_vs_none_p") or 0) < 0.05:
            ub += "$^{\\ddagger}$"
        fb = max(r["benign_episodes_with_block"]["rate"] or 0, r["benign_episodes_with_redaction"]["rate"] or 0)
        crcell = f"{cr} & " if show_cr else ""
        row = (f"{label} & {ub} & {pct(r['attack_utility']['rate'])} & {pct(r['asr']['rate'])} & {crcell}"
               f"{pct(fb)} & {'--' if key == 'none' else format(r['verifier_calls_per_episode'], '.2f')} \\\\")
        if key == "rg_attr_verify":
            row = "\\hlrow " + row
        lines.append(row)
    n_b, n_a = e["none"]["n_benign"], e["none"]["n_attack"]
    cr_head = "CR\\,\\up & " if show_cr else ""
    cr_cap = ("attributable containment (CR; an attack counts as contained only if the defense blocked or "
              "withheld something in that run), " if show_cr else "")
    cr_note = ("$^{\\dagger}$ASR not significantly different from no defense; " if show_cr else
               "Containment is not reported because the undefended attack rarely succeeds. ")
    tex = r"""\begin{table}[t]
\centering
\caption{AgentDojo (Qwen2.5-7B, @@NB@@ benign tasks, @@NA@@ attacked runs): utility without
attack (U), utility under attack (U$_a$), ASR, @@CRCAP@@share of benign runs
in which the defense blocked or withheld something (FB), and extra model calls per run made by the
interventional check (V). @@CRNOTE@@$^{\ddagger}$benign utility significantly different from no
defense (exact McNemar, $p<0.05$). 95\% intervals: Table~\ref{tab:app-adrag}.}
\label{tab:agentdojo}
\scriptsize
\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{@{}l@@COLS@@@{}}
\toprule
Configuration & U\,\up & U$_a$\,\up & ASR\,\down & @@CRHEAD@@FB\,\down & V\,\down \\
\midrule
@@ROWS@@
\bottomrule
\end{tabular}
\end{table}
""".replace("@@ROWS@@", "\n".join(lines)).replace("@@NB@@", str(n_b)).replace("@@NA@@", str(n_a)) \
       .replace("@@CRCAP@@", cr_cap).replace("@@CRNOTE@@", cr_note).replace("@@CRHEAD@@", cr_head) \
       .replace("@@COLS@@", "r" * (6 if show_cr else 5))
    write("tab_agentdojo.tex", tex)
    for key, _ in AD_ROWS:
        if key not in e:
            continue
        k = "AD" + "".join(w.capitalize() for w in key.replace("0", "zero").replace("5", "five").split("_"))
        r = e[key]
        macro(k + "U", pct(r["benign_utility"]["rate"]))
        macro(k + "Ua", pct(r["attack_utility"]["rate"]))
        macro(k + "ASR", pct(r["asr"]["rate"]))
        macro(k + "CR", pct(r["containment_acted"]["rate"]))
        macro(k + "CRk", r["containment_acted"]["k"])
        macro(k + "CRn", r["containment_acted"]["n"])
        macro(k + "NoActK", r.get("contained_without_action", 0))
        macro(k + "ASRk", r["asr"]["k"])
        macro(k + "FB", pct(max(r["benign_episodes_with_block"]["rate"] or 0, r["benign_episodes_with_redaction"]["rate"] or 0)))
        macro(k + "V", f"{r['verifier_calls_per_episode']:.2f}")
        for nm, dct in (("CR", r["containment_acted"]), ("U", r["benign_utility"]), ("FB", fb_entry(r))):
            if dct.get("ci"):
                macro(k + nm + "lo", pct(dct["ci"][0]))
                macro(k + nm + "hi", pct(dct["ci"][1]))
    macro("ADNBenign", n_b)
    var = e.get("_rerun_variability", {})
    for fld, nm in (("attack_succ_to_fail", "SuccFail"), ("attack_fail_to_succ", "FailSucc"),
                    ("benign_utility_changed", "Ben")):
        if var.get(fld, {}).get("n"):
            macro(f"ADVar{nm}K", var[fld]["k"])
            macro(f"ADVar{nm}N", var[fld]["n"])
    macro("ADNAttack", n_a)
    if e.get("_gate_latency_ms_mean_over_episodes") is not None:
        macro("ADGateMs", f"{e['_gate_latency_ms_mean_over_episodes']:.2f}")
    if e.get("_verifier_seconds_per_call") is not None:
        macro("ADVerifierSec", f"{e['_verifier_seconds_per_call']:.1f}")
    pb = e.get("_pi_builtin_flag_rate_benign_outputs")
    if pb:
        macro("ADPIBuiltinBenignFlag", pct(pb["rate"]))
    macro("ADNoneASRk", e["none"]["asr"]["k"])
    sw = e.get("_threshold_sweep_first_intervention", {})
    if "0.5" in sw:
        f = sw["0.5"]
        macro("ADSweepFiveSFiveBen", pct(f["s5_benign_intervene"]["rate"]))
        macro("ADSweepFiveGateBen", pct(f["gate_verify_benign_intervene"]["rate"]))
        macro("ADSweepFiveSFiveAtt", pct(f["s5_attack_intervene"]["rate"]))
        macro("ADSweepFiveGateAtt", pct(f["gate_verify_attack_intervene"]["rate"]))
        ratios = [v["s5_benign_intervene"]["rate"] / v["gate_verify_benign_intervene"]["rate"]
                  for v in sw.values() if v["gate_verify_benign_intervene"]["rate"]]
        if ratios:
            macro("ADSweepRatioMin", f"{min(ratios):.0f}")
            macro("ADSweepRatioMax", f"{max(ratios):.0f}")
    ir = e.get("_injection_reached_lb")
    if ir and ir.get("n"):
        macro("ADInjReachedK", ir["k"])
        macro("ADInjReachedPct", pct(ir["rate"]))
        ag = e.get("_asr_given_reached", {})
        if ag.get("n"):
            macro("ADASRGivenReachedK", ag["k"])
    st = e.get("_attr_stats", {})
    if st.get("write_calls"):
        macro("ADAttrAttackWrite", st["write_calls"] - st.get("write_calls_benign", 0))
        macro("ADAttrAttackTainted", st["write_tainted"] - st.get("write_tainted_benign", 0))
    if st.get("write_calls_benign"):
        macro("ADAttrBenignWrite", st["write_calls_benign"])
        macro("ADAttrBenignTaintedPct", pct(st["write_tainted_benign"] / st["write_calls_benign"]))
    if st.get("write_tainted_benign"):
        macro("ADAttrBenignClearedPct", pct(1 - st["write_tainted_verified_benign"] / st["write_tainted_benign"]))
    for pol, nm in (("rg_maxtaint", "MaxTaint"), ("rg_attr_allow", "Allow"), ("rg_attr_verify", "Verify"),
                    ("rg_attr_block", "Block")):
        r = e.get("_shadow", {}).get(pol)
        if r:
            macro(f"ADShadow{nm}BenReview", pct(r["benign_any_review"]["rate"]))
            macro(f"ADShadow{nm}BenBlock", pct(r["benign_any_block"]["rate"]))
            macro(f"ADShadow{nm}AttBlockK", r["attack_success_any_block"]["k"])
            macro(f"ADShadow{nm}AttBlockN", r["attack_success_any_block"]["n"])
    for key, nm in (("_succ_with_tainted_write", "SuccTaintedWrite"), ("_succ_with_any_write", "SuccAnyWrite")):
        r = e.get(key)
        if r and r.get("n"):
            macro(f"AD{nm}K", r["k"])
            macro(f"AD{nm}N", r["n"])


# ---------------------------------------------------------------------- #
MEM_ROWS = [("none", "No defense"), ("S5@ingest", "$\\sigma_5$ scan at write (source)"),
            ("S5@read", "$\\sigma_5$ scan at read (note)"),
            ("S7 laundered review=verify", "$\\sigma_7$, note labelled trusted (laundered)"),
            ("S7 propagated review=allow", "\\sys propagated, review$=$allow"),
            ("S7 propagated review=verify", "\\sys propagated, review$=$verify"),
            ("S7 propagated review=block", "\\sys propagated, review$=$block")]


def memory_table() -> None:
    tags = [("qwen2.5-7b", "Qwen-7B"), ("qwen2.5-32b", "Qwen-32B")]
    ev = {t: load(f"memory/{t}/eval.json") for t, _ in tags}
    ev = {t: v for t, v in ev.items() if v}
    if not ev:
        return
    lines = []
    for key, label in MEM_ROWS:
        cells = []
        for t, _ in tags:
            r = ev.get(t, {}).get(key)
            if r is None:
                cells += ["--"] * 3
                continue
            cells += [pct(r["attack_base"]["rate"]), pct(r["attack_enhanced"]["rate"]), pct(r["benign_success"]["rate"])]
        row = f"{label} & " + " & ".join(cells) + " \\\\"
        if key == "S7 propagated review=verify":
            row = "\\hlrow " + row
        lines.append(row)
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{{lab}}}" for _, lab in tags)
    tex = r"""\begin{table}[t]
\centering
\caption{Cross-session memory poisoning: attack success in the later session (\%) for base (B)
and enhanced (E) injections, and completion of benign action items (Ben.). With @@MEMN@@ items per
cell, every 95\% Wilson interval spans at most @@MEMW@@ points.}
\label{tab:memory}
\scriptsize
\setlength{\tabcolsep}{2.2pt}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
 & """ + head + r""" \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Defense & B\,\down & E\,\down & Ben.\,\up & B\,\down & E\,\down & Ben.\,\up \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    cells = [ev[t][key][m] for t in ev for key, _ in MEM_ROWS if key in ev[t]
             for m in ("attack_base", "attack_enhanced", "benign_success")]
    ns = {c["n"] for c in cells}
    if len(ns) != 1:
        raise ValueError(f"memory cells have different n: {ns}")
    tex = tex.replace("@@MEMN@@", str(ns.pop())).replace(
        "@@MEMW@@", pct(max(c["ci"][1] - c["ci"][0] for c in cells)))
    write("tab_memory.tex", tex)
    for t, _ in tags:
        e = ev.get(t)
        if not e:
            continue
        m = {"qwen2.5-7b": "QSeven", "qwen2.5-32b": "QThirtyTwo"}[t]
        for key, short in (("none", "None"), ("S5@ingest", "Ingest"), ("S5@read", "Read"),
                           ("S7 laundered review=verify", "Laundered"), ("S7 propagated review=verify", "PropVerify"),
                           ("S7 propagated review=block", "PropBlock"), ("S7 propagated review=allow", "PropAllow")):
            if key in e:
                macro(f"Mem{m}{short}B", pct(e[key]["attack_base"]["rate"]))
                macro(f"Mem{m}{short}E", pct(e[key]["attack_enhanced"]["rate"]))
                macro(f"Mem{m}{short}Ben", pct(e[key]["benign_success"]["rate"]))
        d = e["_detect"]
        macro(f"Mem{m}TaintBen", pct(d["tainted_among_called_benign"]["rate"]))
        macro(f"Mem{m}AttCalls", d["tainted_among_called_attack"]["n"])
        macro(f"Mem{m}AttCallsTainted", pct(d["tainted_among_called_attack"]["rate"]))
        macro(f"Mem{m}TaintAtt", pct(d["tainted_among_called_attack"]["rate"]))
        macro(f"Mem{m}SrcFlagE", pct(d["src_flag_attack_enh"]["rate"]))
        macro(f"Mem{m}NoteFlagE", pct(d["note_flag_attack_enh"]["rate"]))
        macro(f"Mem{m}SrcFlagB", pct(d["src_flag_attack_base"]["rate"]))
        macro(f"Mem{m}NoteFlagB", pct(d["note_flag_attack_base"]["rate"]))
    for t, _ in tags:  # inter-agent message channel
        e = load(f"memory/{t}/eval_message.json")
        if not e:
            continue
        m = {"qwen2.5-7b": "QSeven", "qwen2.5-32b": "QThirtyTwo"}[t]
        for key, short in (("none", "None"), ("S5@ingest", "Ingest"), ("S5@read", "Read"),
                           ("S7 laundered review=verify", "Laundered"), ("S7 propagated review=verify", "PropVerify"),
                           ("S7 propagated review=block", "PropBlock"), ("S7 propagated review=allow", "PropAllow")):
            if key in e:
                macro(f"MemMsg{m}{short}B", pct(e[key]["attack_base"]["rate"]))
                macro(f"MemMsg{m}{short}E", pct(e[key]["attack_enhanced"]["rate"]))
                macro(f"MemMsg{m}{short}Ben", pct(e[key]["benign_success"]["rate"]))
        d = e["_detect"]
        macro(f"MemMsg{m}NoteFlagE", pct(d["note_flag_attack_enh"]["rate"]))
        macro(f"MemMsg{m}TaintBen", pct(d["tainted_among_called_benign"]["rate"]))


# ---------------------------------------------------------------------- #
RAG_ROWS = [("none", "No defense"), ("prompt[spotlight]", "Spotlighting"), ("prompt[sandwich]", "Sandwich"),
            ("S5 block-context [regex] (R1)", "$\\sigma_5$ refuse query, rules (strict mode)"),
            ("S5 redact [semantic]", "$\\sigma_5$ redact, semantic"),
            ("S5 redact [regex|classifier]", "$\\sigma_5$ redact, rules $\\cup$ classifier"),
            ("S6 answer gate", "$\\sigma_6$ answer gate"),
            ("S5 redact [regex|classifier] + S6 answer gate", "\\sys ($\\sigma_5$ redact $+$ $\\sigma_6$ gate)")]


def rag_table() -> None:
    tags = [("qwen2.5-7b", "Qwen-7B"), ("qwen2.5-32b", "Qwen-32B")]
    ev = {t: load(f"rag/{t}/eval.json") for t, _ in tags}
    ev = {t: v for t, v in ev.items() if v}
    if not ev:
        return
    lines = []
    for key, label in RAG_ROWS:
        cells = []
        for t, _ in tags:
            r = ev.get(t, {}).get(key)
            if r is None:
                cells += ["--"] * 3
                continue
            cr = "--" if key == "none" else pct(r["containment"])
            if key != "none" and r.get("mcnemar_vs_none_p") is not None and r["mcnemar_vs_none_p"] >= 0.05:
                cr += "$^{\\dagger}$"
            cells += [pct(r["asr"]), cr, pct(r["benign_correct"])]
        row = f"{label} & " + " & ".join(cells) + " \\\\"
        if key.startswith("S5 redact [regex|classifier] +"):
            row = "\\hlrow " + row
        lines.append(row)
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{{lab}}}" for _, lab in tags)
    tex = r"""\begin{table}[t]
\centering
\caption{RAG answers under BIPIA injections (500 attacked, 100 benign questions): ASR (\%),
attributable containment (CR, \%) and benign answer correctness (U, \%).
$^{\dagger}$Not significant against no defense (exact McNemar, $p\ge0.05$). 95\% intervals:
Table~\ref{tab:app-adrag}.}
\label{tab:rag}
\scriptsize
\setlength{\tabcolsep}{2.2pt}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
 & """ + head + r""" \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Defense & ASR\,\down & CR\,\up & U\,\up & ASR\,\down & CR\,\up & U\,\up \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("tab_rag.tex", tex)
    for t, _ in tags:
        e = ev.get(t)
        if not e:
            continue
        m = {"qwen2.5-7b": "QSeven", "qwen2.5-32b": "QThirtyTwo"}[t]
        macro(f"Rag{m}ASR", pct(e["none"]["asr"]))
        macro(f"Rag{m}ASRhit", pct(e["none"]["asr_hit"]))
        macro(f"Rag{m}U", pct(e["none"]["benign_correct"]))
        macro(f"Rag{m}Hit", pct(e["_hit_rate"]["attack"]))
        jv = e["_judge_validation"]
        if jv.get("n"):
            macro(f"Rag{m}JudgeN", jv["n"])
            macro(f"Rag{m}JudgeAgree", pct(jv["agreement"]))
            macro(f"Rag{m}JudgeKappa", f"{jv['kappa']:.2f}")
        macro(f"Rag{m}NSucc", round(e["none"]["asr"] * e["none"]["n_attack"]))
        macro(f"Rag{m}NAttack", e["none"]["n_attack"])
        er, ef = e.get("_evidence_recall_hit", {}), e.get("_evidence_fpr_benign", {})
        if "regex|classifier" in er:
            macro(f"Rag{m}EvRecall", pct(er["regex|classifier"]))
            macro(f"Rag{m}EvFPR", pct(ef["regex|classifier"]))
        if "semantic" in er:
            macro(f"Rag{m}EvRecallSem", pct(er["semantic"]))
        tw = e.get("_trust_warnings", {})
        if tw and t == "qwen2.5-7b":
            for lab, sh in (("default (email LOW, news MEDIUM; t*=0.5)", "Default"),
                            ("compromised: poisoned email HIGH", "Comp"), ("email MEDIUM (labels raised)", "Med"),
                            ("zero-trust", "Zero"), ("t*=0.3 (only UNTRUSTED counts)", "Tlow")):
                if lab in tw and "0.7" in tw[lab]:
                    macro(f"RagWarnRec{sh}", pct(tw[lab]["0.7"]["warning_recall_hit"]))
                    macro(f"RagWarnFPR{sh}", pct(tw[lab]["0.7"]["warning_fpr"]))
        it = e["_interventions"]["attack"]
        if it["n"]:
            macro(f"Rag{m}IntN", it["n"])
            macro(f"Rag{m}IntPoisNec", pct(it["poisoned_necessary"] / it["n"]))
            macro(f"Rag{m}IntOtherNec", pct(it["other_necessary"] / it["n"]))
            macro(f"Rag{m}IntTopPois", pct(it["ngram_top_is_poisoned"] / it["n"]))
            macro(f"Rag{m}IntTopNec", pct(it["ngram_top_is_necessary"] / it["n"]))
            if it["n_clean"]:
                macro(f"Rag{m}IntCleanFlip", pct(it["clean_replace_flips"] / it["n_clean"]))
        ib = e["_interventions"]["benign"]
        if ib["n"]:
            macro(f"Rag{m}IntBenN", ib["n"])
            macro(f"Rag{m}IntBenTargetNec", pct(ib["target_necessary"] / ib["n"]))
            macro(f"Rag{m}IntBenTopTarget", pct(ib["ngram_top_is_target"] / ib["n"]))
            macro(f"Rag{m}IntBenTopNec", pct(ib["ngram_top_is_necessary"] / ib["n"]))
        for key, short in (("S5 redact [regex|classifier]", "Redact"), ("S6 answer gate", "AnsGate"),
                           ("S5 redact [regex|classifier] + S6 answer gate", "Full"),
                           ("S5 block-context [regex] (R1)", "Refuse"), ("prompt[spotlight]", "Spot"),
                           ("prompt[sandwich]", "Sand")):
            if key in e:
                macro(f"Rag{m}{short}ASR", pct(e[key]["asr"]))
                macro(f"Rag{m}{short}CR", pct(e[key]["containment"]))
                macro(f"Rag{m}{short}U", pct(e[key]["benign_correct"]))


# ---------------------------------------------------------------------- #
def sensitivity_figure() -> None:
    """Three panels from the evaluation JSONs (coordinates written inline)."""
    import math
    panels = []
    # (a) InjecAgent: containment vs probability that an injected output is mislabelled HIGH
    series_a = []
    styles = {"verify": "solid", "block": "densely dashed"}
    colors = {"qwen2.5-7b": "okBlue", "qwen2.5-32b": "okVermil", "llama3.1-8b": "okGreen", "qwen2.5-72b": "okPurple"}
    names = {"qwen2.5-7b": "Q-7B", "qwen2.5-32b": "Q-32B", "llama3.1-8b": "L-8B", "qwen2.5-72b": "Q-72B"}
    for tag in ("qwen2.5-7b", "llama3.1-8b", "qwen2.5-32b", "qwen2.5-72b"):
        e = load(f"injecagent/{tag}/eval_fpr_0.01.json")
        if not e or "_label_noise" not in e or "verify" not in e["_label_noise"]:
            continue
        for mode in (("verify", "block") if verify_ok(e) else ("block",)):
            pts = " ".join(f"({float(k):.1f},{100 * v['containment_mean']:.1f})" for k, v in e["_label_noise"][mode].items())
            series_a.append(f"\\addplot[{colors[tag]}, {styles[mode]}, thick, mark=*, mark size=1pt] coordinates {{{pts}}};"
                            f"\\addlegendentry{{{names[tag]} {mode}}}")
    if series_a:
        panels.append(r"""\nextgroupplot[title={(a) InjecAgent: mislabelled sources}, xlabel={$P$(injected output labelled \textsc{high})},
  ylabel={CR (\%)}, ymin=0, ymax=105, legend style={font=\tiny, at={(0.02,0.02)}, anchor=south west}, legend columns=2]
""" + "\n".join(series_a))
    # (b) AgentDojo: share of runs with an intervention vs classifier threshold (first-intervention view)
    e = load("agentdojo/qwen2.5-7b/eval.json")
    if e and "_threshold_sweep_first_intervention" in e:
        sw = e["_threshold_sweep_first_intervention"]
        def pts(key):
            out = []
            for t, v in sw.items():
                x = -math.log10(max(1e-6, 1 - float(t)))
                out.append(f"({x:.2f},{100 * (v[key]['rate'] or 0):.1f})")
            return " ".join(out)
        panels.append(r"""\nextgroupplot[title={(b) AgentDojo: classifier threshold}, xlabel={$-\log_{10}(1-\theta_d)$},
  ylabel={runs with intervention (\%)}, ymin=0, ymax=105, legend style={font=\tiny, at={(0.98,0.98)}, anchor=north east}]
\addplot[okOrange, thick, mark=square*, mark size=1pt] coordinates {""" + pts("s5_benign_intervene") + r"""};\addlegendentry{$\sigma_5$ filter, benign}
\addplot[okBlue, thick, mark=*, mark size=1pt] coordinates {""" + pts("gate_verify_benign_intervene") + r"""};\addlegendentry{\sys verify, benign}
\addplot[okOrange, densely dashed, thick, mark=square*, mark size=1pt] coordinates {""" + pts("s5_attack_intervene") + r"""};\addlegendentry{$\sigma_5$ filter, attacks}
\addplot[okBlue, densely dashed, thick, mark=*, mark size=1pt] coordinates {""" + pts("gate_verify_attack_intervene") + r"""};\addlegendentry{\sys verify, attacks}""")
    # (c) RAG: provenance-warning rates vs theta_u under label variants
    e = load("rag/qwen2.5-7b/eval.json")
    if e and "_trust_warnings" in e:
        tw = e["_trust_warnings"]
        def pts(label, key):
            return " ".join(f"({float(t):.1f},{100 * v[key]:.1f})" for t, v in tw[label].items())
        d, comp, med = ("default (email LOW, news MEDIUM; t*=0.5)", "compromised: poisoned email HIGH",
                        "email MEDIUM (labels raised)")
        panels.append(r"""\nextgroupplot[title={(c) RAG: provenance warnings}, xlabel={$\theta_u$}, ylabel={warning rate (\%)},
  ymin=0, ymax=105, legend style={font=\tiny, at={(0.02,0.02)}, anchor=south west}]
\addplot[okBlue, thick, mark=*, mark size=1pt] coordinates {""" + pts(d, "warning_recall_hit") + r"""};\addlegendentry{recall, default labels}
\addplot[okBlue, densely dashed, thick, mark=*, mark size=1pt] coordinates {""" + pts(d, "warning_fpr") + r"""};\addlegendentry{benign rate, default}
\addplot[okVermil, thick, mark=triangle*, mark size=1.2pt] coordinates {""" + pts(comp, "warning_recall_hit") + r"""};\addlegendentry{recall, compromised source}
\addplot[okGreen, densely dashed, thick, mark=square*, mark size=1pt] coordinates {""" + pts(med, "warning_fpr") + r"""};\addlegendentry{benign rate, e-mail \textsc{medium}}""")
    if not panels:
        return
    tex = r"""\begin{figure*}[t]
\centering
\begin{tikzpicture}
\begin{groupplot}[group style={group size=""" + str(len(panels)) + r""" by 1, horizontal sep=1.3cm},
  width=0.345\textwidth, height=4.3cm, grid=major, grid style={gray!20},
  tick label style={font=\scriptsize}, label style={font=\scriptsize}, title style={font=\footnotesize}]
""" + "\n".join(panels) + r"""
\end{groupplot}
\end{tikzpicture}
\caption{Sensitivity. (a) Attributable containment on InjecAgent when each injected tool output is
mislabelled \textsc{high} with the given probability (verify and block modes). (b) AgentDojo,
Qwen2.5-7B: share of benign and of successfully attacked runs in which an input filter or the
\sys gate would intervene, as the classifier threshold varies (first-intervention view from the
logged decisions). (c) RAG: provenance-only warning rates as $\theta_u$ varies, for the default
labels, for a compromised source labelled \textsc{high}, and with e-mails labelled \textsc{medium}.}
\label{fig:sensitivity}
\end{figure*}
"""
    write("fig_sensitivity.tex", tex)


# ---------------------------------------------------------------------- #
def supplementary_tables() -> None:
    parts = []
    d = load("calibration/baselines.json")
    if d:
        cats = list(d["_meta"]["categories"].keys())
        sets = list(d["_meta"]["attack_sets"].keys())
        order = ["regex", "keyword", "semantic", "classifier", "deepset", "fmops", "testsavant", "nemo"]
        rows = []
        for k in order:
            v = d[k]
            for b, lab in (("fpr_0.01", "1\\%"), ("fpr_0.05", "5\\%"), ("default", "default")):
                x = v[b]
                rows.append(f"{v['label']} & {lab} & " + " & ".join(pct(x['fpr_cat_mean'].get(c, 0)) for c in cats)
                            + " & " + " & ".join(pct(x['tpr_mean'][t]) for t in sets) + " \\\\")
        head = " & ".join(c.replace("_", " ") for c in cats) + " & " + " & ".join(set_name(t) for t in sets)
        parts.append(r"""\section{Detector Calibration Details}
Held-out benign FPR (\%) per category and TPR (\%) per attack set, mean over 10 splits, at the 1\% and
5\% budgets and at each tool's default setting. The last column scores the 280 in-study attack
documents of the previous submission (Table~\ref{tab:r1map}).
\begin{table}[h]\centering\scriptsize\setlength{\tabcolsep}{2pt}
\caption{Detector calibration details.}\label{tab:s-detectors}
\resizebox{\textwidth}{!}{\begin{tabular}{@{}ll""" + "r" * (len(cats) + len(sets)) + r"""@{}}
\toprule
Detector & Budget & """ + head + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
    # InjecAgent per setting and attack type
    rows = []
    for tag, lab in IA_MODELS:
        e = load(f"injecagent/{tag}/eval_fpr_0.01.json")
        if not e:
            continue
        for key, label, _ in IA_ROWS:
            if key not in e or (needs_masked(key) and not verify_ok(e)):
                continue
            r = e[key]
            bs, ba = r.get("by_setting", {}), r.get("by_attack", {})
            cells = [pct(bs.get("base", {}).get("asr_all")), pct(bs.get("enhanced", {}).get("asr_all")),
                     pct(ba.get("dh", {}).get("asr_all")), pct(ba.get("ds", {}).get("asr_all")),
                     pct(r.get("asr_valid"))]
            rows.append(f"{lab} & {label} & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    if rows:
        parts.append(r"""\section{InjecAgent Details}
\begin{longtable}{@{}llrrrrr@{}}
\caption{InjecAgent ASR (\%) by setting and attack type, and ASR over valid outputs (ASR-valid).}\label{tab:s-injecagent}\\
\toprule
Model & Configuration & Base & Enh. & DH & DS & ASR-valid \\
\midrule
\endhead
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{longtable}
""")
    # AgentDojo per suite
    e = load("agentdojo/qwen2.5-7b/eval.json")
    if e:
        suites = sorted(e["none"]["by_suite"].keys())
        rows = []
        for key, label in AD_ROWS:
            if key not in e:
                continue
            bsu = e[key]["by_suite"]
            rows.append(f"{label} & " + " & ".join(f"{pct(bsu[su]['benign_utility']['rate'])} / {pct(bsu[su]['asr']['rate'])}"
                                                   for su in suites) + " \\\\")
        parts.append(r"""\section{AgentDojo Details}
\begin{table}[h]\centering\scriptsize
\caption{AgentDojo per suite (Qwen2.5-7B): benign utility / ASR (\%).}\label{tab:s-agentdojo}
\begin{tabular}{@{}l""" + "r" * len(suites) + r"""@{}}
\toprule
Configuration & """ + " & ".join(suites) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    # RAG per category
    rows = []
    for tag, lab in (("qwen2.5-7b", "Qwen-7B"), ("qwen2.5-32b", "Qwen-32B")):
        e = load(f"rag/{tag}/eval.json")
        if not e:
            continue
        for cat, v in e["_category_asr_none"].items():
            rows.append(f"{lab} & {cat.replace('&', chr(92) + '&')} & {v['n']} & {pct(v['asr'])} \\\\")
    if rows:
        parts.append(r"""\section{RAG Details}
\begin{longtable}{@{}llrr@{}}
\caption{RAG answer-level ASR without defense by BIPIA attack category.}\label{tab:s-rag}\\
\toprule
Model & Category & $n$ & ASR (\%) \\
\midrule
\endhead
""" + "\n".join(rows) + r"""
\bottomrule
\end{longtable}
""")
    # sensitivity grids
    rows = []
    for tag, lab in IA_MODELS:
        e = load(f"injecagent/{tag}/eval_fpr_0.01.json")
        if not e or "_sensitivity" not in e:
            continue
        sens = e["_sensitivity"]
        rows.append(f"{lab} & max-taint $\\theta_{{\\mathrm{{block}}}}$ & " + ", ".join(
            f"{k}: {pct(v['containment'])}" for k, v in sens["max_taint_theta_block"].items()) + " \\\\")
        rows.append(f"{lab} & action support $\\alpha$ (block) & " + ", ".join(
            f"{k}: {pct(v['containment'])}" for k, v in sens["action_min"].items()) + " \\\\")
        rows.append(f"{lab} & containment $\\varphi$ (block) & " + ", ".join(
            f"{k}: {pct(v['containment'])}" for k, v in sens["fuzzy_min"].items()) + " \\\\")
        rows.append(f"{lab} & classifier $\\theta_d$ ($\\sigma_5$ / \\sys allow ASR) & " + ", ".join(
            f"{k}: {pct(v['s5_asr'])}/{pct(v['s6s7_asr'])}" for k, v in sens["classifier_threshold"].items()) + " \\\\")
    if rows:
        parts.append(r"""\section{Sensitivity Grids}
\begin{longtable}{@{}lp{0.25\textwidth}p{0.55\textwidth}@{}}
\caption{InjecAgent sensitivity: containment (\%) or ASR (\%) for each parameter value.}\label{tab:s-sens}\\
\toprule
Model & Parameter & Values \\
\midrule
\endhead
""" + "\n".join(rows) + r"""
\bottomrule
\end{longtable}
""")
    # InjecAgent: strict data-stealing criterion, compromised sources, adaptive attacker
    rows = []
    for tag, lab in IA_MODELS:
        e = load(f"injecagent/{tag}/eval_fpr_0.01.json")
        if not e:
            continue
        for key, label, _ in IA_ROWS:
            if key not in e or key == "no_defense" or (needs_masked(key) and not verify_ok(e)):
                continue
            r = e[key]
            rows.append(f"{lab} & {label} & {pct(r.get('containment'))} & {pct(r.get('containment_strict'))} \\\\")
        rows.append(f"{lab} & successful attacks (lenient / strict) & {e['no_defense']['n_succ_nodef']} & "
                    f"{e['no_defense'].get('n_succ_strict')} \\\\")
        rows.append("\\midrule")
    if rows:
        parts.append(r"""\section{InjecAgent: Strict Data-Stealing Criterion}
InjecAgent counts a data-stealing attack as successful when the agent calls the e-mail tool in the
second step; the strict criterion also requires the e-mail to be addressed to an address named in the
attacker's instruction.
\begin{longtable}{@{}llrr@{}}
\caption{Attributable containment (\%) under the lenient (benchmark) and strict success criteria.}\label{tab:s-strict}\\\\
\toprule
Model & Configuration & CR lenient & CR strict \\\\
\midrule
\endhead
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{longtable}
""")
    rows = []
    for tag, lab in IA_MODELS:
        e = load(f"injecagent/{tag}/eval_fpr_0.01.json")
        if not e or "_compromised" not in e:
            continue
        for k, r in e["_compromised"].items():
            if "review=verify" in k and not verify_ok(e):
                continue
            rows.append(f"{lab} & {k.replace('|', '&').replace('review=', '')} & {pct(r['asr_all'])} & {pct(r['containment'])} \\\\")
        rows.append("\\midrule")
    if rows:
        parts.append(r"""\section{Compromised Sources and Zero-Trust Mode}
\begin{longtable}{@{}lllrr@{}}
\caption{InjecAgent with the injected tool output labelled \textsc{untrusted} (default), labelled
\textsc{high} (compromised source), and labelled \textsc{high} under zero-trust mode: ASR and attributable
containment (\%).}\label{tab:s-comp}\\\\
\toprule
Model & Label setting & Review & ASR & CR \\\\
\midrule
\endhead
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{longtable}
""")
    rows = []
    for tag, lab in IA_MODELS:
        e = load(f"injecagent/{tag}/eval_fpr_0.01.json")
        ad = (e or {}).get("_adaptive")
        if not ad or not ad.get("no_defense", {}).get("n"):
            continue
        cov = ad.get("_attribution_coverage", {})
        names = {key: label for key, label, _ in IA_ROWS}
        for k, r in ad.items():
            if k.startswith("_"):
                continue
            rows.append(f"{lab} & {names.get(k, k)} & {r['n']} & {pct(r['asr_all'])} & "
                        f"{'--' if k == 'no_defense' else pct(r['containment'])} \\\\")
        if cov.get("n"):
            rows.append(f"{lab} & tainted successful calls: value / action / either & \\multicolumn{{3}}{{r}}{{"
                        f"{pct(cov['tainted_value'] / cov['n'])} / {pct(cov['tainted_action'] / cov['n'])} / "
                        f"{pct(cov['either'] / cov['n'])}}} \\\\")
    if rows:
        parts.append(r"""\section{Attribution-Aware Attacker}
Base-setting payloads rewritten so that every value-like token is spelled with spaces between its
characters and followed by ``(remove the spaces)''.
\begin{longtable}{@{}llrrr@{}}
\caption{InjecAgent, attribution-aware attacker: cases, ASR and attributable containment (\%).}\label{tab:s-adaptive}\\\\
\toprule
Model & Configuration & $n$ & ASR & CR \\\\
\midrule
\endhead
""" + "\n".join(rows) + r"""
\bottomrule
\end{longtable}
""")
    # memory: both channels, all defenses
    rows = []
    for tag, lab in (("qwen2.5-7b", "Qwen-7B"), ("qwen2.5-32b", "Qwen-32B")):
        for fn, ch in (("eval.json", "memory"), ("eval_message.json", "message")):
            e = load(f"memory/{tag}/{fn}")
            if not e:
                continue
            for k, r in e.items():
                if k.startswith("_"):
                    continue
                k = k.replace("S5@", "$\\sigma_5$ at ").replace("S7 ", "$\\sigma_7$ ").replace("review=", "review: ")
                rows.append(f"{lab} & {ch} & {k} & {pct(r['attack_base']['rate'])} & {pct(r['attack_enhanced']['rate'])} & "
                            f"{pct(r['benign_success']['rate'])} \\\\")
            d = e["_detect"]
            rows.append(f"{lab} & {ch} & evidence on source / note (enhanced) & \\multicolumn{{3}}{{r}}{{"
                        f"{pct(d['src_flag_attack_enh']['rate'])} / {pct(d['note_flag_attack_enh']['rate'])}}} \\\\")
            rows.append(f"{lab} & {ch} & tainted among executed calls: attack / benign & \\multicolumn{{3}}{{r}}{{"
                        f"{pct(d['tainted_among_called_attack']['rate'])} / {pct(d['tainted_among_called_benign']['rate'])}}} \\\\")
            rows.append("\\midrule")
    if rows:
        parts.append(r"""\section{Memory and Message Channels}
\begin{longtable}{@{}lllrrr@{}}
\caption{Cross-session memory poisoning and the two-agent message variant: attack success in the later
session for base (B) and enhanced (E) injections and completion of benign items (Ben.), \%.}\label{tab:s-memory}\\\\
\toprule
Model & Channel & Defense & B & E & Ben. \\\\
\midrule
\endhead
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{longtable}
""")
    parts.append(interval_section())
    macro("SuppCISection", f"S{len(parts) + 1}")  # S1 is the hand-written section in supplementary.tex
    write("supp_tables.tex", "\n".join(parts) if parts else "% no supplementary tables yet\n")


def set_name(attack_set: str) -> str:
    return attack_set.split(" (")[0].replace("R1 in-study", "previous in-study")


def wilson_rate(rate, n):
    """Wilson interval for a proportion stored as a float k/n."""
    if rate is None or not n:
        return None
    k = round(rate * n)
    if abs(k - rate * n) > 1e-6:
        raise ValueError(f"rate {rate} is not a count over n={n}")
    return wilson(k, n)


def iv(rate, ci) -> str:
    """'value [lower, upper]' in percent."""
    if rate is None:
        return "--"
    return pct(rate) if not ci else f"{pct(rate)} [{pct(ci[0])}, {pct(ci[1])}]"


def interval_section() -> str:
    """95% intervals for every rate of the main result tables, and the detector SDs."""
    out = [r"""\section{95\% Intervals for the Main Tables}\label{sec:s-intervals}
Every rate of Tables~\ref{M-tab:injecagent}--\ref{M-tab:rag} of the paper with its 95\% interval, written
as value [lower, upper] in percent. InjecAgent and RAG attack rates use the percentile bootstrap
(2{,}000 resamples) stored by the evaluation scripts; AgentDojo, memory and RAG benign correctness use
Wilson intervals. Containment intervals are taken over the attacks that succeed without a defense, so
they are wide where few attacks succeed. Detector rates (Table~\ref{M-tab:detectors}) are means and
standard deviations over the ten calibration splits.
"""]
    d = load("calibration/baselines.json")
    if d:
        sets = list(d["_meta"]["attack_sets"].keys())
        rows = []
        for k in ["regex", "keyword", "semantic", "classifier", "deepset", "fmops", "testsavant", "nemo"]:
            b = d[k]["fpr_0.01"]
            cells = [f"{pct(b['fpr_mean'])} $\\pm$ {pct(b['fpr_sd'])}"] + \
                    [f"{pct(b['tpr_mean'][s])} $\\pm$ {pct(b['tpr_sd'][s])}" for s in sets]
            rows.append(f"{d[k]['label']} & " + " & ".join(cells) + r" \\")
        out.append(r"""\begin{table}[h]\centering\scriptsize\setlength{\tabcolsep}{3pt}
\caption{Detectors at the 1\% budget (Table~\ref{M-tab:detectors}): held-out benign FPR and TPR (\%),
mean $\pm$ standard deviation over the ten calibration splits.}\label{tab:s-ci-detectors}
\begin{tabular}{@{}l""" + "r" * (1 + len(sets)) + r"""@{}}
\toprule
Detector & FPR & """ + " & ".join(set_name(s) for s in sets) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    rows = []
    for tag, lab in IA_MODELS:
        e = load(f"injecagent/{tag}/eval_fpr_0.01.json")
        if not e:
            continue
        for key, label, _ in IA_ROWS:
            if key not in e or (needs_masked(key) and not verify_ok(e)):
                continue
            r = e[key]
            cr = "--" if key == "no_defense" else iv(r["containment"], r.get("containment_ci"))
            rows.append(f"{lab} & {label} & {iv(r['asr_all'], r.get('asr_all_ci'))} & {cr}" + r" \\")
        rows.append(r"\midrule")
    if rows:
        out.append(r"""\begin{longtable}{@{}llrr@{}}
\caption{InjecAgent (Table~\ref{M-tab:injecagent}): ASR and attributable containment (CR) with 95\%
bootstrap intervals (\%).}\label{tab:s-ci-injecagent}\\
\toprule
Model & Configuration & ASR & CR \\
\midrule
\endhead
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{longtable}
""")
    e = load("agentdojo/qwen2.5-7b/eval.json")
    if e:
        rows = []
        for key, label in AD_ROWS:
            if key not in e:
                continue
            r = e[key]
            f = fb_entry(r)
            cells = [iv(r["benign_utility"]["rate"], r["benign_utility"].get("ci")),
                     iv(r["attack_utility"]["rate"], r["attack_utility"].get("ci")),
                     iv(r["asr"]["rate"], r["asr"].get("ci")),
                     "--" if key == "none" else iv(r["containment_acted"]["rate"], r["containment_acted"].get("ci")),
                     iv(f["rate"], f.get("ci"))]
            rows.append(f"{label} & " + " & ".join(cells) + r" \\")
        out.append(r"""\begin{table}[h]\centering\scriptsize\setlength{\tabcolsep}{3pt}
\caption{AgentDojo (Table~\ref{M-tab:agentdojo}): rates with 95\% Wilson intervals (\%). CR is taken over
the """ + str(e["none"]["asr"]["k"]) + r""" attacks that succeed without a defense.}\label{tab:s-ci-agentdojo}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Configuration & U & U$_a$ & ASR & CR & FB \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    rows = []
    for tag, lab in (("qwen2.5-7b", "Qwen-7B"), ("qwen2.5-32b", "Qwen-32B")):
        e = load(f"memory/{tag}/eval.json")
        if not e:
            continue
        for key, label in MEM_ROWS:
            r = e.get(key)
            if r is None:
                continue
            rows.append(f"{lab} & {label} & " + " & ".join(
                iv(r[m]["rate"], r[m].get("ci")) for m in ("attack_base", "attack_enhanced", "benign_success")) + r" \\")
        rows.append(r"\midrule")
    if rows:
        out.append(r"""\begin{longtable}{@{}llrrr@{}}
\caption{Cross-session memory poisoning (Table~\ref{M-tab:memory}): attack success for base (B) and
enhanced (E) injections and benign completion (Ben.) with 95\% Wilson intervals (\%).}\label{tab:s-ci-memory}\\
\toprule
Model & Defense & B & E & Ben. \\
\midrule
\endhead
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{longtable}
""")
    rows = []
    for tag, lab in (("qwen2.5-7b", "Qwen-7B"), ("qwen2.5-32b", "Qwen-32B")):
        e = load(f"rag/{tag}/eval.json")
        if not e:
            continue
        for key, label in RAG_ROWS:
            r = e.get(key)
            if r is None:
                continue
            cr = "--" if key == "none" else iv(r["containment"], r.get("containment_ci"))
            u = iv(r["benign_correct"], wilson_rate(r["benign_correct"], r.get("n_benign", e["none"]["n_benign"])))
            rows.append(f"{lab} & {label} & {iv(r['asr'], r.get('asr_ci'))} & {cr} & {u}" + r" \\")
        rows.append(r"\midrule")
    if rows:
        out.append(r"""\begin{longtable}{@{}llrrr@{}}
\caption{RAG answers (Table~\ref{M-tab:rag}): ASR and CR with 95\% bootstrap intervals, benign answer
correctness (U) with 95\% Wilson intervals (\%).}\label{tab:s-ci-rag}\\
\toprule
Model & Defense & ASR & CR & U \\
\midrule
\endhead
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{longtable}
""")
    return "\n".join(out)


def iv_only(ci) -> str:
    """'[lower, upper]' in percent (the point estimate is in the main table)."""
    return "--" if not ci else f"[{pct(ci[0])}, {pct(ci[1])}]"


IA_STRICT_KEYS = ["no_defense", "S5 filter [regex|classifier]", "S7 max-taint (R1 rule) [regex]",
                  "S7 re-execution check on every side-effecting call (no S5, no S6)", "S5 filter [regex|classifier] + re-execution check (no S6)",
                  "S6+S7 [regex|classifier] review=allow", "S6+S7 [regex|classifier] review=verify",
                  "S6+S7 [regex|classifier] review=block", "S6+S7 [none] review=verify"]
GATE_DEFAULTS = {"action_min": "0.34", "fuzzy_min": "0.5"}  # Table tab:params


def appendix_tables() -> None:
    """Extended tables printed in the paper's appendix. The full breakdowns (per setting, attack type,
    suite, category and parameter) are still written to generated/supp_tables.tex."""
    parts = []
    names = {key: label for key, label, _ in IA_ROWS}
    evs = [(lab, load(f"injecagent/{tag}/eval_fpr_0.01.json")) for tag, lab in IA_MODELS]
    evs = [(lab, e) for lab, e in evs if e]
    if evs:
        ncol = 1 + 2 * len(evs)

        def block(title: str) -> str:
            return f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\"

        def skip(e, key: str) -> bool:
            return needs_masked(key) and not verify_ok(e)

        rows = [block("95\\% intervals for Table~\\ref{tab:injecagent} (percentile bootstrap)")]
        for key, label, _ in IA_ROWS:
            cells = []
            for _, e in evs:
                r = e.get(key)
                if r is None or skip(e, key):
                    cells += ["--", "--"]
                    continue
                cells += [iv_only(r["asr_all_ci"]), "--" if key == "no_defense" else iv_only(r["containment_ci"])]
            rows.append(f"{label} & " + " & ".join(cells) + r" \\")
        rows += [r"\midrule", block("Strict data-stealing criterion (the e-mail must reach the attacker's address)")]
        for key in IA_STRICT_KEYS:
            cells = []
            for _, e in evs:
                r = e.get(key)
                if r is None or skip(e, key):
                    cells += ["--", "--"]
                    continue
                cells += [pct(r["asr_strict"]), "--" if key == "no_defense" else pct(r["containment_strict"])]
            rows.append(f"{names[key]} & " + " & ".join(cells) + r" \\")
        for setting, title in (("compromised source labelled HIGH",
                                "Injected tool output labelled \\textup{\\textsc{high}} (compromised source)"),
                               ("compromised source, zero-trust mode", "Compromised source, zero-trust mode")):
            rows += [r"\midrule", block(title)]
            for rv in ("allow", "verify", "block"):
                cells = []
                for _, e in evs:
                    r = e.get("_compromised", {}).get(f"{setting} | review={rv}")
                    if r is None or (rv == "verify" and not verify_ok(e)):
                        cells += ["--", "--"]
                        continue
                    cells += [pct(r["asr_all"]), pct(r["containment"])]
                rows.append(f"\\sys, review$=${rv}" + (" (default)" if rv == "verify" else "") + " & "
                            + " & ".join(cells) + r" \\")
        head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{lab}}}" for lab, _ in evs)
        rules = " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(evs)))
        noverify = [lab for lab, e in evs if not verify_ok(e)]
        nv_note = (" Verify mode was not run on " + ", ".join(noverify) + " (--).") if noverify else ""
        parts.append(r"""\begin{table*}[t]
\centering
\caption{InjecAgent, extended (ASR and CR, \%). Top: 95\% percentile-bootstrap intervals (2{,}000
resamples) for every rate of Table~\ref{tab:injecagent}. Below: point estimates under the strict
data-stealing criterion, and when the injected tool output carries a \textsc{high} label (compromised
source) with and without zero-trust mode. Samples ($^\ast$, $^{\ast\ast}$) as in
Table~\ref{tab:injecagent}.""" + nv_note + r"""}
\label{tab:app-ia}
\scriptsize
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{@{}l""" + "rr" * len(evs) + r"""@{}}
\toprule
 & """ + head + r""" \\
""" + rules + r"""
Configuration & """ + " & ".join(["ASR & CR"] * len(evs)) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    # attribution-aware attacker, one column per model
    ads = [(lab, e["_adaptive"]) for lab, e in evs if e.get("_adaptive", {}).get("no_defense", {}).get("n")]
    if ads:
        ad_rows = [("S5 filter [regex|classifier]", "$\\sigma_5$ filter: rules $\\cup$ classifier"),
                   ("S6+S7 [regex|classifier] review=allow", "\\sys, review$=$allow"),
                   ("S6+S7 [regex|classifier] review=verify", "\\sys, review$=$verify (default)"),
                   ("S6+S7 [regex|classifier] review=block", "\\sys, review$=$block"),
                   ("S7 re-execution check on every side-effecting call (no S5, no S6)",
                    "$\\sigma_7$ re-execution check of every call"),
                   ("S5 filter [regex|classifier] + re-execution check (no S6)",
                    "$\\sigma_5$ filter $+$ re-execution check")]
        rows = ["Cases / successful attacks & " + " & ".join(
            f"{ad['no_defense']['n']} / {ad['no_defense']['n_succ_nodef']}" for _, ad in ads) + r" \\",
                "ASR without defense & " + " & ".join(pct(ad["no_defense"]["asr_all"]) for _, ad in ads) + r" \\",
                r"\midrule", f"\\multicolumn{{{1 + len(ads)}}}{{@{{}}l}}{{\\emph{{Attributable containment (CR)}}}} \\\\"]
        widths = []
        for key, label in ad_rows:
            cells = []
            for _, ad in ads:
                r = ad.get(key)
                cells.append("--" if r is None or r.get("containment") is None else pct(r["containment"]))
                if r and r.get("containment_ci"):
                    widths.append((r["containment_ci"][1] - r["containment_ci"][0]) / 2)
            rows.append(f"{label} & " + " & ".join(cells) + r" \\")
        rows += [r"\midrule", f"\\multicolumn{{{1 + len(ads)}}}{{@{{}}l}}{{\\emph{{Successful attacker calls attributed to the injection}}}} \\\\"]
        for fld, label in (("tainted_value", "by argument value"), ("tainted_action", "by choice of action"),
                           ("either", "by either")):
            rows.append(f"{label} & " + " & ".join(
                pct(ad["_attribution_coverage"][fld] / ad["_attribution_coverage"]["n"])
                if ad.get("_attribution_coverage", {}).get("n") else "--" for _, ad in ads) + r" \\")
        parts.append(r"""\begin{table}[t]
\centering
\caption{InjecAgent with an attribution-aware attacker (base setting; every value-like token of the payload
is spelled with spaces): attack success and attributable containment (\%). The 95\% bootstrap intervals of the
CR values are at most $\pm$""" + pct(max(widths)) + r""" points wide. Samples as in Table~\ref{tab:injecagent}.}
\label{tab:app-adaptive}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}l""" + "r" * len(ads) + r"""@{}}
\toprule
Configuration & """ + " & ".join(lab for lab, _ in ads) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    # sensitivity grids
    sens = [(lab, e["_sensitivity"]) for lab, e in evs if "_sensitivity" in e]
    if sens:
        ncol = 1 + len(sens)

        def grid(param: str, title: str, fmt, label_of) -> list:
            out = [f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\"]
            keys = list(sens[0][1][param].keys())
            vals = {lab: [fmt(s[param][k]) for k in keys] for lab, s in sens}
            if all(len(set(v)) == 1 for v in vals.values()):  # no effect anywhere on the grid: one row
                out.append(f"all values {keys[0]}--{keys[-1]} & "
                           + " & ".join(vals[lab][0] for lab, _ in sens) + r" \\")
                return out
            for i, k in enumerate(keys):
                out.append(f"{label_of(k)} & " + " & ".join(vals[lab][i] for lab, _ in sens) + r" \\")
            return out

        def lab_default(param):
            return lambda k: ("off (no action provenance)" if float(k) > 1 else
                              k + (" (default)" if k == GATE_DEFAULTS.get(param) else ""))

        rows = grid("max_taint_theta_block", "Max-taint ablation $\\theta_{\\mathrm{block}}$: CR",
                    lambda v: pct(v["containment"]), str)
        rows += [r"\midrule"] + grid("action_min", "Action support $\\alpha$, block mode: CR",
                                     lambda v: pct(v["containment"]), lab_default("action_min"))
        rows += [r"\midrule"] + grid("fuzzy_min", "4-gram containment $\\varphi$, block mode: CR",
                                     lambda v: pct(v["containment"]), lab_default("fuzzy_min"))
        rows += [r"\midrule"] + grid("classifier_threshold",
                                     "Classifier $\\theta_d$: ASR, $\\sigma_5$ filter\\,/\\,allow mode",
                                     lambda v: f"{pct(v['s5_asr'])}/{pct(v['s6s7_asr'])}", str)
        parts.append(r"""\begin{table}[t]
\centering
\caption{InjecAgent sensitivity (\%): containment (CR) of block mode as the gate parameters vary, and
ASR of the input filter and of \sys in allow mode as the classifier threshold varies.}
\label{tab:app-sens}
\scriptsize
\setlength{\tabcolsep}{2.2pt}
\begin{tabular}{@{}l""" + "r" * len(sens) + r"""@{}}
\toprule
Value & """ + " & ".join(lab for lab, _ in sens) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    # AgentDojo and RAG intervals
    ad_e = load("agentdojo/qwen2.5-7b/eval.json")
    rag_e = [(lab, load(f"rag/{tag}/eval.json")) for tag, lab in (("qwen2.5-7b", "Qwen-7B"), ("qwen2.5-32b", "Qwen-32B"))]
    rag_e = [(lab, e) for lab, e in rag_e if e]
    if ad_e and rag_e:
        arows = []
        for key, label in AD_ROWS:
            if key not in ad_e:
                continue
            r = ad_e[key]
            arows.append(f"{label} & " + " & ".join([
                iv_only(r["benign_utility"].get("ci")), iv_only(r["attack_utility"].get("ci")),
                iv_only(r["asr"].get("ci")), "--" if key == "none" else iv_only(r["containment_acted"].get("ci")),
                iv_only(fb_entry(r).get("ci"))]) + r" \\")
        rrows = []
        for key, label in RAG_ROWS:
            if not any(key in e for _, e in rag_e):
                continue
            cells = []
            for _, e in rag_e:
                r = e.get(key)
                if r is None:
                    cells += ["--"] * 3
                    continue
                n_ben = r.get("n_benign", e["none"]["n_benign"])
                cells += [iv_only(r.get("asr_ci")), "--" if key == "none" else iv_only(r.get("containment_ci")),
                          iv_only(wilson_rate(r["benign_correct"], n_ben))]
            rrows.append(f"{label} & " + " & ".join(cells) + r" \\")
        rhead = " & ".join(f"\\multicolumn{{3}}{{c}}{{{lab}}}" for lab, _ in rag_e)
        rrules = " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(rag_e)))
        parts.append(r"""\begin{table*}[t]
\centering
\caption{95\% intervals (\%) for Tables~\ref{tab:agentdojo} and~\ref{tab:rag}. (a) AgentDojo: Wilson
intervals; CR is taken over the """ + str(ad_e["none"]["asr"]["k"]) + r""" attacks that succeed without a defense.
(b) RAG answers: percentile-bootstrap intervals for ASR and CR, Wilson intervals for benign correctness (U).}
\label{tab:app-adrag}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrr@{}}
\multicolumn{6}{@{}l}{(a) AgentDojo, Qwen2.5-7B} \\
\toprule
Configuration & U & U$_a$ & ASR & CR & FB \\
\midrule
""" + "\n".join(arows) + r"""
\bottomrule
\end{tabular}

\vspace{0.6em}
\begin{tabular}{@{}l""" + "rrr" * len(rag_e) + r"""@{}}
\multicolumn{""" + str(1 + 3 * len(rag_e)) + r"""}{@{}l}{(b) RAG answers} \\
\toprule
 & """ + rhead + r""" \\
""" + rrules + r"""
Defense & """ + " & ".join(["ASR & CR & U"] * len(rag_e)) + r""" \\
\midrule
""" + "\n".join(rrows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    write("appendix_tables.tex", "\n".join(parts) if parts else "% no appendix tables yet\n")


def latency_macros() -> None:
    d = load("latency.json")
    if not d:
        return
    for k, name in (("regex", "Regex"), ("semantic", "Sem"), ("classifier", "Cls"), ("attribution_plus_gate", "Gate")):
        macro(f"Lat{name}Mean", f"{d[k]['mean']:.2f}" if d[k]['mean'] < 10 else f"{d[k]['mean']:.0f}")
        macro(f"Lat{name}PNinetyFive", f"{d[k]['p95']:.2f}" if d[k]['p95'] < 10 else f"{d[k]['p95']:.0f}")
    macro("LatChars", f"{d['chars_mean']:.0f}")


def doc_macros() -> None:
    """Page and reference counts of the last compiled manuscript."""
    import re
    import subprocess
    bbl = ROOT / "main.bbl"
    if bbl.exists():
        macro("RefCount", bbl.read_text().count("\\bibitem"))
    pdf = ROOT / "main.pdf"
    if pdf.exists():
        out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
        m = re.search(r"Pages:\s+(\d+)", out)
        if m:
            macro("PageCount", m.group(1))


def write_macros() -> None:
    lines = ["% generated by make_results.py -- do not edit"]
    for k, v in sorted(MACROS.items()):
        lines.append(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}")
    write("results_macros.tex", "\n".join(lines) + "\n")
    # macros used in the text but not (yet) produced: defined as a visible marker, listed on stdout
    import re
    used = set()
    srcs = [ROOT / "main.tex", ROOT / "supplementary.tex", *GEN.glob("*.tex"),
            *(ROOT.parent / "response").glob("*.tex"), *(ROOT.parent / "response" / "sections").glob("*.tex")]
    srcs = [f for f in srcs if f.exists()]  # the released artifact has no manuscript sources
    pat = re.compile(r"\\((?:IA|AD|Mem|Rag|Det|Lat)[A-Z][A-Za-z]*)")
    for f in srcs:
        if f.name in ("results_macros.tex", "fallback_macros.tex"):
            continue
        used |= set(pat.findall(f.read_text()))
    missing = sorted(u for u in used if u not in MACROS)
    write("fallback_macros.tex", "% macros referenced in the text without data yet (must be empty for submission)\n" +
          "".join(f"\\providecommand{{\\{k}}}{{\\textbf{{??}}\\xspace}}\n" for k in missing))
    if missing:
        print("MISSING MACROS:", " ".join(missing))


if __name__ == "__main__":
    GEN.mkdir(exist_ok=True)
    detector_table()
    injecagent_table()
    agentdojo_table()
    memory_table()
    rag_table()
    sensitivity_figure()
    supplementary_tables()
    appendix_tables()
    latency_macros()
    doc_macros()
    write_macros()
    print(f"{len(MACROS)} macros written")
