# -*- coding: utf-8 -*-
"""Every audit number of acl2023_pii_audit.tex, as LaTeX fragments and macros.

Reads the JSON written by the analysis scripts -- nothing is typed by hand:
  out_pii_baselines/pii_audit.json                      scripts/pii_audit.py
  out_pii_baselines/pii_sensitivity.json                scripts/pii_sensitivity.py
  out_pii_baselines/rare_pii_tables.json                scripts/rare_pii_tables.py
  out_pii_baselines/rare_label_analysis.json            scripts/rare_label_analysis.py
  out_pii_baselines/audit_validation/sizes.json         scripts/pii_audit_sample.py
  out_pii_baselines/audit_validation/validation_scores.json   (after the hand check; else "--")

Writes into paper/generated/:
  audit_macros.tex   \au{corpus}{field}, \sn{variant}{field}, \rt{field}
  tab_audit_corpora_body.tex, tab_audit_probes_body.tex, tab_sensitivity_body.tex,
  tab_validation_summary_body.tex, tab_validation_body.tex

    python scripts/audit_paper_tables.py
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path("out_pii_baselines")
OUT = Path("paper/generated")
NA = r"\na"
ORDER = ["privy", "pii_500k", "kiji", "gretel", "nemotron", "ai4privacy_300k", "ai4privacy_400k",
         "gretel_finance", "tab", "mapa"]
PROTOCOL = {"privy", "pii_500k", "kiji", "gretel"}
NAME = {"privy": "Privy", "pii_500k": "OpenPII", "kiji": "Kiji", "gretel": "Gretel-PII",
        "nemotron": "Nemotron-PII", "ai4privacy_300k": "AI4Privacy-300k",
        "ai4privacy_400k": "AI4Privacy-400k", "gretel_finance": "Gretel-Finance", "tab": "TAB", "mapa": "MAPA"}
# Release facts, not measurements: origin/domain from the dataset cards; languages counted from
# the data (language fields or per-language files) on 2026-09-27.
META = {"privy": ("Synthetic", "API payloads", "1"), "pii_500k": ("Synthetic", "mixed", "8"),
        "kiji": ("Synthetic", "mixed", "6"), "gretel": ("Synthetic", "business documents", "1"),
        "nemotron": ("Synthetic", "business documents", "1"),
        "ai4privacy_300k": ("Synthetic", "mixed", "6"), "ai4privacy_400k": ("Synthetic", "mixed", "6"),
        "gretel_finance": ("Synthetic", "finance documents", "7"),
        "tab": ("Human", "court judgments", "1"), "mapa": ("Human", "EU legal texts", "21")}
PROBE_NAME = {"O": "O offsets", "C": "C consistency", "U": "U unannotated", "G": "G artifacts"}


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def num(x, nd=1) -> str:
    if x is None:
        return NA
    return f"{x:,.{nd}f}".replace(",", "{,}") if isinstance(x, float) else f"{x:,}".replace(",", "{,}")


def macro(store: list, group: str, key: str, value: str) -> None:
    if value[:1] in "+-" and value[1:].replace(".", "").replace("0", "") == "":
        value = value[1:]                              # a signed zero (+0.00 / -0.00) has no sign
    if value[:1] == "-" and value[1:2].isdigit():     # a signed number used in prose: typeset a minus
        value = "$-$" + value[1:]
    store.append(rf"\expandafter\def\csname {group}@{key}\endcsname{{{value}}}")


def audit(macros: list) -> None:
    A = load(ROOT / "pii_audit.json")
    corpora, probes = [], []
    for ds in ORDER:
        r = A[ds]
        s, l, u, g = r["S_support"], r["L_leakage"], r["U_unannotated"], r.get("G_artifacts", {})
        o, b, c = r.get("O_offsets", {}), r.get("B_boundaries", {}), r["C_consistency"]
        leak_na = "na" in l
        vals = {
            "types": str(s["n_types"]), "train": num(r["n_train"]) if r["n_train"] else NA, "test": num(r["n_test"]),
            "lttwenty": str(s["types_lt_20"]), "raremin": num(s["rare_half_min_support"]),
            "raremed": num(float(s["rare_half_median_support"]), 0),
            "tmplleak": NA if leak_na else num(l["template_leak_pct"]),
            "exactdup": NA if leak_na else num(l["exact_text_dup_pct"], 2),
            "valseen": NA if leak_na else num(l["value_seen_pct"]),
            "cnm": num(c["non_majority_label_pct"]),
            "uin": num(u["in_schema_per_1k_docs"]), "uout": num(u["out_of_schema_per_1k_docs"]),
            "uhits": num(u["uncovered_hits"]),
            "slotdocs": num(g.get("docs_with_unfilled_slot_pct"), 2), "slotdocsn": num(g.get("docs_with_unfilled_slot")),
            "nullvals": num(g.get("null_values")), "markupvals": num(g.get("markup_values")),
            "badvals": num(g.get("null_value_pct", 0.0) + g.get("markup_value_pct", 0.0), 2),
            "offset": NA if (not o or "na" in o) else num(o["mismatch_pct"], 2),
            "offsetn": NA if (not o or "na" in o) else num(o["mismatch"]),
            "overlap": num(b.get("overlapping_pct"), 2),
            "iaa": num(r["IAA"]["test"]["exact_f1_mean"]) if "IAA" in r else NA,
            "iaaspan": num(r["IAA"]["test"]["span_only_f1_mean"]) if "IAA" in r else NA,
            "iaapairs": num(r["IAA"]["test"]["pairs"]) if "IAA" in r else NA,
            "iaadocs": num(r["IAA"]["test"]["docs_multi"]) if "IAA" in r else NA,
        }
        for k, (slot, n) in enumerate(list(g.get("top_slots", {}).items())[:3]):
            vals[f"slot{'abc'[k]}"] = r"\texttt{" + slot.replace("_", r"\_").replace("#", r"\#") + "}"
            vals[f"slot{'abc'[k]}n"] = num(n)
        xb = r.get("X_consequence", {}).get("b")
        if xb:
            vals["recseen"], vals["recnovel"] = num(xb["recall_seen_value"]), num(xb["recall_unseen_value"])
        for k, v in vals.items():
            macro(macros, "au", f"{ds}@{k}", v)
        origin, domain, langs = META[ds]
        mark = r"$^\dagger$" if ds in PROTOCOL else ""
        corpora.append(f"{NAME[ds]}{mark} & {origin} & {domain} & {langs} & {vals['types']} & {vals['train']} & {vals['test']} \\\\")
        probes.append(" & ".join([NAME[ds], vals["raremin"], vals["lttwenty"], vals["tmplleak"], vals["valseen"],
                                  vals["cnm"], vals["uin"], vals["uout"], vals["slotdocs"], vals["badvals"],
                                  vals["offset"], vals["overlap"]]) + r" \\")
        if ds == "gretel_finance":      # synthetic above, human-annotated below
            corpora.append(r"\midrule")
            probes.append(r"\midrule")
    # a quoted C example: a digit string annotated as a name
    top = {x["value"]: x["labels"] for x in A["gretel_finance"]["C_consistency"]["top_inconsistent"]}
    macro(macros, "au", "gretel_finance@digitname", num(top["123456789"]["NAME"]))
    # ranges over the synthetic corpora with a train split
    syn = [ds for ds in ORDER if META[ds][0] == "Synthetic"]
    seen = [A[ds]["L_leakage"]["value_seen_pct"] for ds in syn]
    leak = [A[ds]["L_leakage"]["template_leak_pct"] for ds in syn]
    macro(macros, "au", "all@valseenmin", num(min(seen)))
    macro(macros, "au", "all@valseenmax", num(max(seen)))
    macro(macros, "au", "all@tmplleakmax", num(max(leak), 1))
    (OUT / "tab_audit_corpora_body.tex").write_text("\n".join(corpora) + "\n", encoding="utf-8")
    (OUT / "tab_audit_probes_body.tex").write_text("\n".join(probes) + "\n", encoding="utf-8")


def sensitivity(macros: list) -> None:
    S = load(ROOT / "pii_sensitivity.json")
    rows = []
    label = {"V0": "V0 original", "V1-10": r"V1 support $\geq 10$", "V1-20": r"V1 support $\geq 20$",
             "V1-50": r"V1 support $\geq 50$", "V2": "V2 novel values", "V3": "V3 merged types",
             "V4": "V4 flagged regions"}
    for v, rec in S["variants"].items():
        a, b, c = (rec["arms"][x]["rare"] for x in "abc")
        ba, cb = rec["contrasts"]["b-a"]["rare"], rec["contrasts"]["c-b"]["rare"]
        for arm, x in zip("abc", (a, b, c)):
            macro(macros, "sn", f"{v}@{arm}", f"{x:.2f}")
        for key, con in (("ba", ba), ("cb", cb)):
            macro(macros, "sn", f"{v}@{key}", f"{con['diff']:+.2f}")
            macro(macros, "sn", f"{v}@{key}p", f"{con['p_sign_flip']:.3f}")
            macro(macros, "sn", f"{v}@{key}n", f"{con['n_positive']}/{con['n_pairs']}")
        macro(macros, "sn", f"{v}@shift", f"{b - S['variants']['V0']['arms']['b']['rare']:+.1f}")
        if v == "V2":
            kiji = rec["removed"]["kiji"]["a"]
            macro(macros, "sn", "V2@kijidrop", num(kiji["gold_dropped"]))
            macro(macros, "sn", "V2@kijitotal", num(kiji["gold_dropped"] + kiji["gold_kept"]))
        if v.startswith("V1"):
            dropped = sum(x["labels_total"] - x["labels_kept"] for x in rec["removed"].values())
            macro(macros, "sn", f"{v}@dropped", str(dropped))
        rows.append(f"{label[v]} & {a:.2f} & {b:.2f} & {c:.2f} & ${ba['diff']:+.2f}$ & {ba['p_sign_flip']:.3f} & "
                    f"{ba['n_positive']}/{ba['n_pairs']} & ${cb['diff']:+.2f}$ & {cb['p_sign_flip']:.3f} & "
                    f"{cb['n_positive']}/{cb['n_pairs']} \\\\")
    ba_all = [r["contrasts"]["b-a"]["rare"] for r in S["variants"].values()]
    cb_all = [r["contrasts"]["c-b"]["rare"] for r in S["variants"].values()]
    macro(macros, "sn", "all@bamin", f"{min(x['diff'] for x in ba_all):+.2f}")
    macro(macros, "sn", "all@bamax", f"{max(x['diff'] for x in ba_all):+.2f}")
    macro(macros, "sn", "all@bapmax", f"{max(x['p_sign_flip'] for x in ba_all):.3f}")
    macro(macros, "sn", "all@cbmin", f"{min(x['diff'] for x in cb_all):+.2f}")
    macro(macros, "sn", "all@cbmax", f"{max(x['diff'] for x in cb_all):+.2f}")
    macro(macros, "sn", "all@cbpmin", f"{min(x['p_sign_flip'] for x in cb_all):.3f}")
    kiji_groups = S["variants"]["V3"]["removed"]["kiji"]["merged_groups"]
    macro(macros, "sn", "V3@kijilargest", str(max(len(g.split("+")) for g in kiji_groups)))
    (OUT / "tab_sensitivity_body.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main_results(macros: list) -> None:
    R = load(ROOT / "rare_pii_tables.json")
    for key, name in (("b-a", "ba"), ("c-b", "cb"), ("c-a", "ca")):
        t = R["tests_tuned"][key]
        macro(macros, "rt", f"{name}", f"{t['rare']['diff']:+.2f}")
        macro(macros, "rt", f"{name}p", f"{t['rare']['p_sign_flip']:.3f}")
        macro(macros, "rt", f"{name}n", f"{t['rare']['n_positive']}/{t['rare']['n_pairs']}")
        macro(macros, "rt", f"{name}up", str(t["labels"]["improved"]))
        macro(macros, "rt", f"{name}down", str(t["labels"]["worse"]))
        macro(macros, "rt", f"{name}labels", str(t["labels"]["n"]))
        macro(macros, "rt", f"{name}wil", f"{t['labels']['p_wilcoxon']:.3f}")
        macro(macros, "rt", f"{name}micro", f"{t['micro']['diff']:+.2f}")
        boot = R.get("contrasts_tuned", {}).get(key, {}).get("rare")
        if boot:
            macro(macros, "rt", f"{name}cilo", f"{boot['ci_lo']:+.2f}")
            macro(macros, "rt", f"{name}cihi", f"{boot['ci_hi']:+.2f}")
    for arm in "abc":
        macro(macros, "rt", f"{arm}rare", f"{R['main']['tuned'][arm]['rare_avg']:.2f}")
    pc = R["per_corpus"]["tuned"]
    sds = {ds: statistics.stdev(v["rare"]) for ds, v in pc["b"].items()}
    micro = [statistics.mean(v["micro"]) for v in pc["b"].values()]
    macro(macros, "rt", "bmicromin", f"{min(micro):.1f}")
    macro(macros, "rt", "bmicromax", f"{max(micro):.1f}")
    macro(macros, "rt", "bsdmax", f"{max(sds.values()):.2f}")
    macro(macros, "rt", "bsdmin", f"{min(sds.values()):.2f}")
    g = pc["c"]["gretel"]["rare"]
    macro(macros, "rt", "cgretelspread", f"{max(g) - min(g):.1f}")
    # the one-label finding of PROTOCOL.md 9.3
    L = load(ROOT / "rare_label_analysis.json")
    rows = L["c-b"]["rows"]
    pin = next(r for r in rows if r["corpus"] == "gretel" and r["label"] == "PIN")
    macro(macros, "rt", "pindelta", f"{pin['delta']:+.1f}")
    by = defaultdict(list)
    for r in rows:
        if not (r["corpus"] == "gretel" and r["label"] == "PIN"):
            by[r["corpus"]].append(r["delta"])
    macro(macros, "rt", "cbnopin", f"{statistics.mean(statistics.mean(v) for v in by.values()):+.2f}")


ZS_KEY = {"privy": "privy", "pii_500k": "pii_500k", "kiji": "kiji", "gretel": "gretel",
          "nemotron_pii": "nemotron", "ai4p_300k": "ai4privacy_300k", "ai4p_400k": "ai4privacy_400k",
          "gretel_finance": "gretel_finance", "tab": "tab", "mapa": "mapa"}


def zeroshot(macros: list) -> None:
    """PROTOCOL.md 9.7: zero-shot GLiNER2 on the ten corpora (scripts/pii_zeroshot_analysis.py)."""
    Z = load(ROOT / "pii_zeroshot_analysis.json")
    if not Z:
        return
    rows = []
    for zk, ak in ZS_KEY.items():
        v = Z.get(zk)
        if not v:
            continue
        rh, rf, pr, d = v["rare_half"], v["rare_half_floor10"], v["precision"], v.get("did_by_type") or {}
        vals = {"rare": f"{rh['rare']:.1f}", "rarefloor": f"{rf['rare']:.1f}",
                "floorshift": f"{rf['rare'] - rh['rare']:+.1f}", "micro": f"{rh['micro']:.1f}",
                "prec": f"{pr['precision']:.1f}", "precclean": f"{pr['precision_without_flagged']:.1f}",
                "fpflag": f"{pr['fp_in_flagged_pct']:.1f}", "failed": str(v.get("docs_failed_excluded", 0))}
        if d.get("types"):
            vals.update({"didequal": f"{d['equal_weight']:+.1f}", "didmention": f"{d['mention_weight']:+.1f}",
                         "didpos": f"{d['types_positive']}/{d['types']}"})
            z, b = v["recall"], v["recall_trained_b"]
            vals["didpooled"] = f"{(b['seen'] - b['novel']) - (z['seen'] - z['novel']):+.1f}"
        for k, x in vals.items():
            macro(macros, "zs", f"{ak}@{k}", x)
        signed = lambda k: f"${vals[k]}$" if k in vals else NA
        rows.append(" & ".join([NAME[ak], vals["rare"], vals["rarefloor"], vals["micro"],
                                f"{vals['prec']} $\\to$ {vals['precclean']}", vals["fpflag"],
                                signed("didpooled"), signed("didequal"), vals.get("didpos", NA)]) + r" \\")
    dids = [v["did_by_type"]["equal_weight"] for v in Z.values() if (v.get("did_by_type") or {}).get("types")]
    macro(macros, "zs", "all@didn", str(len(dids)))
    macro(macros, "zs", "all@didneg", str(sum(d < 0 for d in dids)))
    macro(macros, "zs", "all@didpos", str(sum(d > 0 for d in dids)))
    pooled = [float(v["recall_trained_b"]["seen"] - v["recall_trained_b"]["novel"]
                    - (v["recall"]["seen"] - v["recall"]["novel"]))
              for v in Z.values() if (v.get("did_by_type") or {}).get("types")]
    macro(macros, "zs", "all@pooledpos", str(sum(x > 0 for x in pooled)))
    others = [v["precision"]["fp_in_flagged_pct"] for k, v in Z.items() if k != "ai4p_400k"]
    macro(macros, "zs", "all@fpflagmaxother", f"{max(others):.1f}")
    (OUT / "tab_zeroshot_body.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


OS_SYSTEMS = ["GLiNER2 Multi-v1", "GLiNER2-PII", "GLiNER multi-PII", "NVIDIA GLiNER-PII", "Gretel GLiNER"]
OS_TRAINED_ON = {"NVIDIA GLiNER-PII": "nemotron_pii", "Gretel GLiNER": "gretel"}


def offshelf(macros: list) -> None:
    """PROTOCOL.md 9.10: released systems, rare-half macro-F1 per corpus (scripts/pii_offshelf_ranking.py)."""
    R = load(ROOT / "pii_offshelf_ranking.json")
    if not R:
        return
    rows, winners = [], set()
    for zk, ak in ZS_KEY.items():
        rec = R["corpora"].get(zk)
        if not rec:
            continue
        v0 = rec["variants"]["V0"]["rare_half"]
        top = rec["variants"]["V0"]["ranking"][0]
        winners.add(top)
        cells = []
        for s in OS_SYSTEMS:
            x = v0.get(s)
            cell = NA if x is None else f"{x:.1f}"
            if s == top:
                cell = r"\textbf{" + cell + "}"
            if OS_TRAINED_ON.get(s) == zk:
                cell += r"$^\ast$"
            cells.append(cell)
        changed = [v for v, vr in rec["variants"].items() if v != "V0" and vr.get("top_changed")]
        rows.append(" & ".join([NAME[ak]] + cells + [", ".join(changed) or "--"]) + r" \\")
        if zk == "gretel_finance":
            rows.append(r"\midrule")
    for v, sm in R["summary"].items():
        macro(macros, "os", f"{v}@topchanged", str(sm["top_changed"]))
        macro(macros, "os", f"{v}@corpora", str(sm["corpora"]))
        macro(macros, "os", f"{v}@tau", f"{sm['median_tau']:.1f}")
    macro(macros, "os", "all@winners", str(len(winners)))
    for s, h in R["trained_on_own_corpus"].items():
        key = "nvidia" if s.startswith("NVIDIA") else "gretel"
        own = R["corpora"][h["own_corpus"]]["variants"]["V0"]["rare_half"]
        others = sorted((v for k, v in own.items() if k != s), reverse=True)
        elsewhere = [c["variants"]["V0"]["rare_half"][s] for k, c in R["corpora"].items()
                     if k != h["own_corpus"] and s in c["variants"]["V0"]["rare_half"]]
        macro(macros, "os", f"{key}@own", f"{own[s]:.1f}")
        macro(macros, "os", f"{key}@second", f"{others[0]:.1f}")
        macro(macros, "os", f"{key}@margin", f"{own[s] - others[0]:.1f}")
        macro(macros, "os", f"{key}@maxelse", f"{max(elsewhere):.1f}")
        macro(macros, "os", f"{key}@minelse", f"{min(elsewhere):.1f}")
        macro(macros, "os", f"{key}@lastelse", str(sum(
            c["variants"]["V0"]["ranking"][-1] == s for k, c in R["corpora"].items() if k != h["own_corpus"])))
        macro(macros, "os", f"{key}@nelse", str(len(elsewhere)))
        macro(macros, "os", f"{key}@rankown", str(h["rank_own"]))
        macro(macros, "os", f"{key}@rankelse", f"{h['median_rank_elsewhere']:g}")
        macro(macros, "os", f"{key}@n", str(h["n_systems"]))
    (OUT / "tab_offshelf_body.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def ablation_and_ai4p(macros: list) -> None:
    """PROTOCOL.md 9.9 (full-grid ablation) and 9.8 (trained runs on AI4Privacy-400k)."""
    Ab = load(ROOT / "pii_ablation_fullgrid.json")
    if Ab:
        macro(macros, "ab", "fa", f"{Ab['pooled']['f_minus_a']:+.2f}")
        macro(macros, "ab", "ba", f"{Ab['pooled']['b_minus_a']:+.2f}")
        for ds, r in Ab["corpora"].items():
            macro(macros, "ab", f"{ds}@fa", f"{r['f_minus_a']:+.2f}")
            macro(macros, "ab", f"{ds}@ba", f"{r['b_minus_a']:+.2f}")
    A4 = load(ROOT / "pii_sensitivity_ai4p.json")
    Z = load(ROOT / "pii_zeroshot_analysis.json")
    if A4:
        v = A4["variants"]
        macro(macros, "af", "ba", f"{v['V0']['b-a']['diff']:+.2f}")
        macro(macros, "af", "banpos", f"{v['V0']['b-a']['n_positive']}/{v['V0']['b-a']['n']}")
        macro(macros, "af", "banovel", f"{v['V2']['b-a']['diff']:+.2f}")
        macro(macros, "af", "vfourchange", f"{A4['v4_change']:+.2f}")
        dropped = sum(x.get("preds_dropped", 0) for x in v["V4"]["removed"]["ai4p_400k"].values())
        macro(macros, "af", "vfourdropped", str(dropped))
        macro(macros, "af", "arare", f"{v['V0']['arms']['a']['rare']:.1f}")
        macro(macros, "af", "brare", f"{v['V0']['arms']['b']['rare']:.1f}")
    if Z:
        macro(macros, "af", "zsflagfp", str(Z["ai4p_400k"]["precision"]["fp_in_flagged"]))


TM_HOMES = ["privy", "pii_500k", "kiji", "gretel", "nemotron_pii", "ai4p_300k", "ai4p_400k", "gretel_finance"]


def transfer(macros: list) -> None:
    """PROTOCOL.md 9.11: rare-half macro-F1 of single-corpus (b) models on every corpus."""
    M = load(ROOT / "pii_transfer_matrix.json")
    if not M:
        return
    homes = [h for h in TM_HOMES if f"(b) {h}" in M["home"]]
    rows = []
    for zk, ak in ZS_KEY.items():
        rec = M["corpora"].get(zk)
        if not rec:
            continue
        cells = []
        for h in homes:
            x = rec["rare_half"].get(f"(b) {h}")
            cell = NA if x is None else f"{x:.1f}"
            if h == zk:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        tc = (M.get("tab_control") or {}).get("rare_half", {})
        if tc:
            x = tc.get(zk)
            cell = NA if x is None else f"{x:.1f}"
            cells.append(r"\textbf{" + cell + "}" if zk == "tab" else cell)
        base = rec["rare_half"].get("GLiNER2 Multi-v1")
        cells.append(NA if base is None else f"{base:.1f}")
        rows.append(" & ".join([NAME[ak]] + cells) + r" \\")
        if zk == "gretel_finance":
            rows.append(r"\midrule")
    (OUT / "tab_transfer_body.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    h = M["home"]
    first = sum(r["rank_home"] == 1 for r in h.values())
    macro(macros, "tm", "first", str(first))
    macro(macros, "tm", "homes", str(len(h)))
    macro(macros, "tm", "nsystems", str(max(r["n_systems_home"] for r in h.values())))
    ranks = [r["median_rank_elsewhere"] for r in h.values()]
    macro(macros, "tm", "rankmin", f"{min(ranks):g}")
    macro(macros, "tm", "rankmax", f"{max(ranks):g}")
    gains = {k.replace("(b) ", ""): r["median_minus_base_elsewhere"] for k, r in h.items()}
    macro(macros, "tm", "gainmin", f"{min(gains.values()):+.1f}")
    macro(macros, "tm", "gainmax", f"{max(gains.values()):+.1f}")
    macro(macros, "tm", "belowbase", str(sum(g < 0 for g in gains.values())))
    for k, g in gains.items():
        macro(macros, "tm", f"{k}@gain", f"{g:+.1f}")
    exact = sum(r.get("home_reeval") == r.get("home_original") for r in h.values())
    macro(macros, "tm", "exact", str(exact))


def review(macros: list) -> None:
    """PROTOCOL.md 9.17: macros for the checks asked for by the review."""
    C = load(ROOT / "pii_review_checks.json")
    M = load(ROOT / "pii_transfer_matrix.json") or {}
    if not C:
        return
    w2 = C["W2"]["summary"]
    macro(macros, "rv", "pairs", str(w2["pairs"]))
    macro(macros, "rv", "below", str(w2["pairs_below_untrained"]))
    macro(macros, "rv", "belowzero", str(w2["below_with_more_zero_types"]))
    macro(macros, "rv", "dall", f"{w2['median_delta_rare']:+.1f}")
    macro(macros, "rv", "dboth", f"{w2['median_delta_on_both']:+.1f}")
    for k in ("unseen_recall_trained", "unseen_recall_untrained", "shared_recall_trained", "shared_recall_untrained"):
        macro(macros, "rv", k.replace("_recall_", "").replace("_", ""), f"{w2[k]:.1f}")
    if "W2_tab" in C:
        t = C["W2_tab"]["summary"]
        for k in ("unseen_recall_trained", "unseen_recall_untrained", "shared_recall_trained", "shared_recall_untrained"):
            macro(macros, "rv", "tab" + k.replace("_recall_", "").replace("_", ""), f"{t[k]:.1f}")
    tc = M.get("tab_control")
    if tc:
        rank = M["corpora"]["tab"]["ranking"] if "tab" in M.get("corpora", {}) else []
        others = sorted((v for s_, v in M["corpora"]["tab"]["rare_half"].items()), reverse=True)
        macro(macros, "rv", "tabhome", f"{tc['rare_half']['tab']:.1f}")
        macro(macros, "rv", "tabnext", f"{others[0]:.1f}" if others else NA)
        macro(macros, "rv", "tabmedrank", f"{tc['median_rank_elsewhere']:g}")
        macro(macros, "rv", "tabnsys", str(tc["n_systems_home"]))
        macro(macros, "rv", "tabbelow", str(tc["below_base_elsewhere"]))
        macro(macros, "rv", "tabnelse", str(len(tc["minus_base_elsewhere"])))
        macro(macros, "rv", "tabgain", f"{tc['median_minus_base_elsewhere']:+.1f}")
        macro(macros, "rv", "tabmapa", f"{tc['minus_base_elsewhere'].get('mapa', float('nan')):+.1f}")
        macro(macros, "rv", "tabexact", "yes" if tc.get("home_reeval") == tc.get("home_original") else "no")
    w3 = C["W3"]
    macro(macros, "rv", "adj", str(w3["n_adjacent"]))
    macro(macros, "rv", "close", str(w3["n_close_lt3"]))
    macro(macros, "rv", "closeflip", str(w3["close_flipped"]))
    macro(macros, "rv", "wide", str(w3["n_adjacent"] - w3["n_close_lt3"]))
    flipped = [a["margin"] for a in w3["adjacent"] if a["flips_under"]]
    macro(macros, "rv", "maxflip", f"{max(flipped):.1f}" if flipped else "0")
    ob = w3["objectives_by_corpus"]
    macro(macros, "rv", "bacorpus", f"{min(v['positive'] for k, v in ob.items() if k.endswith('b-a'))}")
    cbpos = sorted({v["positive"] for k, v in ob.items() if k.endswith("c-b")})
    macro(macros, "rv", "cbcorpus", f"{cbpos[0]}--{cbpos[-1]}" if len(cbpos) > 1 else str(cbpos[0]))
    macro(macros, "rv", "minp", f"{next(iter(ob.values()))['min_attainable_p']:.3f}")
    w5 = C.get("W5")
    if w5:
        macro(macros, "rv", "miclose", str(w5["n_close_lt3"]))
        macro(macros, "rv", "micloseflip", str(w5["close_flipped"]))
        wide = [a for a in w5["adjacent"] if a["flips_under"] and a["margin"] >= 3]
        macro(macros, "rv", "miwideflip", str(len(wide)))
        macro(macros, "rv", "miwidevariants", "/".join(sorted({v for a in wide for v in a["flips_under"]})) or "none")
        macro(macros, "rv", "miwidemax", f"{max((a['margin'] for a in wide), default=0):.0f}")
        macro(macros, "rv", "mitauvtwo", f"{w5['median_tau'].get('V2', float('nan')):.1f}")
        ob = w5["objectives_micro"]
        macro(macros, "rv", "mibagap", f"{ob['V0:b-a']['mean']:+.2f}")
        macro(macros, "rv", "mibapos", str(ob["V0:b-a"]["positive"]))
        same_sign = all(v["mean"] < 0 for k, v in ob.items() if k.endswith("b-a"))
        macro(macros, "rv", "mibasamesign", "every" if same_sign else "not every")
    w8 = C["W8"]
    macro(macros, "rv", "tauall", f"{w8['median_tau_all']:.2f}")
    macro(macros, "rv", "taurel", f"{w8['median_tau_released']:.2f}")
    macro(macros, "rv", "topsame", str(w8["top_released_same"]))
    macro(macros, "rv", "topchanged", str(w8["n_corpora"] - w8["top_released_same"]))
    macro(macros, "rv", "ncorp", str(w8["n_corpora"]))
    macro(macros, "rv", "homemicro", str(w8["home_first_under_micro"]))


LP_NAME = {"privy": "Privy", "pii_500k": "OpenPII", "kiji": "Kiji", "gretel": "Gretel-PII",
           "nemotron_pii": "Nemotron-PII", "ai4p_300k": "AI4Privacy-300k", "ai4p_400k": "AI4Privacy-400k",
           "gretel_finance": "Gretel-Finance", "tab": "TAB"}


def labelpara(macros: list) -> None:
    """PROTOCOL.md 9.18: renaming each type to its description at test time."""
    L = load(ROOT / "pii_labelpara.json")
    if not L or not L.get("rows"):
        return
    rows = L["rows"]
    sm = L["summary"]
    dt = [r["d_trained"] for r in rows]
    du = [r["d_untrained"] for r in rows]
    macro(macros, "lp", "dtrained", f"{sm['median_d_trained']:+.2f}")
    macro(macros, "lp", "dtrainedmin", f"{min(dt):+.1f}")
    macro(macros, "lp", "duntrained", f"{sm['median_d_untrained']:+.1f}")
    macro(macros, "lp", "duntrainedmin", f"{min(du):+.1f}")
    macro(macros, "lp", "duntrainedmax", f"{max(du):+.1f}")
    macro(macros, "lp", "didneg", str(sm["did_negative"]))
    macro(macros, "lp", "homes", str(sm["homes"]))
    macro(macros, "lp", "p", f"{sm['p_sign_flip']:.3f}")
    macro(macros, "lp", "microneg", str(sm["micro_did_negative"]))
    macro(macros, "lp", "microp", f"{sm['micro_p_sign_flip']:.3f}")
    def sg(x: float) -> str:          # signed, one decimal; a value that rounds to zero has no sign
        v = f"{x:+.1f}"
        return "0.0" if v in ("+0.0", "-0.0") else v.replace("-", "$-$")
    body = []
    for r in rows:
        body.append(" & ".join([LP_NAME.get(r["home"], r["home"]),
                                f"{r['trained']:.1f}", f"{r['trained_para']:.1f}", sg(r["d_trained"]),
                                f"{r['untrained']:.1f}", f"{r['untrained_para']:.1f}", sg(r["d_untrained"]),
                                sg(r["did"])]) + r" \\")
    (OUT / "tab_labelpara_body.tex").write_text("\n".join(body) + "\n", encoding="utf-8")


NM_SHORT = {"S0": "names", "S1": "human.", "S2": "descr."}


def naming(macros: list) -> None:
    """PROTOCOL.md 9.19: type names chosen on validation for models not trained on the corpus."""
    N = load(ROOT / "pii_naming.json")
    if not N or "test" not in N:
        return
    t = N["test"]
    for key, m in (("un", "untrained"), ("pii", "GLiNER2-PII")):
        r = t[m]
        macro(macros, "nm", f"{key}@improved", str(r["improved"]))
        macro(macros, "nm", f"{key}@n", str(r["n"]))
        macro(macros, "nm", f"{key}@gain", f"{r['median_gain']:+.1f}")
        macro(macros, "nm", f"{key}@p", f"{r['p_sign_flip']:.3f}")
        macro(macros, "nm", f"{key}@globalimproved", str(r["global_improved"]))
        macro(macros, "nm", f"{key}@globalgain", f"{r['median_global_gain']:+.1f}")
        macro(macros, "nm", f"{key}@desc", str(r["chosen_counts"]["S2"]))

    def sg(x: float) -> str:
        v = f"{x:+.1f}"
        return "0.0" if v in ("+0.0", "-0.0") else v.replace("-", "$-$")
    un = {r["corpus"]: r for r in t["untrained"]["rows"]}
    pi = {r["corpus"]: r for r in t["GLiNER2-PII"]["rows"]}
    body = []
    for ds in [r["corpus"] for r in t["untrained"]["rows"]]:
        cells = [LP_NAME.get(ds, ds)]
        for r in (un[ds], pi[ds]):
            cells += [NM_SHORT[r["chosen"]], f"{r['S0']:.1f}", f"{r['selected']:.1f}", sg(r["gain"])]
        body.append(" & ".join(cells) + r" \\")
    (OUT / "tab_naming_body.tex").write_text("\n".join(body) + "\n", encoding="utf-8")


def compute(macros: list) -> None:
    """ARR checklist C: GPU hours from the run records (scripts/pii_compute_budget.py)."""
    B = load(ROOT / "pii_compute_budget.json")
    if not B or not B.get("gpu_hours"):
        return
    g = B["gpu_hours"]
    macro(macros, "cb", "total", f"{B['total_gpu_hours']:.1f}")
    for key, name in (("aten", "A10G"), ("afour", "A100-40GB"), ("aeight", "A100-80GB"), ("h", "H200")):
        macro(macros, "cb", key, f"{g.get(name, 0.0):.1f}")
    macro(macros, "cb", "evalonly", f"{B['by_kind'].get('evaluation only', 0.0):.1f}")
    macro(macros, "cb", "smoke", f"{sum(B['smoke_gpu_hours'].values()):.1f}")
    macro(macros, "cb", "runs", str(sum(B["runs"].values())))


def validation(macros: list) -> None:
    sizes = load(ROOT / "audit_validation" / "sizes.json") or {}
    scores = load(ROOT / "audit_validation" / "validation_scores.json") or {}
    rows, pooled = [], defaultdict(lambda: {"flags": 0, "sampled": 0, "cells": 0})
    for ds in ORDER:
        for probe in "OCUG":
            key = f"{ds}/{probe}"
            if key not in sizes:
                continue
            sz, sc = sizes[key], scores.get(key)
            p = pooled[probe]
            p["flags"] += sz["flags"]; p["sampled"] += sz["sampled"]; p["cells"] += 1
            cells = [NAME[ds], probe, num(sz["flags"]), num(sz["sampled"])]
            cells += ([f"{sc['problem_pct']:.0f} [{sc['ci'][0]:.0f}, {sc['ci'][1]:.0f}]",
                       _kap(sc.get("fleiss_kappa", sc.get("kappa"))), num(sc["est_real"])]
                      if sc else [NA, NA, NA])
            rows.append(" & ".join(cells) + r" \\")
    summary = []
    for probe in "OCUG":
        if probe in pooled:
            p = pooled[probe]
            summary.append(f"{PROBE_NAME[probe]} & {p['cells']} & {num(p['flags'])} & {num(p['sampled'])} & {NA} & {NA} \\\\"
                           if not scores else
                           f"{PROBE_NAME[probe]} & {p['cells']} & {num(p['flags'])} & {num(p['sampled'])} & "
                           + _pooled_share(scores, probe) + r" \\")
    # pooled per probe, from the scorer's own all/<probe> rows (model passes, PROTOCOL 9.16)
    for probe in "OCUG":
        a = scores.get(f"all/{probe}")
        if a:
            # stratified by cell (weights = flags per cell), review W1; the raw sample share
            # is kept as @samplepct for the appendix only
            pct, ci = (a["strat_pct"], a["strat_ci"]) if "strat_pct" in a else (a["problem_pct"], a["ci"])
            macro(macros, "au", f"val@{probe}@pct", f"{pct:.0f}")
            macro(macros, "au", f"val@{probe}@lo", f"{ci[0]:.0f}")
            macro(macros, "au", f"val@{probe}@hi", f"{ci[1]:.0f}")
            macro(macros, "au", f"val@{probe}@samplepct", f"{a['problem_pct']:.0f}")
            macro(macros, "au", f"val@{probe}@kappa", _kap(a["fleiss_kappa"]))
            macro(macros, "au", f"val@{probe}@unan", f"{a['unanimous_pct']:.0f}")
            macro(macros, "au", f"val@{probe}@nmaj", num(a["n_majority"]))
            if "pii_pct" in a:
                macro(macros, "au", f"val@{probe}@piipct", f"{a.get('strat_pii_pct', a['pii_pct']):.0f}")
    # per-cell shares the text cites (corpus/probe -> u{val@<ds>@<probe>}{pct})
    for key, v in scores.items():
        if "/" in key and not key.startswith(("all/", "_")):
            ds, probe = key.split("/")
            macro(macros, "au", f"val@{ds}@{probe}@pct", f"{v['problem_pct']:.0f}")
            macro(macros, "au", f"val@{ds}@{probe}@est", num(v["est_real"]))
            if "pii_pct" in v:
                macro(macros, "au", f"val@{ds}@{probe}@piipct", f"{v['pii_pct']:.0f}")
    if any(k.startswith("all/") for k in scores):
        alls = [scores[f"all/{q}"] for q in "OCUG" if f"all/{q}" in scores]
        pk = [x for a in alls for x in a["pairwise_kappa"] if x == x]
        macro(macros, "au", "val@pairmin", _kap(min(pk)))
        macro(macros, "au", "val@pairmax", _kap(max(pk)))
        n_all = sum(a["n"] for a in alls)
        unan = sum(a["n"] * a["unanimous_pct"] / 100 for a in alls)
        macro(macros, "au", "val@unanall", f"{100 * unan / n_all:.0f}")
    total = sum(p["sampled"] for p in pooled.values())
    macro(macros, "au", "all@sampled", num(total))
    (OUT / "tab_validation_body.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (OUT / "tab_validation_summary_body.tex").write_text("\n".join(summary) + "\n", encoding="utf-8")


def _kap(x) -> str:
    """kappa with a proper minus sign; '--' when undefined (all answers identical)."""
    if x is None or x != x:
        return NA
    return f"{x:.2f}".replace("-", "$-$")


def _pooled_share(scores: dict, probe: str) -> str:
    a = scores.get(f"all/{probe}")
    if a:                                   # model passes: stratified share + Fleiss' kappa
        pct, ci = (a["strat_pct"], a["strat_ci"]) if "strat_pct" in a else (a["problem_pct"], a["ci"])
        return f"{pct:.0f} [{ci[0]:.0f}, {ci[1]:.0f}] & {_kap(a['fleiss_kappa'])}"
    cells = [v for k, v in scores.items() if k.endswith(f"/{probe}")]
    n = sum(v["n"] * v["agree_pct"] / 100 for v in cells)
    k = sum(v["n"] * v["agree_pct"] / 100 * v["problem_pct"] / 100 for v in cells)
    kap = statistics.mean(v["kappa"] for v in cells)
    return f"{100 * k / max(1, n):.0f} & {kap:.2f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    macros = ["% generated by scripts/audit_paper_tables.py -- do not edit",
              r"\newcommand{\au}[2]{\csname au@#1@#2\endcsname}",
              r"\newcommand{\sn}[2]{\csname sn@#1@#2\endcsname}",
              r"\newcommand{\rt}[1]{\csname rt@#1\endcsname}",
              r"\newcommand{\zs}[2]{\csname zs@#1@#2\endcsname}",
              r"\newcommand{\os}[2]{\csname os@#1@#2\endcsname}",
              r"\newcommand{\ab}[1]{\csname ab@#1\endcsname}",
              r"\newcommand{\af}[1]{\csname af@#1\endcsname}",
              r"\newcommand{\tm}[1]{\csname tm@#1\endcsname}",
              r"\newcommand{\rv}[1]{\csname rv@#1\endcsname}",
              r"\newcommand{\lp}[1]{\csname lp@#1\endcsname}",
              r"\newcommand{\nm}[1]{\csname nm@#1\endcsname}",
              r"\newcommand{\cb}[1]{\csname cb@#1\endcsname}"]
    audit(macros)
    sensitivity(macros)
    main_results(macros)
    validation(macros)
    zeroshot(macros)
    offshelf(macros)
    ablation_and_ai4p(macros)
    transfer(macros)
    review(macros)
    labelpara(macros)
    naming(macros)
    compute(macros)
    (OUT / "audit_macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    print(f"wrote {len(macros) - 8} macros and 7 table bodies -> {OUT}")


if __name__ == "__main__":
    main()
