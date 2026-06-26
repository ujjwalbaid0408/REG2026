#!/usr/bin/env python3
"""Exploratory analysis of REG2026 train_CoT.json.

Goal: quantify how templated the reasoning-graph target is, per organ, so we can
exploit it for Edge-F1 / BPV / MESS. Writes summary tables + JSON artifacts to outputs/.
"""
import json, os, re, sys
from collections import Counter, defaultdict

DATA = "/group/anantm-g00/REG2026/Data/train_CoT.json"
OUT = "/group/anantm-g00/REG2026/analysis/outputs"
os.makedirs(OUT, exist_ok=True)

def canon(s):
    """Canonicalize a question/answer string the way the metric does:
    normalize capitalization, whitespace, trailing punctuation."""
    if s is None:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(" .?!,;:")
    return s

def main():
    data = json.load(open(DATA))
    n = len(data)
    print(f"# cases: {n}\n")

    # ---- per-case basic structure ----
    steps_per_case = []
    nodes_per_case = []
    edges_per_case = []
    organ_of = {}            # id -> organ answer
    has_final_report = 0
    final_report_qs = Counter()

    all_questions = Counter()      # canonical question text -> count of steps
    all_next = Counter()
    edge_counter = Counter()       # (q, nq) canonical -> count
    q_to_answers = defaultdict(Counter)   # question -> answer distribution
    organ_edges = defaultdict(Counter)    # organ -> Counter[(q,nq)]
    organ_count = Counter()
    organ_caseids = defaultdict(list)
    # find the "organ" answer per case
    ORGAN_Q = "what is the organ"
    REPORT_KEYS = ["final pathology report", "pathology report", "final report",
                   "what is the final diagnosis", "final diagnosis"]

    for case in data:
        cid = case["id"]
        cot = case.get("chain-of-thought", [])
        steps_per_case.append(len(cot))
        nodes = set()
        edges = set()
        organ = None
        for st in cot:
            q = canon(st.get("question"))
            a = st.get("answer", "")
            nq = canon(st.get("next_question"))
            all_questions[q] += 1
            if nq:
                all_next[nq] += 1
            nodes.add(q)
            if nq:
                nodes.add(nq)
                edges.add((q, nq))
                edge_counter[(q, nq)] += 1
            q_to_answers[q][a.strip()] += 1
            if q == ORGAN_Q and organ is None:
                organ = a.strip()
        organ = organ or "UNKNOWN"
        organ_of[cid] = organ
        organ_count[organ] += 1
        organ_caseids[organ].append(cid)
        for e in edges:
            organ_edges[organ][e] += 1
        nodes_per_case.append(len(nodes))
        edges_per_case.append(len(edges))

        # detect a final-report-style step (a next_question or question mentioning report)
        joined = " ".join(canon(st.get("question")) + " " + canon(st.get("next_question"))
                          for st in cot)
        if any(k in joined for k in REPORT_KEYS):
            has_final_report += 1
        # capture terminal questions (questions that are some node's next_question
        # but never themselves a question -> leaf). gather candidate report qs:
        for st in cot:
            nq = canon(st.get("next_question"))
            if any(k in nq for k in REPORT_KEYS):
                final_report_qs[nq] += 1

    def stats(lst):
        s = sorted(lst)
        m = len(s)
        return dict(min=s[0], p25=s[m//4], median=s[m//2], p75=s[3*m//4],
                    max=s[-1], mean=round(sum(s)/m, 2))

    print("== structure per case ==")
    print(" steps:", stats(steps_per_case))
    print(" nodes:", stats(nodes_per_case))
    print(" edges:", stats(edges_per_case))
    print(f" cases with a report-style step: {has_final_report}/{n}")
    print()

    print("== organ distribution (from 'What is the organ?') ==")
    for organ, c in organ_count.most_common():
        print(f"  {c:6d}  {organ}")
    print()

    print(f"== unique canonical questions: {len(all_questions)} ==")
    for q, c in all_questions.most_common(40):
        print(f"  {c:7d}  {q}")
    print()

    print(f"== total unique edges (q->nq): {len(edge_counter)} ==")
    print("  top 30 global edges:")
    for (q, nq), c in edge_counter.most_common(30):
        print(f"   {c:6d}  [{q}] -> [{nq}]")
    print()

    # ---- TEMPLATE-NESS: per organ, how concentrated are the graphs? ----
    print("== per-organ graph templating ==")
    print(f"{'organ':<22}{'cases':>7}{'uniq_edges':>12}{'edges/case':>12}{'top10_cov%':>12}")
    organ_summary = {}
    for organ, c in organ_count.most_common():
        ec = organ_edges[organ]
        total_edge_inst = sum(ec.values())
        uniq = len(ec)
        top10 = sum(cnt for _, cnt in ec.most_common(10))
        cov = round(100 * top10 / total_edge_inst, 1) if total_edge_inst else 0
        epc = round(total_edge_inst / c, 1) if c else 0
        print(f"{organ:<22}{c:>7}{uniq:>12}{epc:>12}{cov:>12}")
        organ_summary[organ] = dict(cases=c, uniq_edges=uniq, edges_per_case=epc,
                                    top10_coverage_pct=cov)
    print()

    # ---- exact-graph templating: how many distinct full edge-sets per organ ----
    print("== distinct full reasoning graphs (edge-set signatures) per organ ==")
    print(f"{'organ':<22}{'cases':>7}{'distinct_graphs':>16}{'most_common%':>14}")
    organ_graph_sig = defaultdict(Counter)
    # recompute per-case edge-set signature
    for case in data:
        cid = case["id"]
        organ = organ_of[cid]
        edges = set()
        for st in case.get("chain-of-thought", []):
            q = canon(st.get("question")); nq = canon(st.get("next_question"))
            if nq:
                edges.add((q, nq))
        sig = hash(frozenset(edges))
        organ_graph_sig[organ][sig] += 1
    for organ, c in organ_count.most_common():
        sigs = organ_graph_sig[organ]
        distinct = len(sigs)
        mc = sigs.most_common(1)[0][1] if sigs else 0
        pct = round(100*mc/c, 1) if c else 0
        print(f"{organ:<22}{c:>7}{distinct:>16}{pct:>14}")
    print()

    # ---- answer cardinality for key classification questions ----
    print("== answer vocab size for each question (how classifiable) ==")
    rows = []
    for q, ans in q_to_answers.items():
        rows.append((all_questions[q], len(ans), q))
    rows.sort(reverse=True)
    print(f"{'#steps':>8}{'#answers':>10}  question")
    for steps, nans, q in rows[:45]:
        print(f"{steps:>8}{nans:>10}  {q}")
    print()

    print("== candidate final-report next_questions ==")
    for q, c in final_report_qs.most_common(10):
        print(f"  {c:6d}  {q}")

    # ---- dump artifacts ----
    json.dump({k: v for k, v in organ_summary.items()},
              open(f"{OUT}/organ_summary.json", "w"), indent=2)
    json.dump([{"question": q, "count": c,
                "n_answers": len(q_to_answers[q]),
                "top_answers": q_to_answers[q].most_common(8)}
               for q, c in all_questions.most_common()],
              open(f"{OUT}/questions.json", "w"), indent=2)
    json.dump([{"edge": list(e), "count": c} for e, c in edge_counter.most_common()],
              open(f"{OUT}/edges_global.json", "w"), indent=2)
    # per-organ canonical question list + edges
    per_organ = {}
    for organ in organ_count:
        ec = organ_edges[organ]
        per_organ[organ] = dict(
            cases=organ_count[organ],
            edges=[{"edge": list(e), "count": c, "freq": round(c/organ_count[organ], 3)}
                   for e, c in ec.most_common()])
    json.dump(per_organ, open(f"{OUT}/per_organ_edges.json", "w"), indent=2)
    json.dump(organ_of, open(f"{OUT}/organ_of_caseid.json", "w"), indent=2)
    print(f"\n[written] artifacts -> {OUT}/")

if __name__ == "__main__":
    main()
