#!/usr/bin/env python3
"""Decisive templating test for REG2026.

Q: given (organ, #1 diagnosis), how deterministic is (a) the edge-set graph and
(b) the full answer map? If high, the task reduces to: classify organ+diagnosis(+grades),
then emit a near-fixed template -> trivially maxes Edge-F1/BPV/MESS.
"""
import json, re
from collections import Counter, defaultdict

DATA = "/group/anantm-g00/REG2026/Data/train_CoT.json"

def canon(s):
    if s is None: return ""
    s = s.strip().lower(); s = re.sub(r"\s+"," ",s); return s.rstrip(" .?!,;:")

data = json.load(open(DATA))

def get_field(cot, qtext):
    for st in cot:
        if canon(st.get("question")) == qtext:
            return st.get("answer","").strip()
    return None

# group cases by (organ, dx)
graph_by_key = defaultdict(Counter)      # key -> Counter[edgeset signature]
ans_by_key   = defaultdict(lambda: defaultdict(Counter))  # key -> question -> Counter[answer]
key_count = Counter()
report_by_key = defaultdict(Counter)

for case in data:
    cot = case.get("chain-of-thought", [])
    organ = get_field(cot, "what is the organ") or "?"
    dx    = get_field(cot, "what is the #1 diagnosis") or "?"
    key = (organ, dx)
    key_count[key] += 1
    edges = frozenset((canon(s.get("question")), canon(s.get("next_question")))
                      for s in cot if canon(s.get("next_question")))
    graph_by_key[key][hash(edges)] += 1
    report = get_field(cot, "what is the final pathology report")
    if report is not None:
        report_by_key[key][report] += 1
    for s in cot:
        q = canon(s.get("question")); a = s.get("answer","").strip()
        ans_by_key[key][q][a] += 1

print(f"distinct (organ, #1 diagnosis) keys: {len(key_count)}")
print(f"keys covering >=10 cases: {sum(1 for k,c in key_count.items() if c>=10)}\n")

# how deterministic is the graph given (organ,dx)?
print("== graph determinism given (organ, #1 diagnosis), keys with >=20 cases ==")
print(f"{'cases':>6}{'graphs':>8}{'mc%':>7}  (organ | diagnosis)")
weighted_graph_purity = 0; total = 0
for key, c in key_count.most_common():
    if c < 20: continue
    sigs = graph_by_key[key]
    mc = sigs.most_common(1)[0][1]
    pct = round(100*mc/c,1)
    weighted_graph_purity += mc; total += c
    organ, dx = key
    print(f"{c:>6}{len(sigs):>8}{pct:>7}  ({organ} | {dx[:55]})")
print(f"\nWeighted graph purity (most-common graph share) over keys>=20: "
      f"{round(100*weighted_graph_purity/total,1)}% of {total} cases")

# answer determinism: for each (organ,dx) key, what fraction of (question)
# fields have a single dominant answer? Average over keys (weighted by cases)
print("\n== answer determinism given (organ, #1 diagnosis) ==")
purities = []
for key, c in key_count.most_common():
    if c < 20: continue
    qpur = []
    for q, ans in ans_by_key[key].items():
        tot = sum(ans.values()); mc = ans.most_common(1)[0][1]
        qpur.append(mc/tot)
    if qpur:
        purities.append((c, sum(qpur)/len(qpur)))
wp = sum(c*p for c,p in purities)/sum(c for c,_ in purities)
print(f"Mean per-question answer purity within (organ,dx), weighted: {round(100*wp,1)}%")
print("(i.e., once organ+diagnosis known, each reasoning answer is ~this predictable)")

# the hard part: final report uniqueness given (organ,dx)
print("\n== final-report determinism given (organ, #1 diagnosis), keys>=20 ==")
print(f"{'cases':>6}{'uniqRep':>9}{'mc%':>7}  (organ | diagnosis)")
for key, c in key_count.most_common(25):
    if c < 20: continue
    reps = report_by_key[key]
    mc = reps.most_common(1)[0][1] if reps else 0
    pct = round(100*mc/c,1)
    organ, dx = key
    print(f"{c:>6}{len(reps):>9}{pct:>7}  ({organ} | {dx[:50]})")

# overall: how concentrated are diagnoses? (label space the classifier must learn)
print("\n== #1 diagnosis label space ==")
dx_count = Counter()
for case in data:
    dx = get_field(case.get("chain-of-thought",[]), "what is the #1 diagnosis") or "?"
    dx_count[dx]+=1
print(f"distinct #1 diagnoses: {len(dx_count)}")
cum=0; tot=sum(dx_count.values())
for i,(dx,c) in enumerate(dx_count.most_common(30),1):
    cum+=c
    print(f"  {c:5d}  ({round(100*cum/tot,1):>5}% cum)  {dx[:65]}")
