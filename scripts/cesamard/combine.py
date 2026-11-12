"""Combine the labelers' votes into a confidence per review and the two flags the VLM experiments need.

Group votes are combined with the Dawid-Skene model: EM estimates, without any human answer, how often each
labeler confuses one group with another, then gives every review a probability for each group. Voters are
Claude, Qwen (prompt v1.3b) and InternVL; Qwen's second prompt is left out because it mostly repeats Qwen's first
answer (95/118 identical on gold), which the model would wrongly count as independent evidence.

Per review (Claude's label stays the label; the flags say how far to trust it and what the photos can show):
  group_conf    probability of Claude's group
  certain       group_conf >= CERTAIN; otherwise alt_group = the most likely other group
  photo         visual (Claude: localized/diffuse) | not_visible | unrelated (SigLIP: the photos look unrelated)
  use           core (certain + visual) | ambiguous (visual, not certain) | text_only (no photo evidence) | excluded

  python combine.py                 # -> main_final_v1.3.json, calibration against the blind sample when it exists
"""
import json
from collections import Counter
from pathlib import Path

LAB = Path(__file__).resolve().parents[2] / "data" / "cesamard" / "labeling"
AUTO = LAB / "auto"
GROUPS = ["damage", "contamination", "manufacturing", "not_as_described", "function", "non_visual"]
VOTERS = {"claude": "main_claude_cmm_v1.3.json", "qwen": "main_qwen_v1.3b.json", "internvl": "main_internvl_v1.3b.json"}
CERTAIN = 0.90          # revisit with the blind sample (calibration table below)
UNRELATED_MAX = 0.02    # SigLIP: best photo-complaint match below this and no photo typed as product -> unrelated
SMOOTH = 1.0


def main_issue(lab):
    issues = (lab or {}).get("issues", [])
    return ([i for i in issues if i.get("main")] or issues[:1] or [{}])[0]


def dawid_skene(votes, iters=50):
    """votes: {row: {voter: class index}} -> ({row: [p_k]}, {voter: confusion [true][answered]})."""
    K = len(GROUPS)
    voters = sorted({w for v in votes.values() for w in v})
    # start from the plain vote share, Claude breaking ties
    T = {}
    for r, v in votes.items():
        t = [0.0] * K
        for w, c in v.items():
            t[c] += 1.01 if w == "claude" else 1.0
        s = sum(t)
        T[r] = [x / s for x in t]
    for _ in range(iters):
        prior = [sum(T[r][k] for r in T) + SMOOTH for k in range(K)]
        prior = [p / sum(prior) for p in prior]
        conf = {}
        for w in voters:
            m = [[SMOOTH] * K for _ in range(K)]
            for r, v in votes.items():
                if w in v:
                    for k in range(K):
                        m[k][v[w]] += T[r][k]
            conf[w] = [[x / sum(row) for x in row] for row in m]
        for r, v in votes.items():
            t = list(prior)
            for w, c in v.items():
                for k in range(K):
                    t[k] *= conf[w][k][c]
            s = sum(t)
            T[r] = [x / s for x in t]
    return T, conf


def photo_flag(vis, rel):
    if vis in ("localized", "diffuse"):
        return "visual"
    if rel and rel["max_complaint"] < UNRELATED_MAX and max(i["type"]["product"] + i["type"]["defect_closeup"] + i["type"]["food"]
                                                         for i in rel["imgs"]) < 0.5:
        return "unrelated"
    return "not_visible"


def main():
    items = {str(x["row_id"]): x for x in json.loads((LAB / "main.json").read_text(encoding="utf-8"))}
    labs = {w: json.loads((AUTO / f).read_text(encoding="utf-8"))["labels"] for w, f in VOTERS.items() if (AUTO / f).exists()}
    rel = json.loads((AUTO / "siglip_relevance.json").read_text(encoding="utf-8"))["labels"] if (AUTO / "siglip_relevance.json").exists() else {}
    auto = json.loads((LAB / "main_auto_v1.3.json").read_text(encoding="utf-8"))["labels"]
    blind_path = LAB / "blind60_labels.json"
    blind = {r: v for r, v in json.loads(blind_path.read_text(encoding="utf-8"))["labels"].items() if v.get("done")} \
        if blind_path.exists() else {}

    votes = {}
    for r in items:
        v = {}
        for w, lab in labs.items():
            g = main_issue(lab.get(r)).get("group")
            if g in GROUPS:
                v[w] = GROUPS.index(g)
        votes[r] = v
    T, conf = dawid_skene(votes)

    out, stats = {}, Counter()
    for r in sorted(items, key=int):
        c = main_issue(labs["claude"][r])
        p = T[r]
        gi = GROUPS.index(c["group"])
        alt = max((k for k in range(len(GROUPS)) if k != gi), key=lambda k: p[k])
        certain = p[gi] >= CERTAIN
        photo = photo_flag(c["vis"], rel.get(r))
        h = blind.get(r)
        use = "excluded" if h and h.get("exclude") else (
            "text_only" if photo != "visual" else "core" if certain else "ambiguous")
        out[r] = {"domain": items[r]["domain"], "duplicates": auto[r]["duplicates"], "issues": [c],
                  "marks": labs["claude"][r].get("marks", {}), "group_conf": round(p[gi], 4), "certain": certain,
                  "alt_group": None if certain else GROUPS[alt], "photo": photo, "use": use,
                  "votes": {w: GROUPS[k] for w, k in votes[r].items()},
                  "siglip_max_complaint": rel.get(r, {}).get("max_complaint"),
                  **({"human_blind": h} if h else {})}
        stats[use] += 1
        stats[f"photo={photo}"] += 1

    meta = {"guide_version": "v1.3", "voters": list(labs), "certain": CERTAIN, "unrelated_max": UNRELATED_MAX,
            "confusion": {w: {GROUPS[k]: [round(x, 3) for x in row] for k, row in enumerate(m)} for w, m in conf.items()},
            "stats": dict(stats)}
    (LAB / "main_final_v1.3.json").write_text(json.dumps({"meta": meta, "labels": out}, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    print(f"voters {list(labs)} | {dict(sorted(stats.items()))}")
    for w, m in conf.items():
        print(f"  {w:8s} estimated accuracy per true group: " +
              " ".join(f"{g[:5]} {m[k][k]:.2f}" for k, g in enumerate(GROUPS)))

    if blind:
        print("\nAgainst the blind sample (lenient: the human's second group also counts):")
        rows = [r for r, h in blind.items() if not h.get("exclude")]
        for th in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
            sel = [r for r in rows if out[r]["group_conf"] >= th]
            ok = sum(out[r]["issues"][0]["group"] in (blind[r]["group"], blind[r].get("alt_group")) for r in sel)
            print(f"  group_conf >= {th:.2f}: {len(sel):2d} reviews, Claude right {ok}/{len(sel)}")
        pc = Counter((out[r]["photo"], blind[r]["photo"]) for r in rows)
        print("  photo flag vs human:", dict(pc))


if __name__ == "__main__":
    main()
