# -*- coding: utf-8 -*-
"""improve_skills.py — Thí nghiệm nâng khả năng bắt skill, đo trên ĐÚNG protocol held-out cũ.

Chạy từ thư mục gốc project:  python data/improve_skills.py

Mọi cấu hình đều chấm trên 129 CV held-out (seed 42, lexicon + IDF chỉ fit trên 80% train)
để so trực tiếp với baseline 39.75 / 25.33 trong README.

Các thí nghiệm:
  DIAG — chẩn đoán: GT (block Skills thật) bị mất ở đâu?
         (không có trong danh bạ? có trong danh bạ nhưng không có trong text?
          có trong text nhưng không vào top-25? chỉ khớp khi so kiểu "chứa"?)
  E1   — chấm ĐÚNG output thật (top-25 TF-IDF + scan toàn văn danh bạ) thay vì chỉ topk_of
  E2   — + khớp GT kiểu "chứa theo biên từ" (sql ⊂ sql server tính là 1 hit) thay vì khớp đứt string
  E3   — + xếp lại thứ tự: hit scan nằm trong vùng section Skills / dòng bullet lên trước
  E4   — + quét thêm biến thể số nhiều (database ↔ databases, sql server ↔ sql servers)
  E5   — + lọc hit generic: skill xuất hiện trong >x% doc train là từ-generic, không phân biệt
  E7   — + mở rộng danh bạ bằng block Skills của 1.000 CV test ngoài (nguồn chưa từng dùng)
"""
import os
import re

import numpy as np
import pandas as pd

from extract_json import (CSV_DIR, TOP_K, STOP_SKILLS, SKILL_HDR_RE, build_lexicon,
                          dedup_overlap, fit, load_extra_lexicon, make_scan_re,
                          norm, parse_skills_block, topk_of)

SKILL_WIN = 1500  # số ký tự sau header "Skills" tính là vùng section Skills
N_TR_DOCS = 0     # điền lúc chạy (số doc train, để tính tỉ lệ df)

BULLET_RE = re.compile(
    r"^\s*(?:[-*•▪◦‣·\u2022\u2023\u25aa\u25cf\uf0a7\uf0b1\uf0b7\uf0d8]|\d{1,2}[.)]|[A-Z]\.)")

WB = r"(?<![a-z0-9]){}(?![a-z0-9])"


def wb_search(needle: str, haystack: str):
    return re.search(WB.format(re.escape(needle)), haystack)


def containment_hit(a: str, b: str) -> bool:
    """a khớp b nếu một cái chứa cái kia theo biên từ ("sql" ⊂ "sql server" → hit)."""
    if a == b:
        return True
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return wb_search(short, long_) is not None


def make_plural_re(lex_set: set):
    """Regex quét biến thể số nhiều: token cuối ≥4 chữ được phép thêm 's'."""
    pats = set()
    for t in lex_set:
        toks = t.split()
        if not toks:
            continue
        last = toks[-1]
        if len(last) >= 4 and not last.endswith("s"):
            parts = [re.escape(x) for x in toks[:-1]] + [re.escape(last) + "s?"]
            pats.add(" ".join(parts))
    if not pats:
        return None
    alts = sorted(pats, key=len, reverse=True)
    return re.compile("(?<![a-z0-9])(?:" + "|".join(alts) + ")(?![a-z0-9])")


def skill_windows(text: str):
    return [(m.end(), m.end() + SKILL_WIN) for m in SKILL_HDR_RE.finditer(text)]


def hit_tier(p: int, windows, text: str) -> int:
    """1 = hit nằm trong 1 vùng section Skills hoặc trên dòng bullet; 2 = còn lại."""
    for a, b in windows:
        if a <= p < b:
            return 1
    ls = text.rfind("\n", 0, p) + 1
    le = text.find("\n", p)
    line = text[ls: le if le != -1 else len(text)]
    if BULLET_RE.match(line):
        return 1
    return 2


def extract_merged(ntext, text, top_terms, scan_re, lex_set, plural_re=None,
                   df_tab=None, df_cap=None, tier=False):
    """Output thật của pipeline: top TF-IDF + scan danh bạ (có các option thí nghiệm)."""
    pos = {}
    for m in scan_re.finditer(ntext):
        w = norm(m.group(0))
        if w not in pos and w not in STOP_SKILLS:
            pos[w] = m.start()
    if plural_re is not None:
        for m in plural_re.finditer(ntext):
            w = norm(m.group(0))
            if w.endswith("s") and w[:-1] in lex_set:
                w = w[:-1]  # "databases" → "database" nếu "database" có trong danh bạ
            if w not in pos and w not in STOP_SKILLS:
                pos[w] = m.start()
    extras = set(pos) - set(top_terms)
    if df_cap is not None:
        extras = {w for w in extras if df_tab.get(w, 0) / N_TR_DOCS <= df_cap}
    if tier:
        wins = skill_windows(text)
        t1 = sorted((w for w in extras if hit_tier(pos[w], wins, text) == 1), key=len)
        s1 = set(t1)
        t2 = sorted((w for w in extras if w not in s1), key=len)
        merged = dedup_overlap(list(top_terms) + t1 + t2)
    else:
        merged = dedup_overlap(list(top_terms) + sorted(extras, key=len))
    return merged, pos


def main() -> None:
    global N_TR_DOCS
    cv = pd.read_csv(os.path.join(CSV_DIR, "cv_docs.csv"))
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(cv))
    n_tr = int(len(cv) * 0.8)
    tr, ho = cv.iloc[idx[:n_tr]], cv.iloc[idx[n_tr:]]
    N_TR_DOCS = len(tr)
    print(f"Held-out protocol cũ: {len(tr)} train / {len(ho)} held-out (seed 42)")

    lex_ho = build_lexicon(tr["text"])
    extra = load_extra_lexicon(quiet=True)
    lex_all = lex_ho + [s for s in extra if s not in set(lex_ho)]
    lex_set = set(lex_all)
    vec = fit(tr["text"])
    feat = list(vec.get_feature_names_out())
    X = vec.transform(ho["text"])
    scan_re = make_scan_re(lex_all)
    plural_re = make_plural_re(lex_set)
    print(f"Danh bạ eval: {len(lex_all)} (tự gieo {len(lex_ho)} + ngoài {len(extra)})")

    texts = [str(t) for t in ho["text"]]
    ntexts = [norm(t) for t in texts]
    gts = [set(parse_skills_block(t)) for t in texts]

    # df trên TRAIN cho E5 (skill xuất hiện trong bao nhiêu doc train)
    df_tab = {}
    for t in tr["text"]:
        for m in scan_re.finditer(norm(t)):
            w = norm(m.group(0))
            if w not in STOP_SKILLS:
                df_tab[w] = df_tab.get(w, 0) + 1

    # ---- DIAG: GT bị mất ở đâu? -------------------------------------------
    top_by_i = [[t for t, _ in topk_of(X, i, feat, lex_set)] for i in range(len(ho))]
    merged_by_i, pos_by_i = [], []
    for i in range(len(ho)):
        mg, ps = extract_merged(ntexts[i], texts[i], top_by_i[i], scan_re, lex_set)
        merged_by_i.append(mg)
        pos_by_i.append(ps)

    tot = in_lex = in_text = in_top = in_merged = cont_only = 0
    for i, gt in enumerate(gts):
        for g in gt:
            tot += 1
            if g in lex_set:
                in_lex += 1
            if wb_search(g, ntexts[i]):
                in_text += 1
            if g in top_by_i[i]:
                in_top += 1
            if g in merged_by_i[i]:
                in_merged += 1
            elif any(containment_hit(e, g) for e in merged_by_i[i]):
                cont_only += 1
    print(f"\n== DIAG trên {sum(len(g) for g in gts)} phrase GT của "
          f"{sum(1 for g in gts if g)} CV có GT ==")
    print(f"  GT có trong danh bạ (đúng string)      : {in_lex / tot:6.1%}")
    print(f"  GT có trong text (đúng string)         : {in_text / tot:6.1%}")
    print(f"  GT vào top-25 TF-IDF                   : {in_top / tot:6.1%}")
    print(f"  GT vào output thật (top+scan)          : {in_merged / tot:6.1%}")
    print(f"  GT chỉ khớp nếu so kiểu 'chứa'         : {cont_only / tot:6.1%}")
    print(f"  → mất vì: không trong text {(tot - in_text) / tot:.1%} | "
          f"trong text nhưng lọt ngoài top+scan {(in_text - in_merged) / tot:.1%}")

    # ---- Bảng thí nghiệm ---------------------------------------------------
    def score(name, outs, match_fn, match_name):
        precs, r25, rfull, n = [], [], [], 0
        for i, merged in enumerate(outs):
            gt = gts[i]
            if not gt or not merged:
                continue
            h25 = sum(1 for e in merged[:TOP_K] if any(match_fn(e, g) for g in gt))
            hfl = sum(1 for e in merged if any(match_fn(e, g) for g in gt))
            precs.append(h25 / min(TOP_K, len(merged)))
            r25.append(h25 / len(gt))
            rfull.append(hfl / len(gt))
            n += 1
        print(f"  {name:<26}| {match_name:<8}| {np.mean(precs):6.2%} | "
              f"{np.mean(r25):6.2%} | {np.mean(rfull):6.2%} | {n}")

    print(f"\n== Bảng kết quả (P@25 / R@25 / R@full trên {sum(1 for g in gts if g)} CV có GT) ==")
    print(f"  {'cấu hình':<26}| {'khớp GT':<8}| {'P@25':>7} | {'R@25':>7} | {'R@full':>7} | n")

    # baseline cũ: chỉ topk_of (để đối chiếu 39.75/25.33)
    score("baseline cũ (chỉ topk)", top_by_i, lambda e, g: e == g, "exact")
    score("E1 output thật", merged_by_i, lambda e, g: e == g, "exact")
    score("E2 output thật + chứa", merged_by_i, containment_hit, "contain")

    tier_outs = [extract_merged(ntexts[i], texts[i], top_by_i[i], scan_re, lex_set,
                                tier=True)[0] for i in range(len(ho))]
    score("E3 + tier order", tier_outs, containment_hit, "contain")

    pl_outs = [extract_merged(ntexts[i], texts[i], top_by_i[i], scan_re, lex_set,
                              plural_re=plural_re, tier=True)[0] for i in range(len(ho))]
    score("E4 + quét số nhiều", pl_outs, containment_hit, "contain")

    for cap in (0.4, 0.5):
        df_outs = [extract_merged(ntexts[i], texts[i], top_by_i[i], scan_re, lex_set,
                                  plural_re=plural_re, tier=True,
                                  df_tab=df_tab, df_cap=cap)[0] for i in range(len(ho))]
        score(f"E5 + lọc generic df<={cap}", df_outs, containment_hit, "contain")

    # E6: nới lỏng min_docs của danh bạ tự gieo (5 → 3/2) — nút cổ chai là độ phủ danh bạ
    for md in (3, 2):
        lex_md = build_lexicon(tr["text"], min_docs=md)
        lex_all_md = lex_md + [s for s in extra if s not in set(lex_md)]
        lex_set_md = set(lex_all_md)
        scan_md = make_scan_re(lex_all_md)
        top_md = [[t for t, _ in topk_of(X, i, feat, lex_set_md)] for i in range(len(ho))]
        md_outs = [extract_merged(ntexts[i], texts[i], top_md[i], scan_md, lex_set_md,
                                  tier=True)[0] for i in range(len(ho))]
        score(f"E6 min_docs={md} (+{len(lex_all_md) - len(lex_all)})", md_outs,
              containment_hit, "contain")

    # E7: mở rộng danh bạ bằng block Skills của 1.000 CV test ngoài
    tcv = pd.read_csv(os.path.join(CSV_DIR, "test_cv_docs.csv"))
    lex_test = build_lexicon(tcv["text"])
    lex_all2 = lex_all + [s for s in lex_test if s not in lex_set]
    lex_set2 = set(lex_all2)
    scan_re2 = make_scan_re(lex_all2)
    plural_re2 = make_plural_re(lex_set2)
    top2_by_i = [[t for t, _ in topk_of(X, i, feat, lex_set2)] for i in range(len(ho))]
    e7_outs = [extract_merged(ntexts[i], texts[i], top2_by_i[i], scan_re2, lex_set2,
                              plural_re=plural_re2, tier=True)[0] for i in range(len(ho))]
    score(f"E7 + danh bạ test-CV (+{len(lex_all2) - len(lex_all)})", e7_outs,
          containment_hit, "contain")
    e7_exact = [extract_merged(ntexts[i], texts[i], top2_by_i[i], scan_re2, lex_set2,
                               tier=True)[0] for i in range(len(ho))]
    score("E7 (khớp exact)", e7_exact, lambda e, g: e == g, "exact")


if __name__ == "__main__":
    main()
