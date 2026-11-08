# -*- coding: utf-8 -*-
"""Chấm điểm fit CV ↔ JD bằng TF-IDF + phần trăm skill khớp.

Input  (tất cả do các script trước tạo):
  data/csv/pairs.csv      93,733 cặp (pair_id, resume_id, jd_id, label, split, ...)
  data/csv/cv_docs.csv    643 CV (doc_id, text)
  data/csv/jd_docs.csv    351 JD (doc_id, text)
  data/extracted/{cv,jd}/ JSON kỹ năng đã trích (extract_json.py)

Output:
  data/scored/pairs_scored.csv   từng cặp: điểm cosine, độ phủ skill, nhãn
  data/scored/demo_matches.json  3 ví dụ minh hoạ matched/missing skills

Cách dùng:  python data/match_score.py   (chạy từ thư mục gốc dự án)
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_json import CSV_DIR, fit  # cùng bộ TfidfVectorizer với pipeline trích xuất

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "scored")

# Normalizer cho việc SO KHỚP chuỗi skill: giữ + # . / ' - vì nằm trong skill thật
# (c++, c#, asp.net, node.js, .net...), mọi ký tự khác coi như khoảng trắng.
_KEEP = re.compile(r"[^a-z0-9+.#/'-]+")


def sk_norm(s: str) -> str:
    """Chuẩn hoá 1 skill/đoạn văn bản để so khớp: viết thường + bỏ ký tự lạ."""
    return _KEEP.sub(" ", str(s).lower()).strip()


def ngram_set(text: str, max_k: int = 4) -> frozenset:
    """Tập mọi cụm 1..max_k từ liền nhau của văn bản (đã sk_norm).

    skill r "khớp trong CV" ⟺ sk_norm(r) nằm trong tập này — tra O(1),
    đồng thời tự xử lý đúng biên từ (không cần regex cho từng cặp).
    """
    toks = sk_norm(text).split()
    out, n = set(), len(toks)
    for k in range(1, max_k + 1):
        for i in range(n - k + 1):
            out.add(" ".join(toks[i:i + k]))
    return frozenset(out)


def load_all():
    pairs = pd.read_csv(os.path.join(CSV_DIR, "pairs.csv"))
    cv = pd.read_csv(os.path.join(CSV_DIR, "cv_docs.csv"))
    jd = pd.read_csv(os.path.join(CSV_DIR, "jd_docs.csv"))
    return pairs, cv, jd


def build_features(pairs, cv, jd, verbose=True):
    """Tính 3 đặc trưng cho MỖI CẶP (tính 1 lần cho các cặp doc trùng rồi join):

    cosine    : cos(góc) giữa vector TF-IDF của CV và của JD (0..1 vì L2 + tf-idf ≥ 0)
    req_cov   : tỷ lệ required_skills của JD xuất hiện trong TOÀN BỘ văn bản CV
    pref_cov  : tỷ lệ preferred_skills của JD xuất hiện trong văn bản CV
    """
    # ---- 1) Ma trận TF-IDF trên ĐẦY ĐỦ 994 tài liệu (fit 1 lần, dùng chung) ----
    corpus = pd.concat([cv["text"], jd["text"]], ignore_index=True).fillna("")
    vec = fit(corpus)
    X = vec.transform(corpus)                     # 994 dòng, mỗi dòng đã chuẩn L2
    C = X[: len(cv)]                              # khối CV 643 dòng
    J = X[len(cv):]                               # khối JD 351 dòng
    crow = pairs["resume_id"].map({d: i for i, d in enumerate(cv["doc_id"])}).to_numpy()
    jrow = pairs["jd_id"].map({d: i for i, d in enumerate(jd["doc_id"])}).to_numpy()
    # cos(c, j) = c·j vì cả hai đã chuẩn L2 → nhân từng phần tử rồi cộng
    cosine = np.asarray(C[crow].multiply(J[jrow]).sum(axis=1)).ravel()

    # ---- 2) Đọc kỹ năng đã trích + dựng bộ n-gram cho từng CV (643 bộ) ----
    def skills_of(folder, doc_id):
        p = os.path.join(os.path.dirname(CSV_DIR), "extracted", folder, f"{doc_id}.json")
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    cv_ids = cv["doc_id"].tolist()
    jd_ids = jd["doc_id"].tolist()
    cv_json = {d: skills_of("cv", d) for d in cv_ids}
    jd_json = {d: skills_of("jd", d) for d in jd_ids}
    cv_ngrams = {d: ngram_set(t) for d, t in zip(cv_ids, cv["text"].fillna(""))}

    # ---- 3) Độ phủ skill cho các cặp DUY NHẤT (resume_id, jd_id) rồi join ----
    uniq = pairs[["resume_id", "jd_id"]].drop_duplicates()
    req_map, pref_map = {}, {}
    for r_id, j_id in uniq.itertuples(index=False):
        jdoc = jd_json.get(j_id, {})
        grams = cv_ngrams.get(r_id, frozenset())

        def cov(sk_list):
            sk_list = [sk_norm(s) for s in sk_list if sk_norm(s)]
            if not sk_list:
                return np.nan          # JD không có danh sách này → không đánh giá được
            hit = sum(1 for s in sk_list if s in grams)
            return hit / len(sk_list)

        req_map[(r_id, j_id)] = cov(jdoc.get("required_skills", []))
        pref_map[(r_id, j_id)] = cov(jdoc.get("preferred_skills", []))

    key = list(zip(uniq["resume_id"], uniq["jd_id"]))
    lut_req = dict(zip(key, [req_map[k] for k in key]))
    lut_pref = dict(zip(key, [pref_map[k] for k in key]))
    pk = list(zip(pairs["resume_id"], pairs["jd_id"]))
    feats = pd.DataFrame({
        "pair_id": pairs["pair_id"].values,
        "resume_id": pairs["resume_id"].values,
        "jd_id": pairs["jd_id"].values,
        "label": pairs["label"].values,
        "split": pairs["split"].values,
        "cosine": cosine,
        "req_cov": [lut_req[k] for k in pk],
        "pref_cov": [lut_pref[k] for k in pk],
    })
    if verbose:
        print(f"cặp duy nhất (cv,jd): {len(uniq):,} / {len(pairs):,} hàng")
        print(feats.groupby("label")[["cosine", "req_cov"]].mean().round(4))
    return feats


# ------------------------- ĐÁNH GIÁ -------------------------

def best_f1_threshold(y, s):
    """Quét ngưỡng trên train, trả (ngưỡng, F1) tốt nhất cho lớp dương."""
    best_t, best_f = 0.5, -1.0
    for t in np.quantile(s, np.linspace(0.01, 0.99, 197)):
        f = f1_score(y, s >= t, zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t, best_f


def binary_report(name, y_true, score, split, t=None):
    """In 1 dòng đánh giá nhị phân; t=None → tự chọn ngưỡng tốt nhất trên tập này."""
    auc = roc_auc_score(y_true, score)
    ap = average_precision_score(y_true, score)
    if t is None:
        t, _ = best_f1_threshold(y_true, score)
    pred = score >= t
    p = precision_score(y_true, pred, zero_division=0)
    r = recall_score(y_true, pred, zero_division=0)
    f = f1_score(y_true, pred, zero_division=0)
    print(f"  {name:<28} AUC {auc:.3f} | PR-AUC {ap:.3f} | t*={t:.4f} "
          f"→ P {p:.3f} R {r:.3f} F1 {f:.3f} [{split}]")
    return {"auc": auc, "pr_auc": ap, "t": t, "p": p, "r": r, "f1": f}


def band_report(feats, col, t_lo, t_hi):
    """Chia 3 vùng điểm → so với nhãn 3 lớp (chỉ đo, không huấn luyện gì thêm)."""
    s = feats[col]
    pred = np.where(s < t_lo, "No Fit", np.where(s < t_hi, "Potential Fit", "Good Fit"))
    labels = ["No Fit", "Potential Fit", "Good Fit"]
    cm = confusion_matrix(feats["label"], pred, labels=labels)
    acc = (pred == feats["label"]).mean()
    return cm, acc, labels


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pairs, cv, jd = load_all()
    print(f"pairs {len(pairs):,} | nhãn: {pairs['label'].value_counts().to_dict()}")

    feats = build_features(pairs, cv, jd)
    tr = feats[feats["split"] == "train"]
    te = feats[feats["split"] == "test"]

    for pos_label, tag in (("Good Fit", "Good vs còn lại"),
                           ("Potential Fit", "Potential Fit vs còn lại"),
                           ("No Fit", "No Fit vs còn lại")):
        print(f"\n== Nhị phân: {tag} (dương = '{pos_label}') ==")
        # mỗi đặc trưng: chọn ngưỡng tốt-F1 trên TRAIN, áp nguyên lên TEST
        for col in ("cosine", "req_cov"):
            mtr, mte = tr[col].notna(), te[col].notna()
            t, _ = best_f1_threshold((tr["label"] == pos_label)[mtr].astype(int), tr[col][mtr])
            binary_report(f"chỉ {col}", (te["label"] == pos_label)[mte].astype(int),
                          te[col][mte], "test", t=t)
        # LogisticRegression học kết hợp 3 đặc trưng trên TRAIN, đo trên TEST
        Xtr, ytr = tr[["cosine", "req_cov", "pref_cov"]].fillna(-1), (tr["label"] == pos_label).astype(int)
        Xte, yte = te[["cosine", "req_cov", "pref_cov"]].fillna(-1), (te["label"] == pos_label).astype(int)
        lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr)
        w = lr.coef_[0].round(3)
        print(f"  LR w = cosine{w[0]:+.3f} req_cov{w[1]:+.3f} pref_cov{w[2]:+.3f} b={lr.intercept_[0]:+.3f}")
        m1 = te["cosine"].notna()
        binary_report("LR(3 đặc trưng) test", yte[m1], lr.predict_proba(Xte[m1])[:, 1], "test")
        binary_report("chỉ cosine test", (te["label"] == pos_label)[m1].astype(int),
                      te["cosine"][m1], "test")

    # Ngưỡng final: chọn trên TRAIN cho bài Good-vs-còn-lại, áp nguyên lên TEST
    print("\n== 3 vùng điểm (ngưỡng chọn trên train, đo trên test) ==")
    ytr_g = (tr["label"] == "Good Fit").astype(int)
    t_hi, _ = best_f1_threshold(ytr_g, tr["cosine"])
    ytr_f = (tr["label"] != "No Fit").astype(int)
    t_lo, _ = best_f1_threshold(ytr_f, tr["cosine"])
    if t_lo > t_hi:
        t_lo, t_hi = min(t_lo, t_hi), max(t_lo, t_hi)
    print(f"  ngưỡng: t_lo={t_lo:.4f} (fit/no-fit), t_hi={t_hi:.4f} (potential/good)")
    cm, acc, labels = band_report(te, "cosine", t_lo, t_hi)
    print(f"  accuracy 3 lớp (test): {acc:.3f}  |  baseline đoán hết 'No Fit': "
          f"{(te['label'] == 'No Fit').mean():.3f}")
    macro = f1_score(te["label"], np.where(te["cosine"] < t_lo, "No Fit",
                     np.where(te["cosine"] < t_hi, "Potential Fit", "Good Fit")),
                     average="macro", labels=labels, zero_division=0)
    print(f"  macro-F1 3 lớp (test): {macro:.3f}")
    print("  confusion (hàng = nhãn thật, cột = dự đoán):")
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())

    feats.to_csv(os.path.join(OUT_DIR, "pairs_scored.csv"), index=False)

    # ---- Demo matched/missing skills cho 3 cặp tiêu biểu ----
    jd_ids = jd["doc_id"].tolist()
    jd_json = {d: json.load(open(os.path.join(os.path.dirname(CSV_DIR),
                                              "extracted", "jd", f"{d}.json"), encoding="utf-8"))
               for d in jd_ids}
    cv_json = {d: json.load(open(os.path.join(os.path.dirname(CSV_DIR),
                                              "extracted", "cv", f"{d}.json"), encoding="utf-8"))
               for d in cv["doc_id"].tolist()}
    grams_all = {d: ngram_set(t) for d, t in zip(cv["doc_id"], cv["text"].fillna(""))}

    demo = []
    for want in ("Good Fit", "Potential Fit", "No Fit"):
        row = te[te["label"] == want].nlargest(1, "cosine").iloc[0]
        jdoc = jd_json[row["jd_id"]]
        grams = grams_all[row["resume_id"]]
        req = [sk_norm(s) for s in jdoc.get("required_skills", [])]
        matched = [s for s in req if s in grams]
        missing = [s for s in req if s not in grams]
        demo.append({
            "label": want, "resume_id": row["resume_id"], "jd_id": row["jd_id"],
            "cosine": round(float(row["cosine"]), 4),
            "req_cov": None if np.isnan(row["req_cov"]) else round(float(row["req_cov"]), 4),
            "job_title": jdoc.get("job_title"),
            "matched_required": matched, "missing_required": missing,
        })
    with open(os.path.join(OUT_DIR, "demo_matches.json"), "w", encoding="utf-8") as f:
        json.dump(demo, f, ensure_ascii=False, indent=2)
    print(f"\nđã ghi {OUT_DIR}\\pairs_scored.csv + demo_matches.json")
    print(json.dumps(demo[0], ensure_ascii=False, indent=2)[:600])


if __name__ == "__main__":
    main()
