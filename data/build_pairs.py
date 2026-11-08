# -*- coding: utf-8 -*-
"""build_pairs.py — Chuẩn hóa dữ liệu matching thành "kho tài liệu + bảng cặp".

Vấn đề của dữ liệu thô:
  - matching_train/test.csv chứa TOÀN BỘ văn bản CV & JD trong từng dòng cặp,
    nên 93k dòng thật ra chỉ xoay quanh ~642 CV và ~351 JD duy nhất.
  - Các file sau khi làm sạch (cv_clean.csv, jd_clean.csv) đã bị MẤT liên kết
    giữa văn bản và cặp (không biết CV nào ghép với JD nào, label gì).

Giải pháp — tách 2 mức:
  1) Mức TÀI LIỆU: cv_docs.csv / jd_docs.csv — mỗi văn bản duy nhất 1 dòng,
     định danh bằng doc_id = md5(text) (ổn định qua các lần chạy/máy khác nhau).
  2) Mức CẶP: pairs.csv — mỗi dòng 1 cặp (resume_id, jd_id, label, split, domain),
     KHÔNG chứa văn bản (tránh nhân bản ~600MB); cần văn bản thì join theo doc_id.

Lợi ích cho phần trích xuất (TF-IDF / BERT):
  - Chỉ chạy extraction MỘT LẦN trên mỗi tài liệu (~1k lần thay vì 93k),
    lưu kết quả theo doc_id, sau đó mọi phép so khớp chỉ cần join bảng.

Chạy từ thư mục gốc project:  python data/build_pairs.py
"""

import hashlib
import re

import pandas as pd

DATA = "data"
CSV_DIR = "data/csv"  # toàn bộ CSV (thô + derived) nằm ở đây từ 2026

# Mask PII (giống tinh thần của build_docs.py): extraction không cần email/SĐT/URL,
# doc_id vẫn tính từ text THÔ nên join với pairs.csv không đổi.
_PII_RE = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    (re.compile(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"), "[PHONE]"),
    (re.compile(r"https?://\S+|www\.\S+"), "[URL]"),
]


def mask_pii(text: str) -> str:
    for pattern, repl in _PII_RE:
        text = pattern.sub(repl, text)
    return text


def doc_id(text: str) -> str:
    """Định danh tài liệu bằng md5 của toàn bộ văn bản (12 ký tự đầu)."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def main() -> None:
    tr = pd.read_csv(
        f"{CSV_DIR}/matching_train.csv",
        encoding="utf-8-sig",
        usecols=["resume", "jd", "label", "resume_domain", "jd_domain"],
    )
    te = pd.read_csv(
        f"{CSV_DIR}/matching_test.csv",
        encoding="utf-8-sig",
        usecols=["resume", "jd", "label", "resume_domain", "jd_domain"],
    )
    print(f"Raw: train={len(tr):,} cặp, test={len(te):,} cặp")

    # ---- 1) Gán doc_id cho từng phía, ghép train+test thành bảng cặp --------
    frames = []
    for split, df in (("train", tr), ("test", te)):
        d = df.copy()
        d["resume_id"] = d["resume"].map(doc_id)
        d["jd_id"] = d["jd"].map(doc_id)
        d["split"] = split
        frames.append(d)
    pairs = pd.concat(frames, ignore_index=True)

    # Sắp xếp deterministic để pair_id không đổi giữa các lần chạy
    pairs = pairs.sort_values(
        ["split", "resume_id", "jd_id", "label"], kind="stable"
    ).reset_index(drop=True)
    pairs.insert(0, "pair_id", [f"p{i:06d}" for i in range(len(pairs))])
    pairs = pairs[
        [
            "pair_id",
            "resume_id",
            "jd_id",
            "label",
            "split",
            "resume_domain",
            "jd_domain",
        ]
    ]
    pairs.to_csv(f"{CSV_DIR}/pairs.csv", index=False, encoding="utf-8-sig")

    # ---- 2) Bảng tài liệu duy nhất (CV & JD) kèm số cặp xuất hiện ----------
    def doc_table(id_col: str, text_col: str, name: str) -> pd.DataFrame:
        g = (
            pairs.groupby(id_col)
            .agg(
                n_train_pairs=("split", lambda s: (s == "train").sum()),
                n_test_pairs=("split", lambda s: (s == "test").sum()),
            )
            .reset_index()
        )
        # text: tra từ dataframe thô (mỗi văn bản có text như nhau trong mọi dòng)
        text_map = pd.concat([tr, te]).drop_duplicates(subset=[text_col])
        text_map = dict(zip(text_map[text_col].map(doc_id), text_map[text_col].map(mask_pii)))
        g["text"] = g[id_col].map(text_map)
        g = g.rename(columns={id_col: "doc_id"})
        g.to_csv(f"{CSV_DIR}/{name}", index=False, encoding="utf-8-sig")
        return g

    cv = doc_table("resume_id", "resume", "cv_docs.csv")
    jd = doc_table("jd_id", "jd", "jd_docs.csv")

    # ---- 3) Kiểm chứng + báo cáo -------------------------------------------
    print(f"pairs.csv : {len(pairs):,} cặp | cv_docs: {len(cv)} CV | jd_docs: {len(jd)} JD")
    print("Label theo split:")
    print(pairs.groupby(["split", "label"]).size().unstack(fill_value=0))

    cv_both = ((cv["n_train_pairs"] > 0) & (cv["n_test_pairs"] > 0)).sum()
    print(
        f"Cảnh báo leakage: {cv_both}/{len(cv)} CV xuất hiện ở CẢ train và test "
        "(chính thức test split của dataset dùng lại CV của train)."
    )
    print(
        "=> Nếu nhóm tự đánh giá matching, cân nhắc split lại theo TÀI LIỆU "
        "(mỗi CV/JD chỉ thuộc 1 phía) thay vì theo cặp."
    )

    # join kiểm tra: mọi pair phải tìm thấy doc của mình
    assert pairs["resume_id"].isin(set(cv["doc_id"])).all()
    assert pairs["jd_id"].isin(set(jd["doc_id"])).all()
    print("OK: mọi pair đều join được với cv_docs/jd_docs.")


if __name__ == "__main__":
    main()
