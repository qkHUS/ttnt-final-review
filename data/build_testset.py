# -*- coding: utf-8 -*-
"""build_testset.py — Dựng bộ KIỂM TRA NGOÀI ~1000 CV + 100 JD (khác nguồn data chính).

Vì sao: matching_train/test của project là MỘT nguồn (80k cặp sinh từ 643 CV +
351 JD; thậm chí 477 CV + 71 JD của test trùng train). Muốn biết pipeline
"dùng được thật không" thì phải test trên data chưa từng thấy, nguồn khác hẳn.

Nguồn (HuggingFace, tự tải về data/_hf_tmp/ nếu thiếu):
  CV: d4rk3r/resumes-raw-pdf — ~1.9k CV THẬT dạng PDF (all-domains, it-domain);
      bắn text bằng pypdf, giữ doc >= 500 ký tự. Nếu thiếu 1000 thì lấy thêm
      CV unique từ Sachinkelenjaguri/Resume_dataset (UpdatedResumeDataSet.csv).
  JD: jacob-hugging-face/job-descriptions — 853 JD duy nhất; lấy mẫu 100 (seed 42).

Output:
  data/csv/test_cv_docs.csv (doc_id, source, category, text đã mask PII)
  data/csv/test_jd_docs.csv (doc_id, source, position_title, company_name, text đã mask PII)
doc_id = md5(text thô)[:12] — cùng quy ước build_pairs.py.
Cuối script in mức CHỒNG LẬP doc_id với corpus chính (cv_docs/jd_docs)
— kỳ vọng ~0 để chứng minh đây là luồng data thật sự mới.

Chạy từ thư mục gốc project:  python data/build_testset.py
"""

import glob
import io
import os

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download
from pypdf import PdfReader

from build_pairs import CSV_DIR, DATA, doc_id, mask_pii

HF_DIR = f"{DATA}/_hf_tmp"
N_CV_TARGET = 1000
N_JD = 100
JD_MIN_CHARS = 200
CV_MIN_CHARS = 500
RANDOM_STATE = 42

PDF_REPO = "d4rk3r/resumes-raw-pdf"
PDF_DIRS = [f"{HF_DIR}/d4rk3r/all-domains", f"{HF_DIR}/d4rk3r/it-domain"]
KAGGLE_REPO, KAGGLE_FN = "Sachinkelenjaguri/Resume_dataset", "UpdatedResumeDataSet.csv"
JD_REPO, JD_FN = "jacob-hugging-face/job-descriptions", "training_data.csv"


def local_first_fetch(repo: str, fn: str) -> str:
    """Dùng file đã cache trong data/_hf_tmp nếu có. HF rate-limit IP ẩn danh (429)
    nên tránh gọi mạng khi file đã tải."""
    local = f"{HF_DIR}/{fn}"
    if os.path.exists(local):
        return local
    return hf_hub_download(repo, fn, repo_type="dataset", local_dir=HF_DIR)


def pdf_text(raw: bytes) -> str:
    """Bắn text từ 1 file PDF (nội dung bytes); trả '' nếu PDF lỗi/scan ảnh."""
    try:
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        return ""


def collect_cvs() -> pd.DataFrame:
    seen: set[str] = set()
    rows: list[dict] = []
    n_fail = n_short = n_dup = 0

    for d in PDF_DIRS:  # all-domains trước, it-domain chỉ để bù nếu thiếu
        if len(rows) >= N_CV_TARGET:
            break
        if not os.path.isdir(d):
            pattern = "all-domains/*" if "all-domains" in d else "it-domain/*"
            print(f"   tải PDF {pattern} ...")
            try:
                snapshot_download(PDF_REPO, repo_type="dataset", local_dir=f"{HF_DIR}/d4rk3r",
                                  allow_patterns=[pattern])
            except Exception as e:  # HF 429 rate-limit → dùng những gì đã có
                print(f"   cảnh báo: không tải tiếp được ({type(e).__name__}) — dùng PDF đã cache")
        for path in sorted(glob.glob(f"{d}/*.pdf")):
            if len(rows) >= N_CV_TARGET:
                break
            text = pdf_text(open(path, "rb").read())
            if len(text) < CV_MIN_CHARS:
                n_short += 1
                continue
            did = doc_id(text)
            if did in seen:
                n_dup += 1
                continue
            seen.add(did)
            rows.append({
                "doc_id": did,
                "source": "d4rk3r-resumes-raw-pdf",
                "category": os.path.basename(os.path.dirname(path)),
                "text": mask_pii(text),
            })
        print(f"   {d}: cộng dồn {len(rows)} CV duy nhất (lỗi ngắn={n_short}, trùng={n_dup})")

    # Bù bằng CV unique từ Kaggle nếu PDF chưa đủ
    if len(rows) < N_CV_TARGET:
        kaggle = pd.read_csv(local_first_fetch(KAGGLE_REPO, KAGGLE_FN)).dropna(subset=["Resume"])
        for _, r in kaggle.iterrows():
            if len(rows) >= N_CV_TARGET:
                break
            text = str(r["Resume"])
            if len(text) < CV_MIN_CHARS:
                continue
            did = doc_id(text)
            if did in seen:
                continue
            seen.add(did)
            rows.append({"doc_id": did, "source": "kaggle-updated-resume",
                         "category": r["Category"], "text": mask_pii(text)})
        print(f"   bù Kaggle: cộng dồn {len(rows)} CV duy nhất")

    return pd.DataFrame(rows)


def collect_jds() -> pd.DataFrame:
    jd = pd.read_csv(local_first_fetch(JD_REPO, JD_FN))
    jd = jd.dropna(subset=["job_description"])
    jd = jd[jd["job_description"].str.len() >= JD_MIN_CHARS].copy()
    jd["did"] = jd["job_description"].map(doc_id)
    jd = jd.drop_duplicates("did")
    jd = jd.sample(n=min(N_JD, len(jd)), random_state=RANDOM_STATE)
    return pd.DataFrame({
        "doc_id": jd["did"],
        "source": "jacob-job-descriptions",
        "position_title": jd["position_title"].fillna(""),
        "company_name": jd["company_name"].fillna(""),
        "text": jd["job_description"].map(mask_pii),
    })


def main() -> None:
    print("== Gom CV ngoài (PDF -> text) ==")
    cv = collect_cvs().sort_values("doc_id").reset_index(drop=True)
    print("== Gom JD ngoài ==")
    jd = collect_jds().sort_values("doc_id").reset_index(drop=True)

    # ---- Kiểm chứng: mức chồng lặp với corpus chính phải ~0 -----------------
    old_ids: set[str] = set()
    for f in ("cv_docs.csv", "jd_docs.csv"):
        path = os.path.join(CSV_DIR, f)
        if os.path.exists(path):
            old_ids |= set(pd.read_csv(path, usecols=["doc_id"])["doc_id"])
    cv_overlap = cv["doc_id"].isin(old_ids).sum()
    jd_overlap = jd["doc_id"].isin(old_ids).sum()
    print(f"\nChồng lặp doc_id với corpus chính: CV {cv_overlap}/{len(cv)}, JD {jd_overlap}/{len(jd)}"
          " (kỳ vọng ~0 → luồng data mới thật sự)")

    cv.to_csv(f"{CSV_DIR}/test_cv_docs.csv", index=False, encoding="utf-8-sig")
    jd.to_csv(f"{CSV_DIR}/test_jd_docs.csv", index=False, encoding="utf-8-sig")
    print(f"\ntest_cv_docs.csv: {len(cv)} CV | test_jd_docs.csv: {len(jd)} JD")
    print("CV theo nguồn:", cv["source"].value_counts().to_dict())


if __name__ == "__main__":
    main()
