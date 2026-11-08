# -*- coding: utf-8 -*-
"""build_lexicon_extra.py — Phình danh bạ skill bằng nguồn NGOÀI (HuggingFace).

Vì sao: danh bạ tự seed từ block Skills của 643 CV matching chỉ có ~707 skill,
trong khi JD thật dùng nhiều skill mà block Skills không cover. Ta gom thêm
skill từ 3 bộ data NGOÀI (không đụng text CV/JD matching nên KHÔNG leakage):

  1. batuhanmtl/job-skill-set          — 1.167 JD, mỗi JD kèm job_skill_set (list)
  2. tabiya/esco_skills_test skillspan — skill được nêu trong câu, nhãn theo ESCO
  3. tabiya/esco_skills_test techwolf  — skill gán theo hồ sơ nhân viên

File raw tự tải bằng huggingface_hub về data/_hf_tmp/ nếu chưa có.
Chạy:  python data/build_lexicon_extra.py
→ data/skills_lexicon_extra.txt (1 skill/dòng; extract_json.py tự nạp nếu file tồn tại)
"""

import ast
import os
import re

import pandas as pd
from huggingface_hub import hf_hub_download

from extract_json import STOP_SKILLS, norm

HF_DIR = "data/_hf_tmp"
OUT_PATH = "data/skills_lexicon_extra.txt"
BASE_PATH = "data/skills_lexicon.txt"

TRAIL_PUNCT = " .,;:-"  # rác cắt được: 'plan ', 'practices .'
JUNK = {"plan", "nan", "etc", "n a", "other", "others", "year", "years"}
# Cùng chuẩn shape với GT block Skills (parse_skills_block của extract_json):
MAX_WORDS, MIN_LEN, MAX_LEN = 4, 2, 40

SOURCES = [
    ("job-skill-set", "batuhanmtl/job-skill-set", "data/train-00000-of-00001.parquet"),
    ("skillspan", "tabiya/esco_skills_test", "data/skillspan_with_id.parquet"),
    ("techwolf", "tabiya/esco_skills_test", "data/techwolf_with_id.parquet"),
]


def fetch(repo: str, fn: str) -> str:
    """Tải file raw về data/_hf_tmp (dùng lại cache nếu đã có)."""
    return hf_hub_download(repo, fn, repo_type="dataset", local_dir=HF_DIR)


def clean(raw) -> str:
    s = norm(str(raw)).rstrip(TRAIL_PUNCT)
    # Nguồn hay viết ".NET" thành ". NET" → ". net"; gập lại thành ".net"
    return re.sub(r"\.\s+(?=[a-z0-9])", ".", s)


def shape_ok(s: str) -> bool:
    return (
        MIN_LEN <= len(s) <= MAX_LEN
        and len(s.split()) <= MAX_WORDS
        and re.search(r"[a-z]", s) is not None
        and not s.isdigit()
        and "<" not in s
        and ">" not in s  # placeholder kiểu "<organization> systems"
        and s not in STOP_SKILLS
        and s not in JUNK
    )


def skills_from_job_skill_set(path: str) -> list:
    df = pd.read_parquet(path)
    out = []
    for v in df["job_skill_set"]:
        try:
            items = ast.literal_eval(str(v))
            if isinstance(items, (list, tuple)):
                out += [str(x) for x in items]
        except (ValueError, SyntaxError):
            pass  # ô lỗi định dạng — bỏ
    return out


def skills_from_skillspan(path: str) -> list:
    df = pd.read_parquet(path)
    return list(df["span"].dropna()) + list(df["sub_span"].dropna())


def skills_from_techwolf(path: str) -> list:
    df = pd.read_parquet(path)
    # Chỉ lấy cột 'label' (nhãn skill ESCO). ĐỪNG auto-pick: cột ID ('te-0')
    # và UUID từng lọt qua bộ lọc shape vì cũng "trông như" cột chữ ngắn.
    return list(df["label"].dropna())


def main() -> None:
    base: set[str] = set()
    if os.path.exists(BASE_PATH):
        base = {norm(l) for l in open(BASE_PATH, encoding="utf-8") if l.strip()}
    print(f"Danh bạ gốc: {len(base)} skill (từ {BASE_PATH})")

    collectors = {
        "job-skill-set": skills_from_job_skill_set,
        "skillspan": skills_from_skillspan,
        "techwolf": skills_from_techwolf,
    }
    extra: set[str] = set()
    for name, repo, fn in SOURCES:
        path = fetch(repo, fn)
        raws = collectors[name](path)
        got = {clean(r) for r in raws}
        got = {s for s in got if shape_ok(s) and s not in base}
        print(f"{name}: {len(raws):,} mục thô → {len(got):,} skill mới đạt chuẩn")
        extra |= got

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(extra)))
    print(f"TỔNG CỘNG: {len(extra)} skill mới → {OUT_PATH}")
    print("Mẫu:", ", ".join(sorted(extra)[:30]))


if __name__ == "__main__":
    main()
