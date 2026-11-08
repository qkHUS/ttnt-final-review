# -*- coding: utf-8 -*-
"""extract_json.py — Trích CV/JD thành FILE JSON CẤU TRÚC (thay cho skills_tfidf.csv).

Schema (đúng theo mẫu user đưa):
  CV: { "doc_id": ..., "name": "Nguyen Van A",
        "skills": ["Java", "SQL", ...],
        "experience_years": 3,
        "education": {"degree": "Bachelor",
                      "major": "Information Technology",
                      "university": "..."} }
  JD: { "doc_id": ..., "job_title": "Java Developer",
        "required_skills": [...], "preferred_skills": [...],
        "required_years": 3 }

Cách lấy:
  * skills          — GIỐNG HỆT pipeline cũ: TF-IDF top-25 (chấm điểm đặc trưng)
                      + scan (Ctrl+F theo danh bạ 707 + 6.108 skill ngoài).
  * name            — dòng đầu tiên hợp lệ của CV (không phải tiêu đề/email).
  * experience_years— cụm "N years of experience" trong CV; không có thì cộng
                      các khoảng thời gian làm việc trong mục Experience.
  * education       — mục "Education": bằng cấp / chuyên ngành / trường theo mẫu.
  * job_title       — dòng "Job Title: ..." của JD (fallback: dòng đầu ngắn).
  * required/preferred — skill xuất hiện TRƯỚC chữ "preferred/nice to have/bonus"
                      lần đầu → required; sau nó → preferred.
  * required_years  — số "N years" trong JD mà gần chữ experience/expertise.

Dữ liệu: CHỈ cv_docs.csv (643 CV) + jd_docs.csv (351 JD) + bộ test ngoài.
Chạy từ thư mục gốc project:  python data/extract_json.py
Output: data/extracted/{cv,jd,test_cv,test_jd}/<doc_id>.json (1 file 1 doc)
Cuối script in lại eval held-out P@25/R@25 (baseline cho BERT) + tỉ lệ phủ các trường.
"""

import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(DATA, "csv")  # toàn bộ file CSV nằm ở data/csv/ (dời từ data/ 2026)
LEXICON_PATH = os.path.join(DATA, "skills_lexicon.txt")
EXTRA_LEXICON_PATH = os.path.join(DATA, "skills_lexicon_extra.txt")
OUT_DIR = os.path.join(os.path.dirname(DATA), "data", "extracted")

TOP_K = 25
TOP_CANDIDATES = 200
# min_docs=3 (đo trên held-out, data/improve_skills.py E6): so với 5 → R@full
# 68.3%→76.6%, P@25 49.1%→50.7% (khớp GT kiểu "chứa"). min_docs=2 còn bắt thêm
# (R@full 80.3%) nhưng P@25 giảm nhẹ → chọn 3.
LEXICON_MIN_DOCS = 3
EVAL_SPLIT = 0.8
RANDOM_STATE = 42
CURRENT_YEAR = datetime.now().year

STOP_SKILLS = frozenset({
    "state", "client", "works", "phones", "phone", "letters",
    "processes", "solutions", "procedures", "content",
    # viết tắt bang Mỹ + đuôi công ty: rác địa chỉ/đơn vị trong đuôi khối Skills
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id",
    "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms",
    "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok",
    "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv",
    "wi", "wy", "dc", "uk", "us", "inc", "llc", "ltd", "llp", "pc", "plc",
})

SKILL_HDR_RE = re.compile(r"Skills(?![a-z])")

# Tier ordering (improve_skills.py E3, đo được +4.3pp P@25 / +5.9pp R@25): hit scan
# nằm trong vùng section Skills (SKILL_WIN ký tự sau header) hoặc trên dòng bullet
# được xếp LÊN TRƯỚC các hit ngoài vùng — phần đầu danh sách nhờ vậy sạch hơn.
SKILL_WIN = 1500
BULLET_RE = re.compile(
    r"^\s*(?:[-*•▪◦‣·\u2022\u2023\u25aa\u25cf\uf0a7\uf0b1\uf0b7\uf0d8]|\d{1,2}[.)]|[A-Z]\.)")

ICON_RE = re.compile(r"[\u0000-\u0008\u000b-\u001f\u007f-\u009f\uf000-\uf8ff\u00a0\u2028\u2029]")
NAME_JUNK_RE = re.compile(r"(?i)(curriculum|vitae|resume|profile|personal|contact|address|"
                          r"phone|email|date of birth|nationality|education|academic|degree|"
                          r"university|college|certif|summary|objective|highlights|background|"
                          r"experience|skill|expected|memberships?|license|gpa)")
SEC_HEADER_RE = re.compile(
    r"(?im)^\s*(education|academic(?:\s+\w+)*\s+background|(?:work|professional|employment)\s+"
    r"(?:history|experience)|experience|projects?(?:\s+experience)?|skills?|technical\s+skills|"
    r"certifications?|licenses?|awards?|publications?|languages|summary|objective|about\s+me|"
    r"interests?|activities|volunteer)\b.*$")
DEGREE_RE = re.compile(
    r"(?i)(ph\.?\s?d\b|doctorate|post-?doc|master(?:'s)?\b|m\.?s\.?c?\b|m\.?b\.?a\b|m\.?eng\b|"
    r"bachelor(?:'s)?\b|b\.?s\.?c?\b|b\.?a\.?\b|b\.?eng\b|b\.?tech\b|associate(?:'s)?\b|diploma\b)")
DEGREE_CANON = {
    "phd": "PhD", "ph.d": "PhD", "doctorate": "PhD", "post-doc": "Postdoc",
    "mba": "MBA", "msc": "Master", "ms": "Master", "meng": "Master",
    "master": "Master", "masters": "Master", "master's": "Master",
    "bsc": "Bachelor", "bs": "Bachelor", "ba": "Bachelor", "beng": "Bachelor",
    "btech": "Bachelor", "bachelor": "Bachelor", "bachelors": "Bachelor",
    "bachelor's": "Bachelor", "associate": "Associate", "diploma": "Diploma",
}
MAJOR_RE = re.compile(
    r"(?i)(?:bachelor(?:'s)?|master(?:'s)?|b\.?s\.?c?|m\.?s\.?c?|b\.?a\.?|m\.?a\.?|b\.?eng|"
    r"m\.?eng|degree|graduat\w+|studied)\s*(?:'s)?\s+(?:of|in)\s+([A-Za-z][A-Za-z &/,]{2,60})")
UNIV_RE = re.compile(
    r"(?i)((?:[A-Z][A-Za-z&.,'’\-]+(?:\s+[A-Z][A-Za-z&.,'’\-]+){0,4})\s+"
    r"(?:University|College|Institute|Academy|Polytechnic))|"
    r"((?:University|National)\s+(?:of\s+)?[A-Z][A-Za-z .&]{2,40})")
YEARS_CV_RE = re.compile(
    r"(?i)(\d{1,2})\s*\+?\s*years?\s*(?:of\s*)?(?:professional\s*)?(?:work\s*)?experience")
DATE_RANGE_RE = re.compile(r"((?:19|20)\d{2})\s*(?:-|–|—|to)\s*((?:19|20)\d{2}|present|current|now)",
                           re.I)
JD_TITLE_RE = re.compile(r"(?im)job\s*title\s*[:\-]\s*(.+)")
JD_TITLE_CUT_RE = re.compile(r"(?i)\s*(?:location|duration|pay\b|salary|company|department|"
                             r"reports?\s+to|employment|type|job\s*type)\b")
PREFERRED_RE = re.compile(
    r"(?i)(nice\s*to\s*have|good\s*to\s*have|preferred(?:\s+\w+){0,2}|bonus(?:\s+points)?|"
    r"desirable|plus(?:es)?\s*[:\-]|advantageous)")
JD_YEAR_RE = re.compile(r"(?i)(\d{1,2})\s*\+?\s*(?:years?|yrs)")
JD_YEAR_CTX_RE = re.compile(r"(?i)experience|expertise|background")


def norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def clean_line(s: str) -> str:
    s = ICON_RE.sub(" ", str(s))
    s = re.sub(r"[^\w\s.,'’\-&/()+]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(r" .,;:|\-–—")


def parse_skills_block(text: str, max_chars: int = 2000, max_items: int = 80):
    """Tách danh sách skill sau từ 'Skills' cuối cùng của CV (pseudo ground truth)."""
    last = None
    for last in SKILL_HDR_RE.finditer(str(text)):
        pass  # lấy lần xuất hiện CUỐI (block Skills thường nằm cuối CV)
    if last is None:
        return []
    tail = str(text)[last.end(): last.end() + max_chars]
    items = []
    for raw in tail.split(",")[:max_items]:
        s = norm(raw.strip(" .;:-").replace("\u00b7", " "))  # · = bullet của PDF hỏng
        if ":" in s:  # CV chữ dính: "big data technologies:hadoop" → lấy phần sau ":"
            s = norm(s.rsplit(":", 1)[1])
        # lọc nhiễu: bỏ quá dài / quá 4 từ / toàn số / từ chặn (bang Mỹ, đuôi công ty...)
        if (2 <= len(s) <= 40 and len(s.split()) <= 4 and not s.isdigit()
                and s not in STOP_SKILLS):
            items.append(s)
    return items


def build_lexicon(texts, min_docs: int = LEXICON_MIN_DOCS) -> list[str]:
    """Đếm tần suất item trong block Skills trên NHIỀU doc, giữ item phổ biến."""
    counts: dict[str, int] = {}
    for t in texts:
        for s in set(parse_skills_block(t)):  # set: đếm theo SỐ DOC, không theo số lần
            counts[s] = counts.get(s, 0) + 1
    ok = ((s, c) for s, c in counts.items() if c >= min_docs and s not in STOP_SKILLS)
    return sorted((s for s, _ in ok), key=len, reverse=True)


def load_lexicon() -> list[str]:
    if os.path.exists(LEXICON_PATH):
        return [l.strip() for l in open(LEXICON_PATH, encoding="utf-8") if l.strip()]
    return []


def load_extra_lexicon(quiet: bool = False) -> list[str]:
    if not os.path.exists(EXTRA_LEXICON_PATH):
        return []
    raw = [l.strip() for l in open(EXTRA_LEXICON_PATH, encoding="utf-8") if l.strip()]
    ok = [s for s in dict.fromkeys(raw) if s not in STOP_SKILLS]
    if not quiet:
        print(f"Danh bạ ngoài: nạp {len(ok)} skill từ {EXTRA_LEXICON_PATH}")
    return ok


def make_scan_re(lexicon: list[str]):
    """Regex Ctrl+F: (?<![a-z0-9])(kỹ năng 1|kỹ năng 2|...)(?![a-z0-9]) — dài trước ngắn sau."""
    alts = sorted((re.escape(t) for t in lexicon), key=len, reverse=True)
    return re.compile("(?<![a-z0-9])(?:" + "|".join(alts) + ")(?![a-z0-9])")


def fit(texts) -> TfidfVectorizer:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, stop_words="english", sublinear_tf=True)
    vec.fit(texts)
    return vec


def dedup_overlap(terms) -> list[str]:
    """Bỏ term bị CHỨA trong term khác cùng danh sách ("sql" ⊂ "sql server"):
    hai ô gần trùng nhau chiếm 2 slot top-K trong khi GT chỉ liệt kê 1 dạng
    (đo trên held-out: +1.2pp precision@25, recall mất ~0.8pp — đã nhận)."""
    out = []
    for a in terms:
        if not any(
            a != b and re.search(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])", b)
            for b in terms
        ):
            out.append(a)
    return out


def topk_of(X, i, features, lex_set, top_k=TOP_K):
    row = X.getrow(i)
    order = sorted(range(len(row.data)), key=lambda j: (-row.data[j], features[row.indices[j]]))
    out = []
    for j in order[:TOP_CANDIDATES]:
        term = features[row.indices[j]]
        if term in lex_set:
            out.append((term, float(row.data[j])))
            if len(out) == top_k:
                break
    kept = set(dedup_overlap([t for t, _ in out]))
    return [(t, s) for t, s in out if t in kept]


def skill_windows(text: str) -> list[tuple[int, int]]:
    return [(m.end(), m.end() + SKILL_WIN) for m in SKILL_HDR_RE.finditer(text)]


def hit_tier(p: int, windows: list[tuple[int, int]], text: str) -> int:
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


def containment_hit(a: str, b: str) -> bool:
    """Khớp 'chứa theo biên từ': "sql" ⊂ "sql server" tính là 1 hit (improve_skills.py E2)."""
    if a == b:
        return True
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return re.search(r"(?<![a-z0-9])" + re.escape(short) + r"(?![a-z0-9])", long_) is not None


def extract_skills(text: str, vec, features: list[str], lexicon_set: set, scan_re):
    """Trả về (danh sách skill gộp, top-K tfidf, vị trí xuất hiện đầu của từng skill).

    merged = top TF-IDF (≤25) + scan toàn văn danh bạ, trong đó hit scan nằm trong
    vùng Skills / dòng bullet (tier 1) xếp trước hit ngoài vùng (tier 2)."""
    X = vec.transform([text])
    top = [t for t, _ in topk_of(X, 0, features, lexicon_set)]
    ntext = norm(text)
    pos: dict[str, int] = {}
    for m in scan_re.finditer(ntext):
        w = norm(m.group(0))
        if w not in pos and w not in STOP_SKILLS:
            pos[w] = m.start()
    extras = set(pos) - set(top)
    wins = skill_windows(text)
    t1 = sorted((w for w in extras if hit_tier(pos[w], wins, text) == 1), key=len)
    s1 = set(t1)
    t2 = sorted((w for w in extras if w not in s1), key=len)
    merged = dedup_overlap(list(top) + t1 + t2)
    return merged, top, pos


# ---------------------------------------------------------------- CV fields

def cv_name(text: str) -> str | None:
    """Tên = dòng đầu tiên của CV nếu nó trông giống họ tên người.
    Corpus matching bị dính chữ + máy sinh đã ẩn tên → 0/643 có tên (null là đúng).
    Trên CV thật (bộ ngoài) quy tắc này bắt được tên ở dòng đầu."""
    NAME_STRICT = re.compile(r"^[A-Z][a-zA-Z.'-]+(?: [A-Z][a-zA-Z.'-]+){1,4}$")
    for line in str(text).splitlines()[:1]:
        line = ICON_RE.sub("", clean_line(line)).strip(" -|")
        if not line or "@" in line or any(c.isdigit() for c in line):
            return None
        if NAME_JUNK_RE.search(line) or not NAME_STRICT.match(line):
            return None
        return line
    return None


def cv_experience_years(text: str) -> int | None:
    text = str(text)
    m = [int(g) for g in YEARS_CV_RE.findall(text) if int(g) <= 45]
    if m:
        return max(m)
    msec = SEC_HEADER_RE.search(text)
    if msec and re.match(r"(?i)(work|professional|employment|experience)", msec.group(1)):
        nxt = SEC_HEADER_RE.search(text, msec.end())
        block = text[msec.end(): nxt.start() if nxt else len(text)]
        spans = []
        for a, b in DATE_RANGE_RE.findall(block):
            b2 = CURRENT_YEAR if re.match(r"(?i)present|current|now", b) else int(b)
            a2 = int(a)
            if 1970 <= a2 <= b2 <= CURRENT_YEAR and b2 - a2 <= 45:
                spans.append((a2, b2))
        if spans:
            spans.sort()
            merged, total = list(spans[0]), 0
            for a, b in spans[1:]:
                if a <= merged[1]:
                    merged[1] = max(merged[1], b)
                else:
                    total += merged[1] - merged[0]
                    merged = [a, b]
            total += merged[1] - merged[0]
            return int(total) if total > 0 else None
    return None


def cv_education(text: str) -> dict:
    text = str(text)
    block = text
    msec = SEC_HEADER_RE.search(text)
    if msec and re.match(r"(?i)(education|academic)", msec.group(1)):
        nxt = SEC_HEADER_RE.search(text, msec.end())
        block = text[msec.end(): nxt.start() if nxt else len(text)][:900]

    degree = None
    md = DEGREE_RE.search(block) or DEGREE_RE.search(text)
    if md:
        key = re.sub(r"[^a-z.']", "", md.group(0).lower()).rstrip(".")
        degree = DEGREE_CANON.get(key, md.group(0).strip().title())

    major = None
    mm = MAJOR_RE.search(block) or MAJOR_RE.search(text)
    if mm:
        major = re.split(r"(?i)\s+(?:from|at|,|\.|;)\s+", mm.group(1).strip())[0].strip(" ,/-")

    university = None
    mu = UNIV_RE.search(block) or UNIV_RE.search(text)
    if mu:
        university = clean_line(mu.group(1) or mu.group(2))

    return {"degree": degree, "major": major, "university": university}


# ---------------------------------------------------------------- JD fields

def jd_title(text: str) -> str | None:
    text = str(text)
    m = JD_TITLE_RE.search(text)
    if m:
        t = JD_TITLE_CUT_RE.split(m.group(1))[0]
        t = clean_line(t)
        if 2 <= len(t) <= 80:
            return t
    for line in text.splitlines()[:5]:
        line = clean_line(line)
        words = line.split()
        if (line and 1 <= len(words) <= 10 and not line.isdigit() and "@" not in line
                and not re.match(r"(?i)\b(job\s*details|job\s*description|about|overview|location|"
                                 r"salary|pay\s*rate|benefits|duration|employment|company)\b", line)):
            return line[:80]
    return None


def jd_required_years(text: str) -> int | None:
    vals = []
    for m in JD_YEAR_RE.finditer(str(text)):
        tail = str(text)[m.end(): m.end() + 60]
        if JD_YEAR_CTX_RE.search(tail) and int(m.group(1)) <= 30:
            vals.append(int(m.group(1)))
    return max(vals) if vals else None


def jd_split_skills(text: str, skills: list[str], pos: dict[str, int]) -> tuple[list, list]:
    """Skill có lần xuất hiện ĐẦU TIÊN sau marker preferred → preferred, ngược lại → required."""
    mp = PREFERRED_RE.search(str(text))
    if not mp:
        return list(skills), []
    cut = mp.start()
    required = [s for s in skills if pos.get(norm(s), 10**9) < cut]
    preferred = [s for s in skills if pos.get(norm(s), 10**9) >= cut]
    return required, preferred


# ---------------------------------------------------------------- pipeline

def extract_corpus(docs: pd.DataFrame, vec, features, lexicon_set, scan_re) -> list[dict]:
    out = []
    for d in docs.itertuples(index=False):
        text = str(d.text)
        skills, _top, pos = extract_skills(text, vec, features, lexicon_set, scan_re)
        if d.doc_type == "cv":
            rec = {
                "doc_id": d.doc_id,
                "name": cv_name(text),
                "skills": skills,
                "experience_years": cv_experience_years(text),
                "education": cv_education(text),
            }
        else:
            req, pref = jd_split_skills(text, skills, pos)
            rec = {
                "doc_id": d.doc_id,
                "job_title": jd_title(text),
                "required_skills": req,
                "preferred_skills": pref,
                "required_years": jd_required_years(text),
            }
        out.append(rec)
    return out


def dump_jsonl(recs: list[dict], folder: str) -> None:
    d = os.path.join(OUT_DIR, folder)
    os.makedirs(d, exist_ok=True)
    for r in recs:
        with open(os.path.join(d, f"{r['doc_id']}.json"), "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)


def heldout_eval(cv: pd.DataFrame, extra: list[str]) -> None:
    """Eval 80/20 Y HỆT pipeline cũ (cùng cách xáo trộn, cùng cách so GT).

    In 2 protocol (data/improve_skills.py):
      A) protocol cũ — chỉ top-25 TF-IDF, khớp đứt string (để so với số trong README cũ);
      B) protocol mới — output thật (top + scan, tier order), khớp GT kiểu "chứa
         theo biên từ" (sql ⊂ sql server tính 1 hit) — đo trung thực chất lượng hiện tại."""
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.permutation(len(cv))
    n_tr = int(len(cv) * EVAL_SPLIT)
    tr, ho = cv.iloc[idx[:n_tr]], cv.iloc[idx[n_tr:]]

    lex_ho = build_lexicon(tr["text"])  # lexicon CHƯA từng thấy 20% còn lại
    lex_ho_set = set(lex_ho) | set(extra)  # nguồn ngoài không thấy CV nào của matching
    vec_ho = fit(tr["text"])  # IDF cũng chỉ fit trên 80%
    feat_ho = list(vec_ho.get_feature_names_out())
    X_ho = vec_ho.transform(ho["text"])  # transform 20% held-out
    scan_ho = make_scan_re(lex_ho + [s for s in extra if s not in set(lex_ho)])

    precs, recs, precs_b, recs25_b, recs_full = [], [], [], [], []
    n_gt = 0
    for i, (_, r) in enumerate(ho.iterrows()):
        gt = set(parse_skills_block(r["text"]))
        topk = topk_of(X_ho, i, feat_ho, lex_ho_set)
        if not gt:
            continue
        if topk:
            hits = len({t for t, _ in topk} & gt)
            precs.append(hits / len(topk))
            recs.append(hits / len(gt))
        # protocol B: output thật = top + scan (tier order), khớp "chứa"
        text = str(r["text"])
        ntext = norm(text)
        pos: dict[str, int] = {}
        for m in scan_ho.finditer(ntext):
            w = norm(m.group(0))
            if w not in pos and w not in STOP_SKILLS:
                pos[w] = m.start()
        extras = set(pos) - {t for t, _ in topk}
        wins = skill_windows(text)
        t1 = sorted((w for w in extras if hit_tier(pos[w], wins, text) == 1), key=len)
        t2 = sorted(set(extras) - set(t1), key=len)
        merged = dedup_overlap([t for t, _ in topk] + t1 + t2)
        if merged:
            h25 = sum(1 for e in merged[:TOP_K] if any(containment_hit(e, g) for g in gt))
            hfl = sum(1 for e in merged if any(containment_hit(e, g) for g in gt))
            precs_b.append(h25 / min(TOP_K, len(merged)))
            recs25_b.append(h25 / len(gt))
            recs_full.append(hfl / len(gt))
        n_gt += 1
    print(f"\nĐánh giá held-out ({len(ho)} CV, đánh giá được {n_gt} CV có GT):")
    print(f"  [A] protocol cũ, chỉ top-{TOP_K} TF-IDF, khớp đứt string:")
    print(f"      precision@{TOP_K} = {np.mean(precs):.2%}  recall@{TOP_K} = {np.mean(recs):.2%}")
    print(f"  [B] output thật (top+scan, tier order), khớp GT kiểu 'chứa':")
    print(f"      P@{TOP_K} = {np.mean(precs_b):.2%}  R@{TOP_K} = {np.mean(recs25_b):.2%}  "
          f"R@full = {np.mean(recs_full):.2%} (toàn bộ list JSON)")


def main() -> None:
    cv = pd.read_csv(os.path.join(CSV_DIR, "cv_docs.csv")).assign(doc_type="cv")
    jd = pd.read_csv(os.path.join(CSV_DIR, "jd_docs.csv")).assign(doc_type="jd")
    docs = pd.concat([cv, jd], ignore_index=True)

    base = [s for s in load_lexicon() if s not in STOP_SKILLS]
    extra = [s for s in load_extra_lexicon() if s not in set(base)]
    lexicon = base + extra
    print(f"Danh bạ tổng: {len(lexicon)} skill (gốc {len(base)} + ngoài {len(extra)})")
    scan_re = make_scan_re(lexicon)
    lex_set = set(lexicon)
    vec = fit(docs["text"])
    features = list(vec.get_feature_names_out())
    print(f"TF-IDF fit trên {len(docs)} doc (643 CV + 351 JD), vocab {len(features):,}")

    recs = extract_corpus(docs, vec, features, lex_set, scan_re)
    dump_jsonl([r for r in recs if r["doc_id"] in set(cv["doc_id"])], "cv")
    dump_jsonl([r for r in recs if r["doc_id"] in set(jd["doc_id"])], "jd")
    n_cv = sum(1 for r in recs if "name" in r)
    n_jd = len(recs) - n_cv
    print(f"\ndata/extracted/cv/     : {n_cv} file JSON")
    print(f"data/extracted/jd/     : {n_jd} file JSON")

    # ---- Bộ test ngoài -------------------------------------------------------
    tcv = pd.read_csv(os.path.join(CSV_DIR, "test_cv_docs.csv")).assign(doc_type="cv")
    tjd = pd.read_csv(os.path.join(CSV_DIR, "test_jd_docs.csv")).assign(doc_type="jd")
    tdocs = pd.concat([tcv, tjd], ignore_index=True)
    tdocs = tdocs[tdocs["text"].notna() & (tdocs["text"].astype(str).str.len() > 0)].copy()
    trecs = extract_corpus(tdocs, vec, features, lex_set, scan_re)
    dump_jsonl([r for r in trecs if r["doc_id"] in set(tcv["doc_id"])], "test_cv")
    dump_jsonl([r for r in trecs if r["doc_id"] in set(tjd["doc_id"])], "test_jd")
    print(f"data/extracted/test_cv/: {len(trecs) - sum(1 for r in trecs if 'name' in r)} "
          f"file JD + {sum(1 for r in trecs if 'name' in r)} file CV")

    # ---- Thống kê phủ các trường --------------------------------------------
    cvr = [r for r in recs if "name" in r]
    jdr = [r for r in recs if "job_title" in r]
    print(f"\n== Phủ trường trên corpus chính ==")
    print(f"CV  name            : {sum(1 for r in cvr if r['name']):6,} / {len(cvr)} "
          f"({sum(1 for r in cvr if r['name']) / len(cvr):.1%})")
    print(f"CV  experience_years: {sum(1 for r in cvr if r['experience_years'] is not None):6,} "
          f"({sum(1 for r in cvr if r['experience_years'] is not None) / len(cvr):.1%})")
    edu = cvr[0]["education"]
    print(f"CV  education.degree: {sum(1 for r in cvr if r['education']['degree']):6,} "
          f"({sum(1 for r in cvr if r['education']['degree']) / len(cvr):.1%})")
    print(f"CV  education.major : {sum(1 for r in cvr if r['education']['major']):6,} "
          f"({sum(1 for r in cvr if r['education']['major']) / len(cvr):.1%})")
    print(f"CV  education.univer: {sum(1 for r in cvr if r['education']['university']):6,} "
          f"({sum(1 for r in cvr if r['education']['university']) / len(cvr):.1%})")
    print(f"JD  job_title       : {sum(1 for r in jdr if r['job_title']):6,} / {len(jdr)} "
          f"({sum(1 for r in jdr if r['job_title']) / len(jdr):.1%})")
    print(f"JD  required_skills : TB {sum(len(r['required_skills']) for r in jdr) / len(jdr):.1f} skill")
    print(f"JD  preferred_skills: TB {sum(len(r['preferred_skills']) for r in jdr) / len(jdr):.1f} skill")
    print(f"JD  required_years  : {sum(1 for r in jdr if r['required_years'] is not None):6,} "
          f"({sum(1 for r in jdr if r['required_years'] is not None) / len(jdr):.1%})")

    # ---- Mẫu -----------------------------------------------------------------
    print("\n--- Mẫu CV JSON ---")
    print(json.dumps(cvr[0], ensure_ascii=False, indent=2)[:900])
    print("\n--- Mẫu JD JSON ---")
    print(json.dumps(jdr[0], ensure_ascii=False, indent=2)[:900])

    heldout_eval(cv, load_extra_lexicon(quiet=True))


if __name__ == "__main__":
    main()
