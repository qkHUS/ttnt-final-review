# Final-project proposal / Đề xuất dự án cuối kỳ

Write in English or Vietnamese. Keep every section and replace every starter
placeholder. The commit containing this file and `team.json` is the authoritative
private proposal. The class topic issue is a public-in-class summary and record.

Before writing, read the [worked topic-proposal example](../examples/topic-proposal/README.md).
The example is fictional and illustrative; do not copy its identities, topic, or
text into a real submission.

## Project title / Tên dự án

<!-- BEGIN:PROJECT_TITLE -->

CV/Resume Parser and Job Matching System

<!-- END:PROJECT_TITLE -->

## Topic source / Nguồn chủ đề

<!-- BEGIN:TOPIC_SOURCE -->

This project is based on and adapts the CV/Resume parser idea listed in Mini-Project Ideas.md. The listed idea proposes reading CVs, identifying information such as work experience, education, and skills, and matching job descriptions with user skills and experiences.

<!-- END:TOPIC_SOURCE -->

## Short class-visible summary / Tóm tắt công khai trong lớp

This summary may be copied into the class topic issue. Do not include official
full names, student IDs, email addresses, phone numbers, credentials, private
repository details beyond the required repository URL, or unnecessary personal
data.

<!-- BEGIN:PUBLIC_SUMMARY -->

This project develops an AI-powered system that analyzes CVs and job descriptions, extracts relevant skills, and compares them to generate a matching score, matched skills, and missing skills.

<!-- END:PUBLIC_SUMMARY -->

## Exact selected problem / Bài toán cụ thể đã chọn

State one precise problem that distinguishes this project from projects in the
same broad area. The exact-duplicate check is performed on this problem, not on
an entire field, method, dataset family, or application domain.

<!-- BEGIN:SELECTED_PROBLEM -->

Given a user's CV and a job description, automatically extract relevant skills and experience from both documents and produce an interpretable CV-to-job matching result that identifies matched skills, missing skills, and an overall matching score.

<!-- END:SELECTED_PROBLEM -->

## Problem and motivation / Bài toán và động lực

<!-- BEGIN:PROBLEM_AND_MOTIVATION -->

Many job applications require comparing a CV with a job description. Doing this manually can be time-consuming.

This project aims to use NLP techniques to automatically extract skills from CVs and job descriptions and compare them. The system will produce a matching score and show matched and missing skills.

<!-- END:PROBLEM_AND_MOTIVATION -->

## Student-defined scope and feasibility / Phạm vi và tính khả thi do nhóm xác định

State what the group intends to complete, what it deliberately does not promise,
and why the plan is feasible with the available time and resources.

<!-- BEGIN:SCOPE_AND_FEASIBILITY -->

The project focuses on analyzing CVs and job descriptions in text format.

The minimum scope includes:

- Extract skills from a CV.
- Extract requirements from a job description.
- Compare skills and requirements.
- Calculate a matching score.
- Show matched and missing skills.
<!-- END:SCOPE_AND_FEASIBILITY -->

## Planned approach or method / Phương pháp dự kiến

<!-- BEGIN:PLANNED_METHOD -->

CV + Job Description → Text Processing → Skill Extraction → Matching → Score + Explanation

- Read the CV and job description.
- Clean and preprocess the text.
- Extract relevant skills and requirements.
- Compare the extracted information.
- Calculate a matching score.
- Display matched and missing skills.
- Evaluate the results using test cases.

<!-- END:PLANNED_METHOD -->

## Data, tools, papers, and existing projects / Dữ liệu, công cụ và tài liệu

Identify expected external resources and how they will be attributed, accessed,
and verified. Do not invent citations, URLs, DOI values, datasets, licenses,
metrics, or expected results.

<!-- BEGIN:DATA_TOOLS_AND_SOURCES -->

The project will use the resume-job-fit-merged-v1 dataset from Hugging Face, containing resume–job description pairs labeled as Good Fit, Potential Fit, or No Fit.

The main tools will be Python and NLP/machine-learning libraries for text preprocessing, skill extraction, similarity calculation, and model evaluation.

The project may use existing NLP models or open-source libraries when appropriate. Relevant research papers and documentation will be reviewed to guide the choice of methods.

<!-- END:DATA_TOOLS_AND_SOURCES -->

## Expected output or demonstration / Kết quả hoặc phần minh họa dự kiến

<!-- BEGIN:EXPECTED_OUTPUT -->

The final program will take a CV and a job description as input and display:

- Matching score.
- Matched skills.
- Missing skills.
- Basic explanation of the result.

The project will also provide source code, README, and evaluation results.

<!-- END:EXPECTED_OUTPUT -->

## Milestones / Các mốc thực hiện

<!-- BEGIN:MILESTONES -->
Milestone 1 — Data preparation

Obtain and inspect the CV/JD dataset.
Analyze labels and data distribution.
Clean and normalize CV and JD text.

Milestone 2 — Baseline matcher

Implement a basic keyword/TF-IDF based matching method.
Calculate an initial CV/JD similarity score.
Establish a baseline for comparison.

Milestone 3 — NLP semantic matching

Implement sentence/document embeddings.
Experiment with a transformer-based model.
Calculate semantic similarity between CVs and JDs.

Milestone 4 — Skill matching and explainability

Extract skills from CVs and JDs.
Identify matched and missing skills.
Combine semantic similarity and skill matching into an interpretable score.

Milestone 5 — Evaluation

Evaluate the system on the dataset.
Compare the proposed method with the baseline.
Analyze incorrect and difficult matching cases.

Milestone 6 — Final system

Build the final CV/JD matching pipeline.
Prepare a simple demonstration interface or script.
Complete documentation, README, and final presentation.
<!-- END:MILESTONES -->

## References / Tài liệu tham khảo

List only sources actually consulted. A proposal may state that no external
publication has yet been used, but it must not fabricate a source.

<!-- BEGIN:REFERENCES -->

The project will use the following source:

resume-job-fit-merged-v1 dataset from Hugging Face: https://huggingface.co/datasets/med2425/resume-job-fit-merged-v1

Additional research papers and technical documentation will be added after they are actually consulted and used in the project.

<!-- END:REFERENCES -->

## Ethical, privacy, safety, and licensing considerations / Đạo đức, riêng tư, an toàn và giấy phép

Discuss the considerations relevant to the group's own proposed scope. When none
is material, state that conclusion and briefly justify it.

<!-- BEGIN:CONSIDERATIONS -->

The project will use publicly available data and follow the dataset's licensing terms.

CVs may contain personal information, so the project will avoid using unnecessary real personal data and will prefer anonymized or publicly available examples.

<!-- END:CONSIDERATIONS -->

## Group self-check / Nhóm tự kiểm tra

Complete this checklist before creating the proposal commit. It is a group
self-assessment, not a teacher certification of quality, feasibility, or
correctness.

- [X] We have one to five members from the same course, and all listed members agree.
- [X] Only our founder accepted the Classroom50 assignment.
- [X] We reviewed the worked example but did not copy its identities, topic, or text.
- [X] We searched the correct class topic board for our exact repository URL and founder username.
- [X] Our selected problem is stated precisely and is not knowingly identical to an earlier recorded problem.
- [X] We chose and justified our own scope, method, resources, outputs, and level of ambition.
- [X] We believe the work is feasible by the common final deadline and will adjust it responsibly when needed.
- [X] We identified the external sources and dependencies currently expected.
- [X] We considered relevant integrity, privacy, licensing, safety, and lawful-use issues.
- [X] The proposal commit linked in the topic issue is the exact version agreed by every listed member.
