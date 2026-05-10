# Milestone Feedback

Project: P02 - Credit risk analysis
Repo: `gsuryashis/ECO6800_FINAL-PROJECT_CREDIT-RISK-ANALYSIS`
Official milestone score after post-lock recovery: 14/20
Post-lock sanity-check score: 18/20
Band: `major_revision`
Reviewed at: 2026-05-09T19:52:36

This is the official milestone feedback after applying the post-lock recovery policy. The locked May 6 snapshot remains the baseline, but real, reproducible fixes made after lock can recover 50% of the lost milestone points.

## Score Recovery Applied

- Locked milestone score: 10/20
- Post-lock sanity-check score: 18/20
- Official milestone score: 14/20
- Formula: locked score + 50% of the post-lock improvement

## Graduating-Student Timeline

This team includes graduating student(s): Suryashis Ghosh (ma2024), Vignesh Kanagaraj (ma2024).
To help us meet the May 15 grade-publishing deadline from OAA, please aim to submit the final version by May 13 if possible, and no later than May 14, 2026 at 11:59 PM IST.

## Rubric Breakdown

- Charter lock: 4/4. instructor records show the charter is approved; an approved charter file was found; instructor approval was used as charter-lock evidence
- Source access proof: 2/4. some data/probe evidence was found, but the manifest source list is incomplete
- Baseline before sophistication: 4/4. `outputs/baseline_metric.json` is readable and contains a real metric/value
- Reproducible dry run: 4/4. `uv run main.py` succeeds and writes the required milestone outputs
- Metric schema readiness: 4/4. `outputs/primary_metric.json` is readable and machine-checkable

## Policy Notes

- post-lock recovery policy applied: locked score 10/20; post-lock sanity check 18/20; official score = 10 + 50% of (18 - 10) = 14/20

## What To Fix Next

- Make source access easy to verify: include a probe file or script, list the source in `outputs/milestone_manifest.json`, and commit a small permitted fallback if the full source is too large/private.

## Final Phase Guidance

- Make the data path boring and reliable: source proof, fallback/sample data, and README instructions should agree.
- The project is viable, but the final phase should start with cleanup. Close the mechanical gaps first, then deepen the analysis.
- For the final submission, keep the repo as the source of truth: `README.md`, `CHARTER.md`, `main.py`, `outputs/`, `report.md`, and `AI_USAGE_LOG.md` should tell one consistent story.

Please treat this feedback as a way to make the final week calmer, not as a ceiling on the final project. A clear, reproducible, honestly interpreted final submission can still be strong.
