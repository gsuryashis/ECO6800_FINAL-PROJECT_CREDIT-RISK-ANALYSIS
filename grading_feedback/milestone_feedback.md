# Milestone Feedback

Project: P02 - Credit risk analysis
Repo: `gsuryashis/ECO6800_FINAL-PROJECT_CREDIT-RISK-ANALYSIS`
Milestone score locked: 10/20
Raw score before policy caps: 10/20
Band: `major_revision`
Reviewed at: 2026-05-09T01:52:26

This is the locked milestone evaluation for the May 6 milestone. The score is based on the latest repository snapshot available to the instructor review workflow when this feedback was generated.

## Graduating-Student Timeline

This team includes graduating student(s): Suryashis Ghosh (ma2024), Vignesh Kanagaraj (ma2024).
To help us meet the May 15 grade-publishing deadline from OAA, please aim to submit the final version by May 13 if possible, and no later than May 14, 2026 at 11:59 PM IST.

## Rubric Breakdown

- Charter lock: 4/4. instructor records show the charter is approved; an approved charter file was found; instructor approval was used as charter-lock evidence
- Source access proof: 2/4. some data/probe evidence was found, but the manifest source list is incomplete
- Baseline before sophistication: 4/4. `outputs/baseline_metric.json` is readable and contains a real metric/value
- Reproducible dry run: 0/4. `uv run main.py` fails from a fresh copy of the repo
- Metric schema readiness: 0/4. primary_metric.json invalid_json: Expecting value: line 5 column 13 (char 86)

## What To Fix Next

- Make source access easy to verify: include a probe file or script, list the source in `outputs/milestone_manifest.json`, and commit a small permitted fallback if the full source is too large/private.
- Make `uv run main.py` work from a fresh clone. If the full data is large, the script should still run on a committed sample or a clearly reproducible download path.
- `outputs/primary_metric.json` should be machine-checkable: include `metric_name`, `value`, `threshold`, and `passed`.

## Reproducibility Error Observed

The reviewer ran `uv run main.py` from a fresh copy of the repo. The relevant tail of the error was:

```text
                 ^^^^^^^^
  File "/Users/kush/.local/share/uv/python/cpython-3.14.4-macos-aarch64-none/lib/python3.14/json/encoder.py", line 444, in _iterencode
    yield from _iterencode_dict(o, _current_indent_level)
  File "/Users/kush/.local/share/uv/python/cpython-3.14.4-macos-aarch64-none/lib/python3.14/json/encoder.py", line 413, in _iterencode_dict
    yield from chunks
  File "/Users/kush/.local/share/uv/python/cpython-3.14.4-macos-aarch64-none/lib/python3.14/json/encoder.py", line 451, in _iterencode
    newobj = _default(o)
  File "/Users/kush/.local/share/uv/python/cpython-3.14.4-macos-aarch64-none/lib/python3.14/json/encoder.py", line 182, in default
    raise TypeError(f'Object of type {o.__class__.__name__} '
                    f'is not JSON serializable')
TypeError: Object of type bool is not JSON serializable
when serializing dict item 'passed'
```

## Final Phase Guidance

- First priority: make the project reproducible from a fresh clone with `uv run main.py`. Do this before adding more modeling complexity.
- Second priority: make the final metric parseable in `outputs/primary_metric.json` with a value, threshold, and pass/fail status.
- Make the data path boring and reliable: source proof, fallback/sample data, and README instructions should agree.
- This needs urgent repair. A simple, reproducible, well-explained project will score better than an ambitious project that cannot be run or verified.
- For the final submission, keep the repo as the source of truth: `README.md`, `CHARTER.md`, `main.py`, `outputs/`, `report.md`, and `AI_USAGE_LOG.md` should tell one consistent story.

Please treat this feedback as a way to make the final week calmer, not as a ceiling on the final project. A clear, reproducible, honestly interpreted final submission can still be strong.
