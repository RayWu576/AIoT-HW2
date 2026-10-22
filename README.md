# AIoT HW2 - Student Performance and Appliances Energy

The official DataLab dashboard extends the existing interface for exactly two problems:

| Problem | Samples | Original predictors | Full encoded matrix | Target |
| --- | ---: | ---: | ---: | --- |
| Student Performance (Mathematics) | 395 | 30 | 56 | G3, grade 0-20 |
| Appliances Energy | 19,735 | 25 | 25 | Appliances, Wh |

Student's authentic CSV has 33 columns; the early-prediction profile excludes G1/G2 and target G3. The requested Kaggle archive contains Portuguese data only, so the authentic Mathematics CSV uses the documented UCI source. Energy excludes date/rv1/rv2/target and retains lights; date stays in temporal metadata. See [schemas and acquisition](datasets/README.md).

## Local dashboard

```powershell
pip install -r requirements.txt
# Reuse the existing real FAST models and Phase 3 diagnostics; no repeated CV
python dashboard_export.py --mode FAST
# Equivalent current entry point
python main.py --mode FAST
python -m http.server 8000 --directory dashboard/dist
```

Open http://localhost:8000 or open dashboard/dist/index.html directly. No build tool or API server is required. [Phase 4 integration, inference contract and verification](PHASE4.md) explains the preserved UI, statistical limitations, and deployment. No arbitrary uploads are supported.

## Backend and tests

```powershell
# Run only when new experiments are required
python phase2_experiment.py --mode FAST --jobs 2
python phase3_evaluation.py --mode FAST
python dashboard_export.py --mode FAST
python -m unittest discover -s tests -p "test_*.py" -v
# Optional browser QA dependency and installed Edge
pip install playwright
python tests/verify_official_dashboard.py
```

FULL uses separate artifacts and more thorough CV; export FULL only after both genuine FULL experiments and diagnostics exist. FAST does not provide equivalent statistical evidence. [Phase 2](PHASE2.md) and [Phase 3](PHASE3.md) document actual results and methods. Their historical frontend-unchanged hash checks apply to those phases, not to the deliberate Phase 4 frontend changes.

## Static deployment

The existing .github/workflows/static.yml uploads dashboard/dist on an authorized main push or manual workflow dispatch. Set repository Settings > Pages > Source to GitHub Actions. Asset links are relative, including project-subpath hosting. Export current data before deployment. This change does not trigger or verify a public deployment; no commit or push is performed.

## Historical work

Boston Housing, 50 Startups and Car Price implementations and prior results are preserved as references, excluded from active official data. [Historical student/car documentation](HISTORICAL_STUDENT_CAR.md) retains the previous workflow; use the commands above for the current dashboard. The official registry is datasets.config.OFFICIAL_DATASETS; LEGACY_DATASETS keeps the historical student/car runner isolated.
