# Cost Semantics DSS Reproducibility Package

This package supports the manuscript:

Cost Semantics in Enterprise Planning Decision Support Systems: A Two-Layer Architecture

Package refresh: 2026-06-22 22:57 -04:00.

## Contents

- dss_cost_semantics_lp_evaluation.py: LP evaluation using PuLP/CBC, including baseline, sensitivity, and scaled stress-test runs.
- requirements.txt: minimal Python requirements.
- outputs/DSS_Cost_Semantics_Evaluation_Summary.csv: baseline model summary output.
- outputs/DSS_Cost_Semantics_Sensitivity_Results.csv: sensitivity-analysis output.
- outputs/DSS_Cost_Semantics_Scaled_Stress_Test.csv: scaled 20-product, 10-location, 12-period stress-test output.

## How to Run

From the package folder:

```bash
pip install -r requirements.txt
python dss_cost_semantics_lp_evaluation.py
```

The script writes the three CSV outputs to an `outputs` folder relative to the run directory.

The checked run used pandas 3.0.1, PuLP 3.3.2, and the CBC 2.10.3 solver bundled with PuLP.

## Data

No confidential, proprietary, or organization-identifying operational data are included. All demand, cost, capacity, and penalty values are synthetic and created for research illustration.

## Notes for Journal Submission

The public repository for this package is https://github.com/tejinderg123/cost-semantics-dss.
