# Cost Semantics DSS Reproducibility Package

This package supports the manuscript:

Cost Semantics in Enterprise Planning Decision Support Systems: A Two-Layer Architecture

## Contents

- dss_cost_semantics_lp_evaluation.py: LP evaluation using PuLP/CBC, including baseline, sensitivity, and scaled stress-test runs.
- dss_cost_semantics_evaluation.py: earlier transparent baseline evaluator retained for audit history.
- dss_cost_semantics_sensitivity.py: earlier sensitivity evaluator retained for audit history.
- DSS_Cost_Semantics_Evaluation_Summary.csv: baseline model summary output.
- DSS_Cost_Semantics_Evaluation_Details.csv: period-level baseline output.
- DSS_Cost_Semantics_Sensitivity_Results.csv: sensitivity-analysis output.
- DSS_Cost_Semantics_Scaled_Stress_Test.csv: scaled 20-product, 10-location, 12-period stress-test output.
- DSS_Cost_Semantics_LP_Evaluation_Summary.csv: full LP summary output.
- DSS_Cost_Semantics_LP_Evaluation_Details.csv: full LP detail output.
- requirements.txt: minimal Python requirements.

## How to Run

From the package folder:

```bash
python dss_cost_semantics_lp_evaluation.py
```

The script writes CSV outputs to an `outputs` folder relative to the run directory.

## Data

No client, confidential, or proprietary operational data are included. All demand, cost, capacity, and penalty values are synthetic and created for research illustration.

## Notes for Journal Submission

Before submission, deposit this package in a public repository and replace the manuscript data availability placeholder with the final repository URL.
