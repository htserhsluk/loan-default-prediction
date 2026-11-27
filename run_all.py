"""
Run the complete project pipeline end-to-end.

    python run_all.py                 # everything
    python run_all.py --from eda      # resume from a given step
    python run_all.py --only ablation # run a single step

Steps (in order):
  download     download the Lending Club file from Kaggle (skipped if present)
  prep         target, matured-vintage filter, leakage removal, sampling, split
  eda          exploratory data analysis figures / tables
  benchmark    train + evaluate all models, PCA analysis
  borrowers    repeat-borrower check (member_id is blank) and its effect on test AUC
  sampling     learning curve on the full population (sampling justification)
  ablation     Analysis A - feature analysis & ablation
  efficiency   Analysis C - model efficiency
  cost         Extension  - calibration, cost-sensitive policy, Head-of-Risk memo
  explain      Analysis F - interpretability & explainability (uses the calibrators)

Approximate run time on an 8-core laptop: 45-60 minutes.
"""
import argparse
import importlib
import sys
import time

STEPS = [
    ("download", "src.download_data"),
    ("prep", "src.data_prep"),
    ("eda", "src.eda"),
    ("benchmark", "src.train_benchmark"),
    ("borrowers", "src.borrower_check"),
    ("sampling", "src.sampling_check"),
    ("ablation", "src.analysis_ablation"),
    ("efficiency", "src.analysis_efficiency"),
    ("cost", "src.cost_sensitive"),
    ("explain", "src.analysis_explainability"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=[s for s, _ in STEPS])
    ap.add_argument("--only", choices=[s for s, _ in STEPS])
    args = ap.parse_args()

    names = [s for s, _ in STEPS]
    todo = STEPS
    if args.only:
        todo = [s for s in STEPS if s[0] == args.only]
    elif args.start:
        todo = STEPS[names.index(args.start):]

    t0 = time.time()
    for name, module in todo:
        print(f"\n{'=' * 70}\n>>> {name}  ({module})\n{'=' * 70}", flush=True)
        t = time.time()
        sys.argv = [module]                 # each step parses its own (default) args
        importlib.import_module(module).main()
        print(f">>> {name} finished in {(time.time() - t) / 60:.1f} min", flush=True)
    print(f"\nAll done in {(time.time() - t0) / 60:.1f} min.  "
          f"Launch the UI with:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
