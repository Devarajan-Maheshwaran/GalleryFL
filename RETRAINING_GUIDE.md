# GalleryFL training guide

The canonical, audited seven-parent pipeline is documented in:

- [`server/retrain/README.md`](server/retrain/README.md)
- [`server/retrain/AUDIT_AND_CHANGES.md`](server/retrain/AUDIT_AND_CHANGES.md)

Quick PowerShell workflow from the repository root:

```powershell
$env:DATASET_DIR = "C:\path\to\gallery_7parent"
Set-Location .\server\retrain
python .\prepare_dataset.py --dataset-dir $env:DATASET_DIR
python .\train.py --dataset-dir $env:DATASET_DIR --epochs 100 --lr 0.0008 --batch-size 32
python .\evaluate.py --checkpoint ..\output\retrain\best_checkpoint.npz --test-manifest "$env:DATASET_DIR\manifests\test.csv" --dataset-dir $env:DATASET_DIR
if ($LASTEXITCODE -ne 0) { throw "Evaluation quality gate failed" }
python .\export_model.py
```

The central task is single-label seven-parent classification. Thresholds are
optional post-argmax confidence gates, not multi-label decision boundaries.
