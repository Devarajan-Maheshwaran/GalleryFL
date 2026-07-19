# ============================================================
# GalleryFL 7-Parent Retrain - FIXED VERSION (run this)
# Run from: server\retrain
# ============================================================

$DATASET_DIR = $env:DATASET_DIR
if (-not $DATASET_DIR) { $DATASET_DIR = "$HOME\data\gallery_7tag" }

Write-Host "=== GalleryFL 7-Parent Clean Training (FIXED - softmax + no label_smoothing) ===" -ForegroundColor Cyan
Write-Host "DATASET_DIR = $DATASET_DIR" -ForegroundColor White

# 1. Pull latest fixes from GitHub
Write-Host "`n[1/6] Pulling latest fixes..." -ForegroundColor Yellow
git pull origin main

# 2. (Optional) Only re-run if you want fresh data
# python convert_coco.py --coco_root "C:\path\to\coco-minitrain-10k" --output $DATASET_DIR --max-per-class 800

# 3. Fresh stratified splits
Write-Host "`n[2/6] Generating fresh splits..." -ForegroundColor Yellow
python prepare_dataset.py --dataset-dir $DATASET_DIR

# 4. Train with the FIXED head (softmax, no label_smoothing)
Write-Host "`n[3/6] Training 7-parent softmax head (FIXED)..." -ForegroundColor Yellow
python train.py --dataset-dir $DATASET_DIR --epochs 90 --lr 0.0008 --batch-size 32

# 5. Evaluate on TEST (now uses correct softmax forward)
Write-Host "`n[4/6] Evaluating on TEST set..." -ForegroundColor Yellow
python evaluate.py `
    --checkpoint ..\output\retrain\best_checkpoint.npz `
    --test-manifest "$DATASET_DIR\manifests\test.csv" `
    --dataset-dir $DATASET_DIR `
    --tune-thresholds

# 6. Export production head (bulletproof thresholds)
Write-Host "`n[5/6] Exporting production head..." -ForegroundColor Yellow
python export_model.py

# 7. Verification
Write-Host "`n[6/6] Verification commands:" -ForegroundColor Green

Write-Host "`n--- Server verification ---" -ForegroundColor Cyan
Write-Host "cd .." -ForegroundColor Gray
Write-Host 'python -c "from model_manager import ModelManager; m=ModelManager(); print(\"Version:\", m.current_version); print(\"w2 shape:\", m.global_weights[2].shape); print(\"Num classes:\", m.global_weights[2].shape[1]); print(\"SUCCESS - 7-parent head loaded\")"' -ForegroundColor Gray

Write-Host "`n--- Android ---" -ForegroundColor Cyan
Write-Host "Look for: ClassificationHead | FGT_Persist | numClasses=7" -ForegroundColor Gray

Write-Host "`nProduction files:" -ForegroundColor Green
Write-Host "  ..\models\head_weights.npz" -ForegroundColor White
Write-Host "  ..\models\model_version.txt" -ForegroundColor White

Write-Host "`n=== DONE. Macro F1 will be modest on COCO (expected). ===" -ForegroundColor Green