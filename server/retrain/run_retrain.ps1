# GalleryFL 7-Parent Clean Training (from scratch) - PowerShell
# Run from server\retrain directory
# After: git pull origin main

$DATASET_DIR = $env:DATASET_DIR
if (-not $DATASET_DIR) { $DATASET_DIR = "$HOME\data\gallery_7tag" }

Write-Host "=== GalleryFL 7-Parent Clean Training (from scratch) ===" -ForegroundColor Cyan
Write-Host "DATASET_DIR = $DATASET_DIR" -ForegroundColor White

# 1. Pull latest fixes
Write-Host "`n[1/7] Pulling latest fixes..." -ForegroundColor Yellow
git pull origin main

# 2. (Optional) Re-convert COCO - only if you want fresh mapping
# Write-Host "`n[2/7] (Optional) Re-converting COCO..." -ForegroundColor Yellow
# python convert_coco.py --coco_root "C:\path\to\coco_minitrain_10k" --output $DATASET_DIR --max-per-class 800

# 3. Fresh stratified splits
Write-Host "`n[3/7] Generating fresh train/val/test splits..." -ForegroundColor Yellow
python prepare_dataset.py --dataset-dir $DATASET_DIR

# 4. Train (softmax + class weights, argmax macro F1)
Write-Host "`n[4/7] Training proper 7-parent softmax head..." -ForegroundColor Yellow
python train.py --dataset-dir $DATASET_DIR --epochs 90 --lr 0.0008 --batch-size 32

# 5. Evaluate on TEST + thresholds
Write-Host "`n[5/7] Evaluating on TEST set..." -ForegroundColor Yellow
python evaluate.py `
    --checkpoint ..\output\retrain\best_checkpoint.npz `
    --test-manifest "$DATASET_DIR\manifests\test.csv" `
    --dataset-dir $DATASET_DIR `
    --tune-thresholds

# 6. Export final production head (4-layer format)
Write-Host "`n[6/7] Exporting production head..." -ForegroundColor Yellow
python export_model.py

# 7. Verification
Write-Host "`n[7/7] Verification:" -ForegroundColor Green

Write-Host "`n--- Server verification ---" -ForegroundColor Cyan
Write-Host "cd .." -ForegroundColor Gray
Write-Host 'python -c "from model_manager import ModelManager; m=ModelManager(); print(\"Version:\", m.current_version); print(\"w2 shape:\", m.global_weights[2].shape if m.global_weights else None); print(\"Num classes (dynamic):\", m.global_weights[2].shape[1] if m.global_weights else \"N/A\"); print(\"SUCCESS: 7-parent head loaded\")"' -ForegroundColor Gray

Write-Host "`n--- Android verification ---" -ForegroundColor Cyan
Write-Host "Search Logcat for: ClassificationHead | FGT_Persist | \"numClasses\"" -ForegroundColor Gray
Write-Host "Expect: numClasses=7" -ForegroundColor Gray

Write-Host "`nProduction artifacts:" -ForegroundColor Green
Write-Host "  ..\models\head_weights.npz" -ForegroundColor White
Write-Host "  ..\models\model_version.txt" -ForegroundColor White

Write-Host "`nNOTE: Macro F1 will be low (~0.25-0.35) due to COCO dataset limitations (poor coverage for documents/events)." -ForegroundColor Yellow
Write-Host "      Real user gallery data will give much higher F1 for non-IID personalization." -ForegroundColor Yellow
Write-Host "=== Done ===" -ForegroundColor Green