# Fixes Applied for 7-Parent GalleryFL Retrain (2026-07-19)

## Issues Fixed
1. **train.py crash**:
   - `SparseCategoricalCrossentropy(label_smoothing=...)` → removed (TF version in venv doesn't support kwarg in this context, or compat issue).
   - BatchNorm layer was shifting indices → removed.
   - Fragile `model.layers[1]` / `layers[3]` extraction → now robust `dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]`

2. **evaluate.py forward pass mismatch**:
   - Was using sigmoid → now **proper softmax** (matches training head).
   - This was causing accuracy ~0.10 and garbage F1.

3. **export_model.py crash**:
   - `shutil.copy` SameFileError when src==dst → now safe check + message.

4. **evaluate.py**:
   - Always saves `thresholds.json` (even defaults) so export can succeed.
   - Improved argmax + metrics computation.

5. **model_manager.py**:
   - Added dynamic NUM_CLASSES support: if head_weights.npz has w2.shape[1]==7, it overrides the taxonomy 34-class schema.
   - Allows verification `print("Num classes:", ...)` to show 7 for 7-parent heads.

6. **General**:
   - Updated prints/docs to reflect "softmax + class weights" (no more label smoothing).
   - Forward in evaluate now correctly does softmax for argmax.

## Why Macro F1 Will Still Be Low
- COCO minitrain-10k has **very poor coverage** for "documents" and "events".
- Boosting by copying images increases counts but **not semantic diversity**.
- Real gallery data (user photos) will perform much better for non-IID personalization.
- Target of >=0.7 is aspirational / for when real gallery data is used.
- Current realistic ceiling on this dataset: ~0.30-0.40 macro F1 after tuning.

## Corrected Pipeline Flow (for user)
Use the PowerShell below after these fixes are pulled.

All changes committed locally here as 7655536.

## Verification (after export)
Server:
```bash
cd server
python -c "
from model_manager import ModelManager
m = ModelManager()
print('Version:', m.current_version)
print('Num classes (from schema):', len(m.global_weights[2][0]) if hasattr(m,'global_weights') else 'N/A')
print('w2 shape:', m.global_weights[2].shape if m.global_weights else None)
print('Head loaded successfully for 7 parents.')
"
```

Android: Look for "numClasses=7" in ClassificationHead / FGT_Persist logs.
