# Plan: TorNet + MADIS Tornado Prediction Research

**Research Question**: Can personal weather station data from MADIS improve tornado prediction when combined with radar data?

**Approach**: Train baseline CNN on radar-only TorNet data (2013-2022), then develop fusion architecture integrating MADIS atmospheric features (7 variables: pressure, wind direction/speed/gust, humidity, temperature/dewpoint). Compare using accuracy, tornado recall, and precision.

---

## Steps

### Phase 1: Data Collection & Preparation (~2-3 weeks)
1. Download remaining TorNet years (2014-2022) from dataset source
2. Audit existing MADIS XML files - verify coverage for 2013 samples *(parallel with step 1)*
3. Download MADIS data for all TorNet samples (2013-2022) using your notebook download script
   - Modify loop to handle all years, add error logging
4. Generate data audit report
   - Sample counts per year (TOR/NUL/WRN breakdown)
   - MADIS coverage percentage
   - Identify gaps

### Phase 2: MADIS Integration Refactoring (~1 week) 🔧
5. **Fix hardcoded path** in [tornet/data/loader.py](tornet/data/loader.py#L69)
   - Replace absolute path with: `os.path.join(data_root, 'madis_data', f'madis_data_{storm_id}_{timestamp}.xml')`
   
6. **Improve validation logic** in `extract_madis_features()`
   - Remove strict `== 0.0` check (wind can legitimately be zero)
   - Distinguish between missing vs zero values
   - Return partial features with validity mask instead of dropping samples
   
7. **Implement missing data strategy**
   - Recommended: Return features with `np.nan` for missing values, handle in normalization
   - Alternative: Mean imputation
   
8. Update dataloader batching to handle partial MADIS data gracefully

### Phase 3: Baseline Model Training (~2-3 days)
9. Train baseline CNN (radar-only, no MADIS)
   - Use [params.json](scripts/tornado_detection/config/params.json), update years: `train: 2013-2020, val: 2021-2022`
   - Command: `python scripts/tornado_detection/train_tornado_keras.py scripts/tornado_detection/config/params.json`
   
10. Evaluate baseline on test set
    - Metrics: accuracy, per-class precision/recall/F1 for TOR/NUL/WRN
    - Save results to `results/baseline_metrics.json`

### Phase 4: MADIS-Enhanced Model (~1-2 weeks) ⭐
11. Design fusion architecture *(reference: [ModelTraining_keras_with_cwop_data.ipynb](notebooks/ModelTraining_keras_with_cwop_data.ipynb))*
    - Radar branch: Existing CNN baseline (VGG + CoordConv)
    - MADIS branch: MLP `(7,) → Dense(32) → Dense(16)`
    - Fusion: Concatenate → classifier
    
12. Implement in `tornet/models/keras/cnn_madis_fusion.py`
    - Accept radar dict + `'madis'` key
    - Normalize using `MADIS_MIN_MAX` from [constants.py](tornet/data/constants.py)
    
13. Train MADIS-enhanced model with `use_madis_data=True`
    
14. Evaluate on test set, save to `results/madis_fusion_metrics.json`

### Phase 5: Comparison & Analysis (~3-5 days)
15. Statistical comparison
    - McNemar's test for significance
    - Per-class improvement breakdown
    
16. Error analysis
    - Identify where MADIS helps vs hurts
    - Feature distribution analysis
    
17. Research paper writing

---

## Verification Checkpoints

### Phase 1 Verification:
- `ls tornet_data/train/` shows 2013-2020 directories
- MADIS file count ≥ TorNet sample count
- Audit report generated with year/type breakdowns

### Phase 2 Verification:
- Run `pytest tests/test_tornet.py -k madis` (passes)
- Load 10 samples with `use_madis_data=True`, no path errors
- Verify batch shapes: radar `(batch, azimuth, range, tilt)`, MADIS `(batch, 7)`

### Phase 3 Verification:
- Model trains 15 epochs without errors
- Test accuracy documented (expect 70-85%)
- Checkpoint saved: `models/tornado_baseline_radar_only.keras`

### Phase 4 Verification:
- Model summary shows both branches and fusion point
- MADIS input receives `(batch_size, 7)` tensors correctly
- Training completes without dimension errors

### Phase 5 Verification:
- Comparison table: baseline vs MADIS metrics side-by-side
- Statistical test p-value computed
- At least 3 visualizations (confusion matrices, error distributions)

---

## Key Decisions

### Data Scope:
- ✅ All geographic areas (no urban filtering)
- ✅ Full dataset 2013-2022 (train: 2013-2020, val: 2021-2022)
- ✅ All sample types: TOR, NUL, WRN

### MADIS Integration:
- ✅ Refactor existing [tornet/data/loader.py](tornet/data/loader.py) (not notebook approach)
- ✅ Fix hardcoded path to use `TORNET_ROOT`
- ✅ Relax zero-value validation
- 🔄 **Decision needed**: Missing data strategy - masking vs imputation vs dropping (recommend masking)

### Architecture:
- ✅ Maintain baseline CNN for radar (proven)
- ✅ Add MLP branch for MADIS (start simple: 2-3 layers)
- ✅ Late fusion architecture
- 🔄 **Decision needed**: MADIS MLP depth - start simple, expand if needed

### Metrics:
- ✅ Overall accuracy + per-class precision/recall
- ✅ Focus on tornado recall (safety critical)
- ✅ Precision for false alarm reduction

**Timeline**: 4-6 weeks (assumes 10-15 hrs/week)

---

## Relevant Files

### Data Loading:
- [tornet/data/loader.py](tornet/data/loader.py) - MADIS integration (lines 66-89), `extract_madis_features()`, **FIX PATH LINE 69**
- [tornet/data/constants.py](tornet/data/constants.py) - `MADIS_MIN_MAX`, `MAIDS_VARIABLES` normalization ranges
- [tornet/data/keras/loader.py](tornet/data/keras/loader.py) - Backend-agnostic dataloader, MADIS batching logic (line 125)

### Models:
- [tornet/models/keras/cnn_baseline.py](tornet/models/keras/cnn_baseline.py) - Baseline radar-only architecture (reuse CNN branch)
- [tornet/models/keras/layers.py](tornet/models/keras/layers.py) - `CoordConv2D` implementation for spatial awareness
- [notebooks/ModelTraining_keras_with_cwop_data.ipynb](notebooks/ModelTraining_keras_with_cwop_data.ipynb) - Reference fusion architecture pattern

### Training:
- [scripts/tornado_detection/train_tornado_keras.py](scripts/tornado_detection/train_tornado_keras.py) - Main training script
- [scripts/tornado_detection/config/params.json](scripts/tornado_detection/config/params.json) - Hyperparameters, update `train_years`/`val_years`

### Notebook Code:
- [assignment-2.ipynb](assignment-2.ipynb) - MADIS download functions (`download_madis_data`, `get_bounding_box`, `extract_madis_features`)

---

## Known Issues to Address

### Critical Issues in Current Implementation:
1. **Hardcoded Path** ([loader.py#L69](tornet/data/loader.py#L69))
   ```python
   # Current (BROKEN):
   madis_file = f"/Users/evanshabsove/Documents/.../madis_data_{storm_id}_{ds['time'].values[0]}.xml"
   
   # Should be:
   madis_file = os.path.join(data_root, 'madis_data', f'madis_data_{storm_id}_{timestamp}.xml')
   ```

2. **Zero Value Handling**
   - Current code treats `0.0` as missing
   - Wind speed can legitimately be 0.0 m/s (calm conditions)
   - Causes unnecessary data loss

3. **All-or-Nothing Validation**
   - Returns `None` if any of 7 variables missing
   - Wastes samples with partial MADIS data
   - No imputation or masking strategy

4. **No Spatial Integration**
   - Doesn't account for multiple weather stations per radar sample
   - No distance weighting from storm center
   - No temporal interpolation between observation times

5. **Model Integration Gap**
   - Baseline CNN doesn't accept MADIS input
   - Requires custom fusion architecture
   - No documented pattern in baseline code

---

## Further Considerations

### 1. MADIS Spatial Handling
Current implementation doesn't account for multiple weather stations per radar sample. Future work could:
- Aggregate multiple PWS readings (mean, weighted by distance)
- Include station distance as additional feature
- Use attention mechanism over station observations

### 2. Temporal Alignment
MADIS observation times may not exactly match radar scan times. Consider:
- Time offset as model feature
- Temporal interpolation for observations
- Document max acceptable time delta

### 3. Data Quality Filtering
MADIS includes QC flags in XML (`QCD`, `QCA`, `QCR` attributes). Currently ignored. Consider:
- Filter by quality codes
- Include quality flags as features
- Ablation study: QC-filtered vs all data

### 4. Backend Flexibility
Plan assumes TensorFlow backend. If switching to PyTorch/JAX:
- Set `KERAS_BACKEND` environment variable
- Verify dataloader compatibility
- Minor syntax adjustments may be needed

### 5. Class Imbalance
TorNet has severe class imbalance (few TOR samples). Current config uses loss weights (`w0`, `w1`, `w2`, `wW`). Monitor if MADIS affects class balance or requires weight tuning.

---

## Critical Next Steps

1. **Start TorNet data download (2014-2022)** - this will take time, run in background
2. **Run MADIS download script on 2013 first** to test coverage and identify gaps
3. **Fix hardcoded path in loader.py** before scaling up MADIS downloads
4. **Test MADIS loading** with 10-20 samples to verify no errors
5. **Create data audit notebook** to track download progress and coverage

---

## Success Metrics

### Primary Metrics:
- **Overall Accuracy**: Multi-class accuracy improvement (baseline vs MADIS-enhanced)
- **Tornado Recall**: Minimize missed tornadoes (TOR class recall) - safety critical
- **Precision**: Minimize false alarms - practical deployment concern

### Secondary Metrics:
- Per-class F1 scores (TOR, NUL, WRN)
- Confusion matrix comparison
- Statistical significance (McNemar's test p-value < 0.05)

### Qualitative Analysis:
- Error case studies: where does MADIS help most?
- Feature importance: which MADIS variables contribute most?
- Atmospheric patterns: do certain conditions benefit from MADIS data?

### Research Paper Requirements:
- Methodology documentation
- Metrics with confidence intervals
- Ablation study (if time permits)
- Limitations discussion (PWS coverage, temporal alignment, data quality)
