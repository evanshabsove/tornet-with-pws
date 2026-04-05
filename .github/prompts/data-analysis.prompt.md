Looking at your MADIS data exploration notebook, you've done excellent foundational work! Here are some concrete next steps for data processing:

## 1. **Temporal Alignment Analysis** ⏰
You're currently using a 10-minute window for matching, but you haven't analyzed:
- Distribution of time differences between radar scans and MADIS observations
- Whether certain storm phases have better/worse MADIS coverage
- If time lag affects prediction accuracy

**Suggested analysis:**
```python
# Add to notebook:
# Calculate actual time differences between radar and nearest MADIS observation
# Plot histogram of time lags
# Analyze: Does tornado detection degrade with larger time gaps?
```

## 2. **Spatial Coverage Analysis** 🗺️
Your notebook doesn't show:
- How far are weather stations from storm centers?
- Are you using nearest station only, or averaging multiple nearby stations?
- Geographic gaps in MADIS coverage (regions with sparse stations)

**Key question:** Should you weight MADIS readings by distance from storm?

## 3. **Feature Engineering Opportunities** 🔧
You're only using raw mean values. Consider adding:
- **Temporal derivatives**: Change in temperature/pressure over time (storm intensification signals)
- **Derived variables**: 
  - Temperature-dewpoint spread (instability indicator)
  - Wind shear (wind direction/speed differences)
  - Pressure tendency (rising/falling)
- **Variance metrics**: You extract std but don't use it - atmospheric turbulence signal?

## 4. **Missing Data Strategy** ❌ → ✅
Currently you're filtering out ~95%+ of samples. Better approaches:
- **Imputation**: Fill missing MADIS with regional/seasonal averages
- **Nearest temporal neighbor**: If exact time missing, use closest observation (already doing this, but could expand window)
- **Indicator variables**: Add binary flag "has_madis" as feature
- **Multi-task learning**: Train model to handle both with-MADIS and without-MADIS samples

## 5. **Coverage Analysis by Split** 📊
Critical for your experiment design:
```python
# Which train/val/test years have best MADIS coverage?
# Are you comparing apples-to-apples?
# If validation has poor MADIS coverage, baseline might look artificially good
```

You need to analyze:
- MADIS coverage % for train years (2013-2014) vs val years (2015)
- Class balance WITH MADIS vs WITHOUT MADIS
- Are tornado events more/less likely to have MADIS data?

## 6. **Feature Normalization/Scaling** 📏
Your final dataset has raw values, but your model uses `normalize_madis()`. Verify:
- Are the min-max ranges in `MADIS_MIN_MAX` correct for your actual data?
- Check for values outside expected ranges (you found outliers but didn't remove them)
- Consider robust scaling (median/IQR) instead of min-max if outliers present

## 7. **Temporal Aggregation Windows** 📈
You're using mean of all observations at a timestamp. Alternative strategies:
- **Moving averages**: 5-min, 10-min, 30-min windows
- **Trend features**: Is pressure rising or falling?
- **Time-weighted averages**: Newer observations weighted higher

## 8. **Multi-Station Fusion** 🛰️
When multiple stations exist at same timestamp, you're averaging. Consider:
- Keep closest N stations separately (spatial diversity)
- Maximum/minimum values (capture local extremes)
- Station consensus metrics (do all stations agree?)

## 9. **Quality Flags** ⚠️
Your outlier detection found issues but didn't act on them:
- Create quality score for each MADIS observation
- Filter or downweight low-quality readings
- Document which storms have questionable MADIS data

## 10. **Cross-Validation Strategy** 🔄
For proving MADIS helps, you need:
- **Matched pairs**: Only evaluate on samples where BOTH models (with/without MADIS) see the same storms
- **Stratified splits**: Ensure tornado/non-tornado balance in both conditions
- **Statistical testing**: Multiple runs with different seeds to prove significance

---

## Most Impactful Next Steps (Priority Order):

1. **Coverage analysis by train/val split** - Critical for experimental validity
2. **Expand missing data handling** - Increase usable dataset from ~19 to hundreds/thousands of samples
3. **Temporal alignment analysis** - Understand if your 10-min window is optimal
4. **Feature engineering** - Temperature-dewpoint spread is a proven tornado indicator
5. **Quality filtering** - Remove outliers identified in your analysis

Would you like me to create a new notebook cell with code for any of these analyses?
