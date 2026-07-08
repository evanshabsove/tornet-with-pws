# MADIS Data Exploration — Key Findings

Reference document summarizing `notebooks/madis_data_exploration.ipynb`.

---

## Dataset Overview

- **Total XML files**: 584,249 (T0 + temporal)
- **T0 files processed**: 224,514
- **Invalid/corrupted**: 85,099 (~38%)
- **Clean dataset**: 139,415 files, 2,408,181 observations

| Storm Type | Storms | Observations | Avg Stations/Storm |
|---|---|---|---|
| NUL | 224,078 | 3,665,318 | 2.57 |
| TOR | 12,023 | 180,254 | 2.43 |
| WRN | 91,144 | 1,451,011 | 2.50 |

---

## Spatial Coverage

- **Median station distance to storm**: 8.5 km
- **Within 10 km**: 82.3% of stations
- **Within 25 km**: 99.0%
- **Average stations per storm**: 2.7 (median 2.0, range 1–9)

Good proximity but **very sparse density** — most storms have only 2–3 stations. This averaging over a small number of noisy sensors is likely a limiting factor.

---

## Temporal Alignment

- **Within 5 min**: 14.7% of radar-MADIS pairs
- **Within 10 min**: 24.4%
- **Within 15 min**: 37.4%
- **Median absolute time gap**: 22 minutes

PWS report every 30 minutes, so most matches are 10–35 minutes away from the radar frame. Time lag has negligible correlation with data completeness (r = -0.04). The **actual matching window in `loader.py` is 15 minutes** (900 seconds) — samples with no MADIS observation within 15 minutes are rejected.

---

## Feature Predictive Power (Cohen's d, TOR vs NUL)

| Feature | Cohen's d | Strength |
|---|---|---|
| `pressure` (v_altse_mean) | -0.697 | **Strong** |
| `wind_gust` (v_ffgust_mean) | +0.529 | **Strong** |
| `wind_speed` (v_ff_mean) | +0.510 | **Strong** |
| `temperature` (v_t_mean) | -0.267 | Moderate |
| `relative_humidity` (v_rh_mean) | +0.242 | Moderate |
| `wind_direction` (v_dd_mean) | -0.182 | Weak |
| `dewpoint` (v_td_mean) | -0.116 | Weak |

Tornadoes are associated with **lower pressure** and **higher wind speeds** relative to non-tornadic events.

---

## Temporal Anomaly Features (Most Important Finding)

24-hour change features are **substantially more predictive** than raw values:

| Feature | Formula | Cohen's d | Strength |
|---|---|---|---|
| `pressure_anomaly_24h` | P_T0 − P_T24h | **1.388** | Very Large |
| `wind_anomaly_24h` | W_T0 − W_T24h | **0.665** | Large |
| `instability_proxy_T2h` | T_T2h − Td_T2h | **0.598** | Large |
| `instability_proxy_T0` | T_T0 − Td_T0 | 0.485 | Medium |

`pressure_anomaly_24h` (|d| = 1.388) is by far the strongest single predictor — nearly double the effect size of raw pressure. This captures the rapid pressure drop preceding tornadic events.

---

## Inter-Feature Correlations

- `wind_speed` ↔ `wind_gust`: r = **0.858** (strong — multicollinearity risk)
- `temperature` ↔ `dewpoint`: r = 0.622
- `temperature` ↔ `relative_humidity`: r = -0.580
- `wind_direction`: weakly correlated with everything (r < 0.06)

PCA: 5 components explain ~97% of variance — the 7 raw features contain significant redundancy.

---

## Tornado Intensity (EF Rating) Correlations

No strong correlations with EF rating (all |r| < 0.35). MADIS is better at **detecting** tornadoes than predicting their **intensity**.

---

## Temporal Matching Window Analysis (June 2026)

We investigated how much the temporal match window affects training data coverage.

**Two different metrics were being conflated:**
- **37.4% (per-observation)**: fraction of all individual MADIS observation timestamps within 15 min of the radar frame. One storm with readings at -5, +25, +55 min = 1/3 = 33%.
- **49.1% (per-catalog-row)**: fraction of catalog rows where the storm has *at least one* valid (pressure + wind_gust) MADIS observation within 15 min. The same storm passes because the -5 min reading qualifies.

**Coverage at different window sizes (full catalog = 203,133 rows):**

| Window | Eligible rows | % of catalog |
|---|---|---|
| 15 minutes (original) | 99,774 | 49.1% |
| No cutoff (±60 min download window) | 105,108 | 51.7% |

**Conclusion:** Removing the 15-minute cutoff only gains ~5,300 rows (+2.6%). PWS stations report on ~30-minute cycles, and if a station reported for a storm at all, there was usually a reading within 15 minutes of the radar frame anyway. **The real bottleneck is storm-level MADIS coverage (valid pressure + wind_gust), not the time window.** The architecture (late fusion) is the more likely performance bottleneck.

`loader.py` now uses no temporal cutoff. `build_madis_eligible_catalog.py` updated to match.

---

## Key Limitations

1. **30-minute reporting interval** — misses brief peak conditions during EF0-EF1 tornadoes (avg duration 4–8 min)
2. **Low station density** — averaging 2.7 noisy PWS sensors introduces noise
3. **~48% of catalog has no MADIS coverage** — storms without any valid pressure + wind_gust observation are excluded entirely from MADIS training runs
4. **Sensor saturation** — RH hits 100% in 1.4% of tornado events; wind speed at 0 m/s in 20% of readings
5. **No correlation with EF intensity** — MADIS can't distinguish strong from weak tornadoes

---

## Recommended Feature Sets

**Current model (6 features):** pressure, wind_gust, pressure_anomaly_24h, wind_anomaly_24h, instability_proxy_T2h, instability_proxy_T0

**Strongest 3 features only:** pressure_anomaly_24h (d=1.39), wind_anomaly_24h (d=0.67), instability_proxy_T2h (d=0.60)

**Raw values only (3 features):** pressure (d=0.70), wind_gust (d=0.53), wind_speed (d=0.51)

---

## Implications for Model Performance

The signal **exists** — particularly in the anomaly features. The likely reasons MADIS hasn't improved AUC in training runs:

1. **Late fusion architecture** gives MADIS features minimal influence over a 40K-dimensional flattened radar representation
2. **~48% of catalog has no MADIS coverage** — MADIS models train on a strict subset of storms, potentially biasing comparisons
3. **Station sparsity** (avg 2.7 stations) means the averaged MADIS values are noisy
4. **The CNN may already implicitly capture surface proxies** through low-level radar signatures

The anomaly features (especially `pressure_anomaly_24h`) have strong enough signal that a better fusion architecture — specifically intermediate fusion where MADIS can condition convolutional feature extraction — may show measurable improvement.
