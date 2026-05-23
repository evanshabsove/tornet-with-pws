# Vast.ai Deployment Guide

## Instance Selection

- **CPU**: 16+ cores (this workload is CPU-bound — data loading is the bottleneck, not GPU compute)
- **GPU**: 2080, Titan XP, or better with at least 8GB VRAM
- **Disk**: 250GB+ (TorNet data is ~155GB)
- **Image**: TensorFlow template

Avoid renting purely for GPU power — a cheaper instance with more CPU cores and NVMe disk will outperform a big GPU with few cores.

---

## Step 1: Prep Local Files

Build your MADIS CSVs locally before renting (saves time on the instance):

```bash
export TORNET_ROOT=/Users/evanshabsove/Documents/tornado_reserch_paper/tornet/tornet_data
python scripts/build_madis_features.py
python scripts/build_madis_eligible_catalog.py
```

Zip the code (excluding data):

```bash
cd /Users/evanshabsove/Documents/tornado_reserch_paper/tornet
zip -r tornet_code.zip . \
  --exclude "tornet_data/*" \
  --exclude ".git/*" \
  --exclude "__pycache__/*" \
  --exclude "*.pyc" \
  --exclude "results/*"
```


---

## Step 2: Rent an Instance on Vast.ai

1. Go to vast.ai → Search
2. Filter: 16+ CPU cores, 250GB+ disk, GPU with 8GB+ VRAM
3. Select the **TensorFlow** image template
4. Set disk to **250GB** in the rental dialog (default is 16GB — must change this)
5. Click Rent

---

## Step 3: Add Your SSH Key

In Vast.ai account settings, paste the contents of:

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the entire line including `ssh-ed25519` at the start and your email at the end.

---

## Step 4: Connect to the Instance

Vast.ai provides an SSH command in the dashboard. Use:

```bash
ssh -p <PORT> root@<IP_ADDRESS> -i ~/.ssh/id_ed25519
```

---

## Step 5: Upload Code and CSVs

From your **local machine** (new terminal tab):

```bash
cd /Users/evanshabsove/Documents/tornado_reserch_paper/tornet

scp -P <PORT> -i ~/.ssh/id_ed25519 tornet_code.zip root@<IP_ADDRESS>:/workspace/
scp -P <PORT> -i ~/.ssh/id_ed25519 tornet_data/madis_features_clean.csv root@<IP_ADDRESS>:/workspace/tornet_data/
scp -P <PORT> -i ~/.ssh/id_ed25519 tornet_data/catalog_madis_eligible.csv root@<IP_ADDRESS>:/workspace/tornet_data/
scp -P <PORT> -i ~/.ssh/id_ed25519 tornet_data/catalog.csv root@<IP_ADDRESS>:/workspace/tornet_data/
```

---

## Step 6: Set Up the Instance

Back in your SSH session:

```bash
mkdir -p /workspace/tornet_data
cd /workspace
unzip tornet_code.zip -d tornet
cd tornet

pip install tensorflow[and-cuda]==2.18.0 keras==3.6.0 --force-reinstall
pip install zenodo-get
pip install -r requirements/basic.txt
pip install .
```

---

## Step 7: Download TorNet Data from Zenodo

The instance has fast internet (~1Gbps) so downloading directly is much faster than uploading from your local machine:

```bash
export TORNET_ROOT=/workspace/tornet_data
export KERAS_BACKEND=tensorflow

python download_tornet_data.py
```

This takes ~20–30 minutes. To speed it up next time, download years in parallel:

```bash
python download_tornet_data.py --years 2013 2014 --tornet-data-dir /workspace/tornet_data &
python download_tornet_data.py --years 2015 2016 --tornet-data-dir /workspace/tornet_data &
python download_tornet_data.py --years 2017 2018 --tornet-data-dir /workspace/tornet_data &
python download_tornet_data.py --years 2019 2020 --tornet-data-dir /workspace/tornet_data &
python download_tornet_data.py --years 2021 2022 --tornet-data-dir /workspace/tornet_data &
wait
```

---

## Step 8: Verify GPU is Visible

Always check before starting a long run:

```bash
nvidia-smi  # check CUDA version

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# Should print: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

If you get a segfault, reinstall TensorFlow matched to your CUDA version:
- CUDA 12.x → `pip install tensorflow[and-cuda]==2.18.0 keras==3.6.0 --force-reinstall`

---

## Step 9: Run Training

```bash
export TORNET_ROOT=/workspace/tornet_data
export KERAS_BACKEND=tensorflow

# MADIS run
python scripts/tornado_detection/train_tornado_keras.py \
    scripts/tornado_detection/config/params_madis.json

# No-MADIS comparison run (run after MADIS completes)
python scripts/tornado_detection/train_tornado_keras.py \
    scripts/tornado_detection/config/params_no_madis_comparison.json
```

You can close your laptop once training starts — the instance keeps running.

---

## Step 10: Download Results

Only grab what you need — each run saves ~34GB of checkpoints but you only need the best model and metrics.

On the **instance**:

```bash
mkdir /workspace/results

# MADIS run
cp <madis_run_folder>/checkpoints/tornadoDetector_best.keras /workspace/results/madis_best.keras
cp <madis_run_folder>/history.csv /workspace/results/madis_history.csv
cp <madis_run_folder>/params.json /workspace/results/madis_params.json

# No-MADIS run
cp <no_madis_run_folder>/checkpoints/tornadoDetector_best.keras /workspace/results/no_madis_best.keras
cp <no_madis_run_folder>/history.csv /workspace/results/no_madis_history.csv
cp <no_madis_run_folder>/params.json /workspace/results/no_madis_params.json

```
ssh -p 60149 root@182.224.239.168 -L 8080:localhost:8080
From your **local machine**:

```bash
scp -P 60149 -i ~/.ssh/id_ed25519 -r root@182.224.239.168 :/workspace/results ./
```

---

## Step 11: Destroy the Instance

Once results are downloaded, destroy the instance in the Vast.ai dashboard to stop billing.

---

## Cost Estimate

| Phase | Time | Cost |
|---|---|---|
| Zenodo data download | ~30 min | ~$0.20 |
| No-MADIS training run | ~13 hrs | ~$2–4 |
| MADIS training run | ~13 hrs (or less with early stopping) | ~$2–4 |
| **Total** | | **~$4–8** |
