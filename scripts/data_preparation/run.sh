#!/bin/bash
# Prepare datasets for all supported benchmarks. Processed files are written
# under datasets/ (git-ignored); see README.md for details.
#
# The first block holds the nine real-world forecasting benchmarks used in the
# paper (Table 15); the remaining blocks cover the other datasets shipped with
# the codebase.

# ---------- Paper forecasting benchmarks (nine datasets) ----------
python scripts/data_preparation/ETTh1/generate_training_data.py
python scripts/data_preparation/ETTh2/generate_training_data.py
python scripts/data_preparation/ETTm1/generate_training_data.py
python scripts/data_preparation/ETTm2/generate_training_data.py
python scripts/data_preparation/Weather/generate_training_data.py
python scripts/data_preparation/Electricity/generate_training_data.py
python scripts/data_preparation/ExchangeRate/generate_training_data.py
# ExchangeRate needs a 6:2:2 resplit to support pred_len=720
python scripts/data_preparation/ExchangeRate/resplit_for_long_horizon.py
# ILI (influenza-like illness); the folder keeps the BasicTS name "Illness"
python scripts/data_preparation/Illness/generate_training_data.py
# ECG: downloads the MIT-BIH records and downsamples 360 Hz -> 36 Hz
python scripts/data_preparation/ECG/generate_training_data.py

# ---------- Synthetic benchmark: Synth-12 ----------
python scripts/data_preparation/SyntheticTS/generate_training_data.py

# ---------- Classification: 30 UEA datasets + HAR + Sleep-EDF ----------
python scripts/data_preparation/UEA/generate_training_data.py

# ---------- Additional long-term forecasting datasets ----------
python scripts/data_preparation/Traffic/generate_training_data.py
python scripts/data_preparation/BeijingAirQuality/generate_training_data.py
python scripts/data_preparation/GlobalTemp/generate_training_data.py
python scripts/data_preparation/GlobalWind/generate_training_data.py

# ---------- Spatial-temporal forecasting datasets ----------
python scripts/data_preparation/METR-LA/generate_training_data.py
python scripts/data_preparation/PEMS-BAY/generate_training_data.py
python scripts/data_preparation/PEMS03/generate_training_data.py
python scripts/data_preparation/PEMS04/generate_training_data.py
python scripts/data_preparation/PEMS07/generate_training_data.py
python scripts/data_preparation/PEMS08/generate_training_data.py
python scripts/data_preparation/JiNan/generate_training_data.py
python scripts/data_preparation/CA/generate_training_data.py
python scripts/data_preparation/GBA/generate_training_data.py
python scripts/data_preparation/GLA/generate_training_data.py
python scripts/data_preparation/SD/generate_training_data.py
python scripts/data_preparation/BLAST/merge_data.py

# ---------- Simulated datasets ----------
python scripts/data_preparation/Gaussian/generate_training_data.py
python scripts/data_preparation/Pulse/generate_training_data.py
