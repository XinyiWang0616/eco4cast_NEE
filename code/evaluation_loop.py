#!/usr/bin/env python
# coding: utf-8

# In[3]:


from pathlib import Path
import pandas as pd
import numpy as np
import properscoring as ps
from properscoring import crps_gaussian


# In[4]:


# =========================
# paths
# =========================
FORECAST_DIR = Path("/home/xinyiw/ondemand/data/site_forecasts/")
TARGET_FILE = "/home/xinyiw/ondemand/data/target/targets_nee.csv"
CLIM_FILE = "/home/xinyiw/ondemand/data/baseline_model/climatology_pred_23_26.csv"
PRW_FILE = "/home/xinyiw/ondemand/data/baseline_model/persistenceRW_pred_23_26.csv"
EVA_DIR = Path("/home/xinyiw/ondemand/data/evaluation/")

START_DATE = pd.Timestamp("2024-01-01")
END_DATE = pd.Timestamp("2024-12-31")


# =========================
# load data
# =========================
targets = pd.read_csv(TARGET_FILE)
targets["datetime"] = pd.to_datetime(targets["datetime"]).dt.tz_localize(None)
targets = targets[targets["variable"] == "nee"].copy()

climatology_all = pd.read_csv(CLIM_FILE)
climatology_all["datetime"] = pd.to_datetime(climatology_all["datetime"]).dt.tz_localize(None)
climatology_all["reference_datetime"] = pd.to_datetime(climatology_all["reference_datetime"]).dt.tz_localize(None)

prw_all = pd.read_csv(PRW_FILE)
prw_all["datetime"] = pd.to_datetime(prw_all["datetime"]).dt.tz_localize(None)
prw_all["reference_datetime"] = pd.to_datetime(prw_all["reference_datetime"]).dt.tz_localize(None)


# =========================
# calculate crps
# =========================
def compute_xgb_crps(site_forecast, targets_site):
    df = site_forecast.copy()
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    df["reference_datetime"] = pd.to_datetime(df["reference_datetime"]).dt.tz_localize(None)

    merged = (
        df[df["model_id"] == "XGBoost"]
        .merge(targets_site, on=["datetime", "site_id", "variable"], how="left")
        .dropna(subset=["observation", "prediction"])
    )

    if merged.empty:
        return pd.DataFrame()

    def crps_one_group(g):
        obs = g["observation"].iloc[0]
        return ps.crps_ensemble(obs, g["prediction"].astype(float).values)

    out = (
        merged
        .groupby(["site_id", "model_id", "reference_datetime", "datetime", "variable"], as_index=False)
        .apply(lambda g: pd.Series({"crps": crps_one_group(g)}))
        .reset_index(drop=True)
    )

    out["horizon"] = (out["datetime"] - out["reference_datetime"]).dt.days
    return out


def compute_climatology_crps(site_id, targets_site):
    clim = climatology_all[
        (climatology_all["site_id"] == site_id) &
        (climatology_all["model_id"] == "climatology") &
        (climatology_all["reference_datetime"] >= START_DATE) &
        (climatology_all["reference_datetime"] <= END_DATE)
    ].copy()

    if clim.empty:
        return pd.DataFrame()

    merged = (
        clim
        .merge(targets_site, on=["datetime", "site_id", "variable"], how="left")
        .dropna(subset=["observation", "prediction"])
    )

    if merged.empty:
        return pd.DataFrame()

    summary = (
        merged
        .groupby(
            ["model_id", "site_id", "reference_datetime", "datetime", "variable", "observation", "parameter"],
            as_index=False
        )["prediction"]
        .mean()
    )

    pivot = summary.pivot(
        index=["model_id", "site_id", "reference_datetime", "datetime", "variable", "observation"],
        columns="parameter",
        values="prediction"
    ).reset_index()

    if ("mu" not in pivot.columns) or ("sigma" not in pivot.columns):
        return pd.DataFrame()

    pivot["mu"] = pd.to_numeric(pivot["mu"], errors="coerce")
    pivot["sigma"] = pd.to_numeric(pivot["sigma"], errors="coerce")
    pivot = pivot.dropna(subset=["mu", "sigma", "observation"])

    pivot["sigma"] = pivot["sigma"].clip(lower=1e-6)

    pivot["crps"] = crps_gaussian(
        pivot["observation"].astype(float).values,
        mu=pivot["mu"].values,
        sig=pivot["sigma"].values
    )

    pivot["horizon"] = (pivot["datetime"] - pivot["reference_datetime"]).dt.days

    return pivot[["site_id", "model_id", "reference_datetime", "datetime", "variable", "crps", "horizon"]]


def compute_persistence_crps(site_id, targets_site):
    prw = prw_all[
        (prw_all["site_id"] == site_id) &
        (prw_all["model_id"] == "persistenceRW") &
        (prw_all["reference_datetime"] >= START_DATE) &
        (prw_all["reference_datetime"] <= END_DATE)
    ].copy()

    if prw.empty:
        return pd.DataFrame()

    for col in ["mean", "sd"]:
        prw[col] = pd.to_numeric(prw[col], errors="coerce")

    merged = (
        prw
        .merge(targets_site, on=["datetime", "site_id", "variable"], how="left")
        .dropna(subset=["observation", "mean", "sd"])
    )

    if merged.empty:
        return pd.DataFrame()

    merged["sd"] = merged["sd"].clip(lower=1e-6)

    merged["crps"] = crps_gaussian(
        merged["observation"].astype(float).values,
        mu=merged["mean"].astype(float).values,
        sig=merged["sd"].astype(float).values
    )

    merged["horizon"] = (merged["datetime"] - merged["reference_datetime"]).dt.days

    return merged[["site_id", "model_id", "reference_datetime", "datetime", "variable", "crps", "horizon"]]


# =========================
# loop over site forecast files
# =========================
all_crps = []

forecast_files = sorted(FORECAST_DIR.glob("reforecast_*.csv"))

for f in forecast_files:
    site_id = f.stem.replace("reforecast_", "")
    print(f"Evaluating {site_id}")

    site_forecast = pd.read_csv(f)
    site_forecast["datetime"] = pd.to_datetime(site_forecast["datetime"]).dt.tz_localize(None)
    site_forecast["reference_datetime"] = pd.to_datetime(site_forecast["reference_datetime"]).dt.tz_localize(None)

    targets_site = targets[targets["site_id"] == site_id].copy()

    if targets_site.empty:
        print(f"{site_id}: skipped, no target observations")
        continue

    crps_xgb = compute_xgb_crps(site_forecast, targets_site)
    crps_cli = compute_climatology_crps(site_id, targets_site)
    crps_prw = compute_persistence_crps(site_id, targets_site)

    site_crps = pd.concat([crps_xgb, crps_cli, crps_prw], ignore_index=True)

    if not site_crps.empty:
        all_crps.append(site_crps)
        site_crps.to_csv(EVA_DIR / f"crps_{site_id}.csv", index=False)

if all_crps:
    crps_all_sites = pd.concat(all_crps, ignore_index=True)
else:
    crps_all_sites = pd.DataFrame()

crps_all_sites.to_csv("/home/xinyiw/ondemand/data/evaluation/crps_all_sites.csv", index=False)
print("success!")


# In[ ]:




