import pandas as pd
import numpy as np

# =========================
# LOAD DATA
# =========================
spot = pd.read_csv("NIFTY 50_Historical_PR_01092025to15122025.csv")
opt = pd.read_csv("OPTIDX_NIFTY_PE_01-Sep-2025_TO_01-Dec-2025.csv")

# =========================
# CLEAN
# =========================
spot.columns = spot.columns.str.strip()
opt.columns = opt.columns.str.strip()

spot["Date"] = pd.to_datetime(spot["Date"], format="%d %b %Y")
opt["Date"] = pd.to_datetime(opt["Date"], format="%d-%b-%Y")
opt["Expiry"] = pd.to_datetime(opt["Expiry"], format="%d-%b-%Y")

for c in ["Strike Price", "Close", "Underlying Value"]:
    opt[c] = pd.to_numeric(opt[c], errors="coerce")

# =========================
# SORT FOR ASOF MERGE
# =========================
spot = spot.sort_values("Date")
opt = opt.sort_values("Expiry")

# =========================
# GET LAST TRADING SPOT BEFORE EXPIRY
# =========================
opt = pd.merge_asof(
    opt,
    spot[["Date", "Close"]].rename(columns={"Close": "Spot_At_Expiry"}),
    left_on="Expiry",
    right_on="Date",
    direction="backward"
)

opt.drop(columns=["Date_y"], inplace=True)
opt.rename(columns={"Date_x": "Date"}, inplace=True)

# =========================
# REALIZED PAYOFF (PUT)
# =========================
opt["Realized_Payoff"] = np.maximum(
    opt["Strike Price"] - opt["Spot_At_Expiry"], 0
)

# =========================
# PREMIUM VIABILITY
# =========================
opt["PnL"] = opt["Realized_Payoff"] - opt["Close"]
opt["Viable"] = opt["Realized_Payoff"] >= opt["Close"]

# =========================
# FINAL DATASET
# =========================
final_dataset = opt[[
    "Date",
    "Expiry",
    "Strike Price",
    "Underlying Value",
    "Spot_At_Expiry",
    "Close",
    "Realized_Payoff",
    "PnL",
    "Viable"
]].dropna()

# =========================
# SAVE
# =========================
final_dataset.to_csv("nifty_pe_expiry_validation.csv", index=False)

print("Rows:", len(final_dataset))
print("Saved: nifty_pe_expiry_validation.csv")
