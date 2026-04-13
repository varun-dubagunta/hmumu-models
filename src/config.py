"""
config.py
---------
Shared constants: feature columns, signal/background process names, 
colour palette, and file paths used across all scripts.
"""
from pathlib import Path

# ── Project root (one level above src/) ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR    = ROOT / "data"
MODEL_DIR   = ROOT / "models"
OUTPUT_DIR  = ROOT / "outputs"

# ── Raw input ─────────────────────────────────────────────────────────────────
RAW_DATA_FILE = DATA_DIR / "DNN_samples_v4.pkl"

# ── Intermediate artifacts written by train.py, read by evaluate.py ───────────
PREDICTIONS_FILE = DATA_DIR / "predictions_with_metadata.pkl"
SCALER_FILE      = MODEL_DIR / "scaler.pkl"
BEST_MODEL_FILE  = MODEL_DIR / "best_transformer_model.keras"
FINAL_MODEL_FILE = MODEL_DIR / "final_transformer_model.keras"

# ── Physics features fed into the DNN ────────────────────────────────────────
DATA_COLUMNS = [
    "mu1_eta", "mu1_pt","mu1_phi",
    "mu2_eta", "mu2_pt","mu2_phi",
    "dR_mumu", "eta_mumu", "cosTheta_CS",
    "phi_mumu", "m_mumu","pt_mumu", "y_mumu", "phi_CS",
    "R_pt", "minDeltaEtaSigned", "minDeltaPhi",
    "Zepperfield_Var", "pt_centrality","j1_phi","j2_phi",
    "j1_eta", "j1_pt", "j1_btagPNetQvG",
    "j2_eta", "j2_pt", "j2_btagPNetQvG",
    "m_jj", "delta_eta_jj", "pt_jj",
    "nJet", "nSoftActivityJet",
    "SoftActivityJetHT", "SoftActivityJetHT2", "SoftActivityJetHT5",
    "SoftActivityJetHT10", "SoftActivityJetNjets2",
    "SoftActivityJetNjets5", "SoftActivityJetNjets10", "NN_Output",
]

# ── Physics token groupings for the Transformer ───────────────────────────────
TOKEN_GROUPS = {
    "mu1":     ["mu1_eta", "mu1_pt"],
    "mu2":     ["mu2_eta", "mu2_pt"],
    "dimu":    ["dR_mumu", "m_mumu", "eta_mumu", "cosTheta_CS",
                "phi_mumu", "pt_mumu", "y_mumu", "phi_CS",
                "R_pt", "Zepperfield_Var", "pt_centrality"],
    "jet1":    ["j1_eta", "j1_pt", "j1_btagPNetQvG"],
    "jet2":    ["j2_eta", "j2_pt", "j2_btagPNetQvG"],
    "dijet":   ["m_jj", "delta_eta_jj", "pt_jj"],
    "softjet": ["nJet", "nSoftActivityJet",
                "SoftActivityJetHT", "SoftActivityJetHT2",
                "SoftActivityJetHT5", "SoftActivityJetHT10",
                "SoftActivityJetNjets2", "SoftActivityJetNjets5",
                "SoftActivityJetNjets10"],
}

# ── Signal / background process names ────────────────────────────────────────
SIGNAL_PROCESSES     = ["VBFHto2Mu_m125_amcatnlo", "GluGluHto2Mu"]
BACKGROUND_PROCESSES = None  # resolved dynamically from data

# ── Process merging (applied before label encoding) ──────────────────────────
# Any process name in the keys will be renamed to its value before training.
# Add or remove entries here to merge processes across datasets.
PROCESS_REMAP = {
    "DY_M_10to50":   "BKG",
    "DYto2E_M_50":   "BKG",
    "DYto2Mu_M_50":  "BKG",
    "DYto2Tau_M_50": "BKG",
    "EWK":           "BKG",
    "TT":            "BKG",
    "VV":            "BKG",
}
# ── Plot colour palette ───────────────────────────────────────────────────────
BRIGHT_COLORS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00",
    "#FF00FF", "#00FFFF", "#FFA500", "#8A2BE2",
    "#FF69B4", "#00FA9A",
]
SIGNAL_COLORS = ["#00008B", "#6A0DAD"]

# ── Model hyper-parameters (Transformer) ─────────────────────────────────────
EMBEDDING_DIM  = 128
NUM_HEADS      = 16
DROPOUT_RATE   = 0.1
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4
BATCH_SIZE     = 512
EPOCHS         = 20
TEST_SIZE      = 0.2
RANDOM_STATE   = 42
FOCAL_GAMMA    = 1.5
FOCAL_ALPHA    = 0.25
USE_INVERSE_FREQUENCY = False

# ── Per-class yield targets for balanced sample weighting ────────────────────
# Lower VBF reduces gradient dominance. Higher DY = harder negative training.
# Set to None to use uniform 25000 for all classes.
CLASS_TARGET_YIELDS = {
    "VBFHto2Mu_m125_amcatnlo": 5000,
    "GluGluHto2Mu":            25000,
    "DY":                      50000,
    "EWK":                     25000,
    "TT":                      25000,
    "VV":                      25000,
}