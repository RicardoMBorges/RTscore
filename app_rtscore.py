import io
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ============================================================
# Optional RDKit imports
# ============================================================
RDKit_AVAILABLE = True
RDKit_IMPORT_ERROR = None
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Draw, Lipinski, Crippen, rdMolDescriptors
except Exception as e:  # pragma: no cover
    RDKit_AVAILABLE = False
    RDKit_IMPORT_ERROR = str(e)


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="RTscore for plausibility for Liquid Chromatography",
    page_icon="🧪",
    layout="wide",
)


# ============================================================
# Constants
# ============================================================
APP_TITLE = "RTscore for plausibility for LC"
APP_SUBTITLE = (
    "Candidate structure filtering based on consistency between molecular "
    "descriptors and observed retention time."
)

REQUIRED_REFERENCE_COLUMNS = ["name", "smiles", "rt"]
OPTIONAL_REFERENCE_COLUMNS = ["class", "adduct", "mode"]

REQUIRED_CANDIDATE_COLUMNS = ["feature_id", "candidate_name", "smiles"]
OPTIONAL_CANDIDATE_COLUMNS = ["observed_rt", "candidate_class", "adduct", "mode", "rank_source"]

DEFAULT_DESCRIPTOR_SET = [
    "MolWt",
    "ExactMolWt",
    "HeavyAtomCount",
    "TPSA",
    "MolLogP",
    "HBD",
    "HBA",
    "RingCount",
    "AromaticRingCount",
    "RotatableBonds",
    "FractionCSP3",
    "FormalCharge",
    "AtomCount",
    "HeteroAtomCount",
]

DEFAULT_MODEL_CHOICE = "Weighted descriptor score"
MODEL_OPTIONS = [
    "Weighted descriptor score",
    "Linear regression (numpy)",
]

DEFAULT_WEIGHTS = {
    "MolLogP": 1.60,
    "TPSA": -0.022,
    "HBD": -0.10,
    "HBA": -0.03,
    "RotatableBonds": 0.04,
    "RingCount": 0.05,
    "AromaticRingCount": 0.03,
    "FractionCSP3": 0.10,
    "FormalCharge": -0.35,
    "MolWt": 0.001,
    "HeavyAtomCount": 0.010,
    "HeteroAtomCount": -0.025,
    "AtomCount": 0.004,
}

DEMO_REFERENCE_CSV = """name,smiles,rt,class,adduct,mode
Caffeine,Cn1cnc2n(C)c(=O)n(C)c(=O)c12,1.92,alkaloid,[M+H]+,positive
Quercetin,O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12,4.25,flavonoid,[M-H]-,negative
Rutin,OC[C@H]1O[C@@H](Oc2cc(O)c3c(c2)oc(-c2ccc(O)c(O)c2)c(O)c3=O)[C@H](O)[C@@H](O)[C@@H]1O,2.80,flavonoid,[M-H]-,negative
Luteolin,O=c1cc(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12,4.05,flavonoid,[M-H]-,negative
Apigenin,O=c1cc(-c2ccc(O)cc2)oc2cc(O)cc(O)c12,4.70,flavonoid,[M-H]-,negative
Naringenin,O=C1CC(c2ccc(O)cc2)Oc2cc(O)cc(O)c21,5.05,flavonoid,[M-H]-,negative
Chlorogenic acid,O=C(O)/C=C/c1ccc(O)c(O)c1O[C@@H]1C[C@](O)(C(=O)O)C[C@@H](O)[C@H]1O,2.30,phenolic_acid,[M-H]-,negative
Ferulic acid,COc1cc(/C=C/C(=O)O)ccc1O,3.40,phenolic_acid,[M-H]-,negative
Gallic acid,O=C(O)c1cc(O)c(O)c(O)c1,0.95,phenolic_acid,[M-H]-,negative
Kaempferol,O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12,4.50,flavonoid,[M-H]-,negative
"""

DEMO_CANDIDATES_CSV = """feature_id,candidate_name,smiles,observed_rt,candidate_class,adduct,mode,rank_source
F001,Quercetin,O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12,4.22,flavonoid,[M-H]-,negative,MS/MS_match
F001,Kaempferol,O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12,4.22,flavonoid,[M-H]-,negative,Formula_match
F001,Luteolin,O=c1cc(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12,4.22,flavonoid,[M-H]-,negative,Library_candidate
F002,Caffeine,Cn1cnc2n(C)c(=O)n(C)c(=O)c12,3.85,alkaloid,[M+H]+,positive,MS1_formula
F002,Naringenin,O=C1CC(c2ccc(O)cc2)Oc2cc(O)cc(O)c21,3.85,flavonoid,[M-H]-,negative,Library_candidate
F003,Chlorogenic acid,O=C(O)/C=C/c1ccc(O)c(O)c1O[C@@H]1C[C@](O)(C(=O)O)C[C@@H](O)[C@H]1O,2.35,phenolic_acid,[M-H]-,negative,Library_candidate
F003,Ferulic acid,COc1cc(/C=C/C(=O)O)ccc1O,2.35,phenolic_acid,[M-H]-,negative,Formula_match
"""


# ============================================================
# Helpers
# ============================================================
def info_box(text: str) -> None:
    st.info(text)


def warn_box(text: str) -> None:
    st.warning(text)


def success_box(text: str) -> None:
    st.success(text)


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    attempts = [
        {"sep": None, "engine": "python"},
        {"sep": ";", "engine": "python", "encoding": "utf-8-sig"},
        {"sep": ";", "engine": "python", "encoding": "latin1"},
        {"sep": ",", "engine": "python", "encoding": "utf-8-sig"},
        {"sep": ",", "engine": "python", "encoding": "latin1"},
    ]

    last_error = None

    for kwargs in attempts:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, **kwargs)
            if df.shape[1] > 1:
                return df
        except Exception as e:
            last_error = e

    raise ValueError(
        "Could not parse this CSV file. Please check the separator, encoding, and malformed rows."
    ) from last_error


@st.cache_data(show_spinner=False)
def load_demo_csv(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


# ============================================================
# RDKit utilities
# ============================================================
def mol_from_smiles(smiles: str):
    if not RDKit_AVAILABLE:
        return None
    if pd.isna(smiles):
        return None
    smiles = str(smiles).strip()
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def calculate_descriptors_for_mol(mol) -> Dict[str, float]:
    if mol is None:
        return {k: np.nan for k in DEFAULT_DESCRIPTOR_SET}

    try:
        formal_charge = Chem.GetFormalCharge(mol)
    except Exception:
        formal_charge = np.nan

    descriptor_values = {
        "MolWt": Descriptors.MolWt(mol),
        "ExactMolWt": Descriptors.ExactMolWt(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "MolLogP": Crippen.MolLogP(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RingCount": Lipinski.RingCount(mol),
        "AromaticRingCount": Lipinski.NumAromaticRings(mol),
        "RotatableBonds": Lipinski.NumRotatableBonds(mol),
        "FractionCSP3": Lipinski.FractionCSP3(mol),
        "FormalCharge": formal_charge,
        "AtomCount": mol.GetNumAtoms(),
        "HeteroAtomCount": rdMolDescriptors.CalcNumHeteroatoms(mol),
    }
    return descriptor_values


@st.cache_data(show_spinner=False)
def add_rdkit_fields(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    out = df.copy()
    mols = []
    canonical = []
    valid = []
    descriptor_rows = []

    for smi in out[smiles_col].astype(str):
        mol = mol_from_smiles(smi)
        mols.append(mol)
        valid.append(mol is not None)
        canonical.append(canonicalize_smiles(smi) if mol is not None else None)
        descriptor_rows.append(calculate_descriptors_for_mol(mol))

    desc_df = pd.DataFrame(descriptor_rows)
    out = pd.concat([out.reset_index(drop=True), desc_df.reset_index(drop=True)], axis=1)
    out["canonical_smiles"] = canonical
    out["rdkit_valid"] = valid
    out["_mol"] = mols
    return out


def mol_to_pil(mol, size: Tuple[int, int] = (350, 250)):
    if mol is None or not RDKit_AVAILABLE:
        return None
    try:
        return Draw.MolToImage(mol, size=size)
    except Exception:
        return None


# ============================================================
# Data preparation
# ============================================================
def validate_columns(df: pd.DataFrame, required_cols: List[str], label: str) -> List[str]:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"{label}: missing required columns: {missing}")
    return missing


def prepare_reference_df(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_columns(df)
    out = add_rdkit_fields(out, "smiles")
    out["rt"] = pd.to_numeric(out["rt"], errors="coerce")
    if "class" not in out.columns:
        out["class"] = "unknown"
    if "adduct" not in out.columns:
        out["adduct"] = "unknown"
    if "mode" not in out.columns:
        out["mode"] = "unknown"
    out = out[out["rdkit_valid"]].copy()
    out = out.dropna(subset=["rt"]).copy()
    out = out.reset_index(drop=True)
    return out


def prepare_candidates_df(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_columns(df)
    out = add_rdkit_fields(out, "smiles")
    if "observed_rt" in out.columns:
        out["observed_rt"] = pd.to_numeric(out["observed_rt"], errors="coerce")
    else:
        out["observed_rt"] = np.nan
    if "candidate_class" not in out.columns:
        out["candidate_class"] = "unknown"
    if "adduct" not in out.columns:
        out["adduct"] = "unknown"
    if "mode" not in out.columns:
        out["mode"] = "unknown"
    if "rank_source" not in out.columns:
        out["rank_source"] = "candidate"
    out = out[out["rdkit_valid"]].copy()
    out = out.reset_index(drop=True)
    return out


# ============================================================
# Modeling
# ============================================================
def zscore_series(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if sd == 0 or pd.isna(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / sd


def build_weighted_score(reference_df: pd.DataFrame, selected_descriptors: List[str]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = reference_df.copy()

    usable_weights = {
        k: v for k, v in DEFAULT_WEIGHTS.items()
        if k in selected_descriptors and k in df.columns
    }

    if not usable_weights:
        raise ValueError("No valid descriptors available for weighted score.")

    score = np.zeros(len(df), dtype=float)
    for desc, w in usable_weights.items():
        score += zscore_series(df[desc].astype(float)).values * w

    df["descriptor_score"] = score
    return df, usable_weights


def fit_linear_regression_numpy(reference_df: pd.DataFrame, selected_descriptors: List[str]) -> Dict[str, np.ndarray]:
    model_df = reference_df.dropna(subset=selected_descriptors + ["rt"]).copy()
    X = model_df[selected_descriptors].astype(float).values
    y = model_df["rt"].astype(float).values

    if X.shape[0] < max(8, len(selected_descriptors) + 2):
        raise ValueError(
            "Not enough rows to fit the linear model. Reduce the number of descriptors or add more reference compounds."
        )

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    Xs = (X - X_mean) / X_std
    Xs_design = np.column_stack([np.ones(Xs.shape[0]), Xs])

    beta, *_ = np.linalg.lstsq(Xs_design, y, rcond=None)
    y_pred = Xs_design @ beta
    residuals = y - y_pred

    return {
        "beta": beta,
        "X_mean": X_mean,
        "X_std": X_std,
        "descriptors": np.array(selected_descriptors),
        "fitted_y": y_pred,
        "residuals": residuals,
        "train_index": model_df.index.values,
    }


def predict_linear_regression_numpy(df: pd.DataFrame, model: Dict[str, np.ndarray]) -> np.ndarray:
    descriptors = list(model["descriptors"])
    X = df[descriptors].astype(float).values
    Xs = (X - model["X_mean"]) / model["X_std"]
    Xs_design = np.column_stack([np.ones(Xs.shape[0]), Xs])
    y_pred = Xs_design @ model["beta"]
    return y_pred


def fit_descriptor_score_to_rt(reference_df: pd.DataFrame) -> Tuple[float, float, float]:
    score = reference_df["descriptor_score"].values
    rt = reference_df["rt"].values
    slope, intercept = np.polyfit(score, rt, 1)
    pred = slope * score + intercept
    residual_sd = float(np.std(rt - pred, ddof=1)) if len(reference_df) > 2 else 0.2
    return slope, intercept, residual_sd


def nearest_neighbor_distance(train_df: pd.DataFrame, target_df: pd.DataFrame, descriptors: List[str]) -> np.ndarray:
    X_train = train_df[descriptors].astype(float).values
    X_tgt = target_df[descriptors].astype(float).values

    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0

    X_train_s = (X_train - mu) / sd
    X_tgt_s = (X_tgt - mu) / sd

    distances = []
    for row in X_tgt_s:
        d = np.sqrt(((X_train_s - row) ** 2).sum(axis=1))
        distances.append(float(np.min(d)))
    return np.array(distances)


def classify_applicability(distance: float) -> str:
    if pd.isna(distance):
        return "unknown"
    if distance <= 1.5:
        return "inside"
    if distance <= 2.5:
        return "borderline"
    return "outside"


def classify_suspicion(score: float) -> str:
    if pd.isna(score):
        return "unknown"
    if score < 1.0:
        return "highly plausible"
    if score < 2.0:
        return "plausible"
    if score < 3.0:
        return "borderline"
    return "suspicious"


# ============================================================
# Main scoring logic
# ============================================================
def run_weighted_pipeline(reference_df: pd.DataFrame, candidates_df: pd.DataFrame, selected_descriptors: List[str]):
    ref_scored, used_weights = build_weighted_score(reference_df, selected_descriptors)
    slope, intercept, residual_sd = fit_descriptor_score_to_rt(ref_scored)

    ref_scored["rt_pred"] = slope * ref_scored["descriptor_score"] + intercept
    ref_scored["rt_residual"] = ref_scored["rt"] - ref_scored["rt_pred"]
    ref_scored["abs_residual"] = ref_scored["rt_residual"].abs()
    ref_scored["suspicion_score"] = ref_scored["abs_residual"] / max(residual_sd, 1e-6)
    ref_scored["suspicion_label"] = ref_scored["suspicion_score"].apply(classify_suspicion)

    cand = candidates_df.copy()
    cand_score = np.zeros(len(cand), dtype=float)
    for desc, w in used_weights.items():
        mu = ref_scored[desc].mean()
        sd = ref_scored[desc].std(ddof=0)
        if sd == 0 or pd.isna(sd):
            z = np.zeros(len(cand))
        else:
            z = (cand[desc].astype(float).values - mu) / sd
        cand_score += z * w

    cand["descriptor_score"] = cand_score
    cand["rt_pred"] = slope * cand["descriptor_score"] + intercept
    cand["abs_error_to_observed"] = (cand["observed_rt"] - cand["rt_pred"]).abs()
    cand["residual_sd_reference"] = residual_sd
    cand["suspicion_score"] = cand["abs_error_to_observed"] / max(residual_sd, 1e-6)
    cand["suspicion_label"] = cand["suspicion_score"].apply(classify_suspicion)
    cand["nn_distance"] = nearest_neighbor_distance(ref_scored, cand, selected_descriptors)
    cand["applicability"] = cand["nn_distance"].apply(classify_applicability)

    return {
        "reference": ref_scored,
        "candidates": cand,
        "weights": used_weights,
        "residual_sd": residual_sd,
        "equation": (slope, intercept),
        "model_name": "Weighted descriptor score",
    }


def run_linear_pipeline(reference_df: pd.DataFrame, candidates_df: pd.DataFrame, selected_descriptors: List[str]):
    model = fit_linear_regression_numpy(reference_df, selected_descriptors)

    ref = reference_df.copy()
    pred_ref = predict_linear_regression_numpy(ref, model)
    ref["rt_pred"] = pred_ref
    ref["rt_residual"] = ref["rt"] - ref["rt_pred"]
    residual_sd = float(np.std(ref.loc[model["train_index"], "rt_residual"], ddof=1)) if len(model["train_index"]) > 2 else 0.25
    ref["abs_residual"] = ref["rt_residual"].abs()
    ref["suspicion_score"] = ref["abs_residual"] / max(residual_sd, 1e-6)
    ref["suspicion_label"] = ref["suspicion_score"].apply(classify_suspicion)

    cand = candidates_df.copy()
    cand["rt_pred"] = predict_linear_regression_numpy(cand, model)
    cand["abs_error_to_observed"] = (cand["observed_rt"] - cand["rt_pred"]).abs()
    cand["residual_sd_reference"] = residual_sd
    cand["suspicion_score"] = cand["abs_error_to_observed"] / max(residual_sd, 1e-6)
    cand["suspicion_label"] = cand["suspicion_score"].apply(classify_suspicion)
    cand["nn_distance"] = nearest_neighbor_distance(reference_df, cand, selected_descriptors)
    cand["applicability"] = cand["nn_distance"].apply(classify_applicability)

    coef_table = pd.DataFrame({
        "descriptor": ["intercept"] + list(model["descriptors"]),
        "coefficient": model["beta"],
    })

    return {
        "reference": ref,
        "candidates": cand,
        "weights": coef_table,
        "residual_sd": residual_sd,
        "equation": None,
        "model_name": "Linear regression (numpy)",
    }


# ============================================================
# Plotting
# ============================================================
def plot_reference_distribution(reference_df: pd.DataFrame, selected_candidate_score: Optional[float] = None):
    fig = px.histogram(
        reference_df,
        x="suspicion_score",
        nbins=25,
        title="Reference suspicion score distribution",
    )
    fig.update_layout(xaxis_title="Suspicion score", yaxis_title="Count")

    if selected_candidate_score is not None and not pd.isna(selected_candidate_score):
        fig.add_vline(
            x=float(selected_candidate_score),
            line_width=3,
            line_dash="dash",
            annotation_text="Selected candidate",
        )
    return fig


def plot_rt_observed_vs_pred(df: pd.DataFrame, observed_col: str, title: str):
    fig = px.scatter(
        df,
        x=observed_col,
        y="rt_pred",
        hover_data=[c for c in ["name", "candidate_name", "feature_id", "suspicion_label"] if c in df.columns],
        title=title,
    )
    min_v = np.nanmin([df[observed_col].min(), df["rt_pred"].min()])
    max_v = np.nanmax([df[observed_col].max(), df["rt_pred"].max()])
    fig.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v)
    fig.update_layout(xaxis_title="Observed RT", yaxis_title="Predicted RT")
    return fig


def plot_feature_candidates(feature_df: pd.DataFrame, feature_id: str):
    sort_df = feature_df.sort_values(["suspicion_score", "nn_distance"], ascending=[True, True]).copy()
    sort_df["candidate_label"] = sort_df["candidate_name"].astype(str)

    fig = px.scatter(
        sort_df,
        x="rt_pred",
        y="suspicion_score",
        size="nn_distance",
        color="suspicion_label",
        hover_data=["candidate_name", "observed_rt", "applicability", "rank_source"],
        title=f"Candidate plausibility map for feature {feature_id}",
        text="candidate_label",
    )
    obs_rt = sort_df["observed_rt"].dropna()
    if not obs_rt.empty:
        fig.add_vline(x=float(obs_rt.iloc[0]), line_dash="dash", annotation_text="Observed RT")
    fig.update_traces(textposition="top center")
    fig.update_layout(xaxis_title="Predicted RT", yaxis_title="Suspicion score")
    return fig


def plot_feature_score_bars(feature_df: pd.DataFrame, feature_id: str):
    sort_df = feature_df.sort_values("suspicion_score", ascending=True).copy()
    fig = px.bar(
        sort_df,
        x="candidate_name",
        y="suspicion_score",
        color="suspicion_label",
        hover_data=["rt_pred", "observed_rt", "applicability", "nn_distance", "rank_source"],
        title=f"Candidate suspicion scores for feature {feature_id}",
    )
    fig.update_layout(xaxis_title="Candidate", yaxis_title="Suspicion score")
    return fig

# ============================================================
from PIL import Image

# Load the logo DAFdiscovery
logo = Image.open("static/RTscore.png")
# Display the logo in the sidebar or header
st.sidebar.image(logo, width=300)

# Load the logo LAABio
logo_LAABio = Image.open("static/LAABio.png")
# Display the logo in the sidebar or header
st.sidebar.image(logo_LAABio, width=300)

# ============================================================
# UI sections
# ============================================================
def sidebar_inputs():
    st.sidebar.header("Inputs")

    use_demo = st.sidebar.checkbox("Use built-in demo files", value=True)

    if use_demo:
        reference_df = load_demo_csv(DEMO_REFERENCE_CSV)
        candidates_df = load_demo_csv(DEMO_CANDIDATES_CSV)
        st.sidebar.caption("Demo reference and candidate files loaded.")
    else:
        ref_upload = st.sidebar.file_uploader(
            "Upload reference CSV",
            type=["csv"],
            help="Required columns: name, smiles, rt",
            key="reference_upload",
        )
        cand_upload = st.sidebar.file_uploader(
            "Upload candidate CSV",
            type=["csv"],
            help="Required columns: feature_id, candidate_name, smiles. Optional: observed_rt",
            key="candidate_upload",
        )
        reference_df = load_csv(ref_upload) if ref_upload is not None else None
        candidates_df = load_csv(cand_upload) if cand_upload is not None else None

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model settings")
    model_choice = st.sidebar.selectbox("Model", MODEL_OPTIONS, index=0)

    descriptor_set = st.sidebar.multiselect(
        "Descriptors",
        DEFAULT_DESCRIPTOR_SET,
        default=DEFAULT_DESCRIPTOR_SET,
    )

    show_only_valid = st.sidebar.checkbox("Filter to RDKit-valid structures only", value=True)
    run_button = st.sidebar.button("Run analysis", type="primary", use_container_width=True)

    st.sidebar.markdown("---")
    with st.sidebar.expander("CSV format examples"):
        st.code(DEMO_REFERENCE_CSV[:800], language="csv")
        st.code(DEMO_CANDIDATES_CSV[:800], language="csv")

    with st.sidebar.expander("How to cite / note"):
        st.write(
            "Use this app as an internal plausibility tool. RT consistency should support, not replace, exact mass, isotope pattern, and MS/MS evidence."
        )

    return {
        "reference_df": reference_df,
        "candidates_df": candidates_df,
        "model_choice": model_choice,
        "descriptor_set": descriptor_set,
        "show_only_valid": show_only_valid,
        "run_button": run_button,
        "use_demo": use_demo,
    }

def render_reference_overview(reference_df: pd.DataFrame):
    if reference_df is None or reference_df.empty:
        st.warning("No reference results available yet.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reference compounds", len(reference_df))
    c2.metric("Median RT", f"{reference_df['rt'].median():.2f}")
    c3.metric("Chemical classes", int(reference_df["class"].nunique()))
    c4.metric("Valid SMILES", int(reference_df["rdkit_valid"].sum()))

    st.dataframe(reference_df.drop(columns=["_mol"], errors="ignore"), use_container_width=True)

def render_candidates_overview(candidates_df: pd.DataFrame):
    if candidates_df is None or candidates_df.empty:
        st.warning("No candidate results available yet.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidate rows", len(candidates_df))
    c2.metric("Features", int(candidates_df["feature_id"].nunique()))
    c3.metric("Candidates / feature", f"{len(candidates_df) / max(candidates_df['feature_id'].nunique(), 1):.2f}")
    c4.metric("Rows with observed RT", int(candidates_df["observed_rt"].notna().sum()))

    st.dataframe(candidates_df.drop(columns=["_mol"], errors="ignore"), use_container_width=True)


def render_candidates_overview(candidates_df: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidate rows", len(candidates_df))
    c2.metric("Features", int(candidates_df['feature_id'].nunique()))
    c3.metric("Candidates / feature", f"{len(candidates_df) / max(candidates_df['feature_id'].nunique(), 1):.2f}")
    c4.metric("Rows with observed RT", int(candidates_df['observed_rt'].notna().sum()))

    st.dataframe(candidates_df.drop(columns=["_mol"], errors="ignore"), use_container_width=True)

def render_structure_gallery(feature_df: pd.DataFrame):
    st.subheader("Candidate structures")
    cols = st.columns(3)

    for idx, (_, row) in enumerate(feature_df.iterrows()):
        with cols[idx % 3]:
            st.markdown(f"**{row['candidate_name']}**")

            img = mol_to_pil(row["_mol"], size=(320, 220))
            if img is not None:
                st.image(img)

            pred_rt = row["rt_pred"] if pd.notna(row["rt_pred"]) else np.nan
            obs_rt = row["observed_rt"] if pd.notna(row["observed_rt"]) else np.nan

            pred_rt_text = f"{pred_rt:.2f}" if pd.notna(pred_rt) else "NA"
            obs_rt_text = f"{obs_rt:.2f}" if pd.notna(obs_rt) else "NA"

            st.caption(f"Pred RT: {pred_rt_text} | Obs RT: {obs_rt_text}")

            suspicion_label = row["suspicion_label"] if pd.notna(row["suspicion_label"]) else "unknown"
            applicability = row["applicability"] if pd.notna(row["applicability"]) else "unknown"
            suspicion_score = row["suspicion_score"] if pd.notna(row["suspicion_score"]) else np.nan
            suspicion_score_text = f"{suspicion_score:.2f}" if pd.notna(suspicion_score) else "NA"

            st.write(
                f"Suspicion: **{suspicion_label}**  \n"
                f"Applicability: **{applicability}**  \n"
                f"Score: **{suspicion_score_text}**"
            )

            st.code(str(row["smiles"]))
            
def build_download_csv(df: pd.DataFrame) -> bytes:
    export_df = df.drop(columns=["_mol"], errors="ignore").copy()
    return export_df.to_csv(index=False).encode("utf-8")


# ============================================================
# Main app
# ============================================================
def main():
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    with st.expander("About this app"):
        st.write(
            "This app evaluates whether a candidate structure is chromatographically plausible under a C18 LC-MS method by comparing observed RT against a descriptor-based expected RT."
        )
        if not RDKit_AVAILABLE:
            st.warning(f"RDKit is not available in this environment. Structure rendering and descriptor calculation are disabled. Error: {RDKit_IMPORT_ERROR}")

    ui = sidebar_inputs()

    reference_raw = ui["reference_df"]
    candidates_raw = ui["candidates_df"]
    selected_descriptors = ui["descriptor_set"]
    model_choice = ui["model_choice"]

    if reference_raw is None or candidates_raw is None:
        info_box("Upload both the reference CSV and the candidate CSV, or enable the demo files in the sidebar.")
        st.stop()

    reference_raw = normalize_columns(reference_raw)
    candidates_raw = normalize_columns(candidates_raw)

    missing_ref = validate_columns(reference_raw, REQUIRED_REFERENCE_COLUMNS, "Reference file")
    missing_cand = validate_columns(candidates_raw, REQUIRED_CANDIDATE_COLUMNS, "Candidate file")
    if missing_ref or missing_cand:
        st.stop()

    if not selected_descriptors:
        st.error("Select at least one descriptor.")
        st.stop()
        
    if ui["run_button"]:
        st.session_state["run_analysis"] = True
    
    if not st.session_state.get("run_analysis", False):
        info_box("Configure inputs in the sidebar and click **Run analysis**.")
        st.stop()

    with st.spinner("Preparing data..."):
        reference_df = prepare_reference_df(reference_raw)
        candidates_df = prepare_candidates_df(candidates_raw)

    if len(reference_df) < 5:
        st.error("The reference dataset is too small after cleaning. Add more validated compounds.")
        st.stop()

    usable_descriptors = [d for d in selected_descriptors if d in reference_df.columns and d in candidates_df.columns]
    if not usable_descriptors:
        st.error("None of the selected descriptors are available.")
        st.stop()

    with st.spinner("Running model..."):
        try:
            if model_choice == "Weighted descriptor score":
                results = run_weighted_pipeline(reference_df, candidates_df, usable_descriptors)
            else:
                results = run_linear_pipeline(reference_df, candidates_df, usable_descriptors)
        except Exception as e:
            st.exception(e)
            st.stop()

    ###
    if ui["run_button"]:
        with st.spinner("Preparing data..."):
            reference_df = prepare_reference_df(reference_raw)
            candidates_df = prepare_candidates_df(candidates_raw)

        if len(reference_df) < 5:
            st.error("The reference dataset is too small after cleaning. Add more validated compounds.")
            st.stop()

        usable_descriptors = [d for d in selected_descriptors if d in reference_df.columns and d in candidates_df.columns]
        if not usable_descriptors:
            st.error("None of the selected descriptors are available.")
            st.stop()

        with st.spinner("Running model..."):
            try:
                if model_choice == "Weighted descriptor score":
                    results = run_weighted_pipeline(reference_df, candidates_df, usable_descriptors)
                else:
                    results = run_linear_pipeline(reference_df, candidates_df, usable_descriptors)
            except Exception as e:
                st.exception(e)
                st.stop()

        st.session_state["reference_result"] = results["reference"]
        st.session_state["candidates_result"] = results["candidates"]
        st.session_state["model_name"] = results["model_name"]
        st.session_state["residual_sd"] = results["residual_sd"]
        st.session_state["results"] = results

    if "reference_result" not in st.session_state or "candidates_result" not in st.session_state:
        info_box("Configure inputs in the sidebar and click **Run analysis**.")
        st.stop()

    reference_result = st.session_state["reference_result"]
    candidates_result = st.session_state["candidates_result"]
    results = st.session_state["results"]

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview",
        "Reference model",
        "RT prediction",
        "Candidate plausibility",
        "Structures",
        "Export",
    ])

    with tab1:
        st.subheader("Input overview")
        st.markdown(f"**Model used:** {results['model_name']}")
        st.markdown(f"**Residual SD from reference set:** {results['residual_sd']:.3f} min")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Reference dataset")
            render_reference_overview(reference_result)
        with c2:
            st.markdown("### Candidate dataset")
            render_candidates_overview(candidates_result)

    with tab2:
        st.subheader("Reference model behavior")
        st.plotly_chart(
            plot_rt_observed_vs_pred(reference_result, "rt", "Reference compounds: observed RT vs predicted RT"),
            use_container_width=True,
        )

        if results["model_name"] == "Weighted descriptor score":
            slope, intercept = results["equation"]
            st.write(f"Weighted score equation to RT: **RT_pred = {slope:.4f} × score + {intercept:.4f}**")
            weights_df = pd.DataFrame({
                "descriptor": list(results["weights"].keys()),
                "weight": list(results["weights"].values()),
            }).sort_values("weight", ascending=False)
            st.dataframe(weights_df, use_container_width=True)
        else:
            st.dataframe(results["weights"], use_container_width=True)

        st.plotly_chart(
            px.scatter(
                reference_result,
                x="rt",
                y="rt_residual",
                color="class",
                hover_data=["name", "canonical_smiles"],
                title="Reference residuals by observed RT",
            ),
            use_container_width=True,
        )

    with tab3:
        st.subheader("Prediction view")

        st.plotly_chart(
            plot_rt_observed_vs_pred(
                candidates_result.dropna(subset=["observed_rt"]),
                "observed_rt",
                "Candidates: observed RT vs predicted RT",
            ),
            use_container_width=True,
        )

        sort_cols = ["feature_id", "suspicion_score"]
        if "nn_distance" in candidates_result.columns:
            sort_cols.append("nn_distance")

        pred_table = (
            candidates_result
            .sort_values(sort_cols, ascending=[True] * len(sort_cols))[
                [
                    "feature_id",
                    "candidate_name",
                    "observed_rt",
                    "rt_pred",
                    "abs_error_to_observed",
                    "suspicion_score",
                    "suspicion_label",
                    "applicability",
                    "rank_source",
                ]
            ]
        )

        st.dataframe(pred_table, use_container_width=True)
        

    with tab4:
        st.subheader("Candidate plausibility")
        feature_ids = sorted(candidates_result["feature_id"].dropna().astype(str).unique().tolist())
        selected_feature = st.selectbox("Select feature", feature_ids, key="feature_selector")
        feature_df = candidates_result[candidates_result["feature_id"].astype(str) == selected_feature].copy()
        feature_df = feature_df.sort_values(["suspicion_score", "nn_distance"], ascending=[True, True])

        if feature_df.empty:
            st.warning("No candidates available for the selected feature.")
        else:
            best_score = feature_df["suspicion_score"].iloc[0]
            st.plotly_chart(plot_reference_distribution(reference_result, best_score), use_container_width=True)
            st.plotly_chart(plot_feature_candidates(feature_df, selected_feature), use_container_width=True)
            st.plotly_chart(plot_feature_score_bars(feature_df, selected_feature), use_container_width=True)

            show_cols = [
                "feature_id", "candidate_name", "candidate_class", "rank_source", "observed_rt", "rt_pred",
                "abs_error_to_observed", "suspicion_score", "suspicion_label", "nn_distance", "applicability",
                "canonical_smiles"
            ]
            st.dataframe(feature_df[show_cols], use_container_width=True)

            top_hit = feature_df.iloc[0]
            st.success(
                f"Top candidate for feature {selected_feature}: {top_hit['candidate_name']} | "
                f"score = {top_hit['suspicion_score']:.2f} | label = {top_hit['suspicion_label']}"
            )

    with tab5:
        st.subheader("Structure viewer")
        feature_ids = sorted(candidates_result["feature_id"].dropna().astype(str).unique().tolist())
        selected_feature_struct = st.selectbox("Select feature for structures", feature_ids, key="feature_structure_selector")
        feature_df_struct = candidates_result[candidates_result["feature_id"].astype(str) == selected_feature_struct].copy()
        feature_df_struct = feature_df_struct.sort_values(["suspicion_score", "nn_distance"], ascending=[True, True])
        render_structure_gallery(feature_df_struct)

    with tab6:
        st.subheader("Export")
        export_reference = build_download_csv(reference_result)
        export_candidates = build_download_csv(candidates_result)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download reference results CSV",
                data=export_reference,
                file_name="reference_rt_plausibility_results.csv",
                mime="text/csv",
                key="download_reference_results",
            )
        with c2:
            st.download_button(
                "Download candidate results CSV",
                data=export_candidates,
                file_name="candidate_rt_plausibility_results.csv",
                mime="text/csv",
                key="download_candidate_results",
            )

        st.markdown("### Notes")
        st.write(
            "Recommended final use: combine exact mass, isotopic profile, fragmentation evidence, and RT plausibility in a single ranking workflow."
        )


if __name__ == "__main__":
    main()
