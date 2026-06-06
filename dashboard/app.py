import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

st.set_page_config(
    page_title="Santé — Réadmission (multiclasse)", layout="wide")

DEFAULT_DATA_PATH = os.path.join("data", "processed", "dataset_model.csv")
TARGET_COL = "readmitted"

SCENARIOS = {
    "🔵 Conservateur (phase pilote)": {
        "N_pct":        10,
        "C_prog":       75.0,
        "C_readm":   3000.0,
        "r":             8,
        "description": "Phase pilote — 500 patients, coût 75 €/patient, réduction attendue 8 %.",
    },
    "🟢 Ambitieux (déploiement)": {
        "N_pct":        20,
        "C_prog":       50.0,
        "C_readm":   3000.0,
        "r":            15,
        "description": "Déploiement à l'échelle — 1 000 patients, coût 50 €/patient, réduction attendue 15 %.",
    },
    "⚙️ Personnalisé": {
        "N_pct":        10,
        "C_prog":       60.0,
        "C_readm":   3000.0,
        "r":            10,
        "description": "Paramètres libres.",
    },
}


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def infer_feature_types(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols):
    numeric = Pipeline(steps=[("scaler", StandardScaler())])
    categorical = Pipeline(steps=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
    )
    return pre


def kpi_block(df: pd.DataFrame):
    total = len(df)
    counts = df[TARGET_COL].value_counts(dropna=False)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nombre de séjours", f"{total:,}".replace(",", "\u202f"))
    c2.metric("Part <30",  f"{counts.get('<30',  0) / total * 100:.2f}%")
    c3.metric("Part >30",  f"{counts.get('>30',  0) / total * 100:.2f}%")
    c4.metric("Part NO",   f"{counts.get('NO',   0) / total * 100:.2f}%")


def plot_target_distribution(df: pd.DataFrame):
    vc = df[TARGET_COL].value_counts().reset_index()
    vc.columns = ["Classe", "Nombre"]
    fig = px.bar(
        vc, x="Classe", y="Nombre",
        title="Distribution de la cible (readmitted)",
        color="Classe",
        color_discrete_map={"<30": "#E74C3C",
                            ">30": "#E67E22", "NO": "#27AE60"},
    )
    st.plotly_chart(fig, use_container_width=True)


def safe_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Vrai {l}" for l in labels],
        columns=[f"Prédit {l}" for l in labels]
    )
    return cm_df


def compute_roi(n_targeted: int, baseline_lt30: float,
                r_pct: float, C_readm: float, C_prog: float) -> dict:
    evitees = n_targeted * baseline_lt30 * (r_pct / 100.0)
    economies = evitees * C_readm
    cout_prog = n_targeted * C_prog
    gain_net = economies - cout_prog
    roi = (gain_net / cout_prog * 100) if cout_prog > 0 else 0.0
    return {
        "evitees":   evitees,
        "economies": economies,
        "cout_prog": cout_prog,
        "gain_net":  gain_net,
        "roi":       roi,
    }


def plot_scenario_comparison(df: pd.DataFrame, baseline_lt30: float):
    rows = []
    for label, s in list(SCENARIOS.items())[:2]:
        n = int(len(df) * s["N_pct"] / 100)
        res = compute_roi(n, baseline_lt30, s["r"], s["C_readm"], s["C_prog"])
        rows.append({
            "Scénario":  label.split("(")[0].strip(),
            "Économies": res["economies"],
            "Coût programme": res["cout_prog"],
            "Gain net":  res["gain_net"],
            "ROI (%)":   res["roi"],
            "Réadmissions évitées": res["evitees"],
        })
    comp = pd.DataFrame(rows)

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        for i, row in comp.iterrows():
            fig.add_trace(go.Bar(
                name=row["Scénario"],
                x=["Économies", "Coût programme"],
                y=[row["Économies"], row["Coût programme"]],
                text=[f"{row['Économies']:,.0f} €",
                      f"{row['Coût programme']:,.0f} €"],
                textposition="outside",
            ))
        fig.update_layout(
            title="Économies vs Coût programme (€)",
            barmode="group", height=380,
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        colors = ["#E74C3C" if g < 0 else "#27AE60" for g in comp["Gain net"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=comp["Scénario"],
            y=comp["Gain net"],
            marker_color=colors,
            text=[f"{g:+,.0f} €" for g in comp["Gain net"]],
            textposition="outside",
            name="Gain net",
        ))
        fig.add_hline(y=0, line_dash="dot", line_color="grey")
        fig.update_layout(
            title="Gain net estimé (€)",
            height=380,
            yaxis_title="€",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Tableau comparatif — Business Case")
    disp = comp.copy()
    disp["Économies"] = disp["Économies"].apply(lambda v: f"{v:,.0f} €")
    disp["Coût programme"] = disp["Coût programme"].apply(
        lambda v: f"{v:,.0f} €")
    disp["Gain net"] = disp["Gain net"].apply(lambda v: f"{v:+,.0f} €")
    disp["ROI (%)"] = disp["ROI (%)"].apply(lambda v: f"{v:+.1f} %")
    disp["Réadmissions évitées"] = disp["Réadmissions évitées"].apply(
        lambda v: f"{v:.1f}")
    st.dataframe(disp.set_index("Scénario"), use_container_width=True)


def plot_sensitivity(df: pd.DataFrame, baseline_lt30: float):
    r_values = [5, 8, 10, 12, 15, 20]
    pct_range = np.arange(2, 32, 1)
    n_range = (pct_range / 100 * len(df)).astype(int)

    fig = go.Figure()
    for r_val in r_values:
        gains = [
            compute_roi(n, baseline_lt30, r_val, 3000, 60)["gain_net"]
            for n in n_range
        ]
        fig.add_trace(go.Scatter(
            x=pct_range, y=gains,
            mode="lines",
            name=f"r = {r_val} %",
            line=dict(dash="dash" if r_val < 10 else "solid",
                      width=1.5 if r_val < 10 else 2.2),
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="red",
                  annotation_text="Seuil rentabilité", annotation_position="right")

    for label, s, marker in [
        ("Conservateur", SCENARIOS["🔵 Conservateur (phase pilote)"], "circle"),
        ("Ambitieux",    SCENARIOS["🟢 Ambitieux (déploiement)"],     "square"),
    ]:
        n = int(len(df) * s["N_pct"] / 100)
        g = compute_roi(n, baseline_lt30, s["r"], s["C_readm"], s["C_prog"])[
            "gain_net"]
        fig.add_trace(go.Scatter(
            x=[s["N_pct"]], y=[g],
            mode="markers+text",
            marker=dict(size=12, symbol=marker),
            text=[f" {label} ({g:+,.0f} €)"],
            textposition="middle right",
            showlegend=False,
        ))

    fig.update_layout(
        title="Analyse de sensibilité — Gain net selon N (%) et réduction r<br>"
              "<sup>p0 = baseline <30, C_readm = 3 000 €, C_prog = 60 €</sup>",
        xaxis_title="Patients ciblés (% du dataset)",
        yaxis_title="Gain net (€)",
        height=430,
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, use_container_width=True)


st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Vue",
    [
        "1) Direction — KPIs & Impact",
        "2) EDA — Cohortes",
        "3) Segments — Clustering",
        "4) Modèle — Performance",
        "5) Explicabilité — SHAP (placeholder)"
    ],
    index=0
)
st.sidebar.markdown("---")
data_path = st.sidebar.text_input(
    "Chemin dataset (CSV)", value=DEFAULT_DATA_PATH)
st.sidebar.caption(
    "Le CSV doit contenir une colonne cible `readmitted` avec: <30, >30, NO.")

if not os.path.exists(data_path):
    st.error(
        f"Fichier introuvable : {data_path}\n\n"
        f"Attendu par défaut : {DEFAULT_DATA_PATH}"
    )
    st.stop()

df = load_data(data_path)

if TARGET_COL not in df.columns:
    st.error(
        "Colonne cible manquante : `" + TARGET_COL + "`.\n"
        "Colonnes trouvées: " + str(list(df.columns)[:20]) + " ..."
    )
    st.stop()

st.title("Dashboard Santé — Réadmission hospitalière (multiclasse)")


if page.startswith("1)"):
    st.header("Vue Direction — KPIs & estimation d'impact")
    kpi_block(df)

    st.markdown("### Répartition des classes")
    plot_target_distribution(df)

    st.markdown("---")
    st.markdown("### Comparaison des scénarios — Business Case")
    st.caption(
        "Scénarios issus du Business Case (notebook) : "
        "**Conservateur** (phase pilote, 8 % de réduction) "
        "vs **Ambitieux** (déploiement, 15 % de réduction). "
        "Paramètres : C_readm = 3 000 €, p0 = taux <30 observé dans le dataset."
    )
    baseline_lt30 = (df[TARGET_COL] == "<30").mean()
    plot_scenario_comparison(df, baseline_lt30)

    # ── Analyse de sensibilité ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔬 Analyse de sensibilité")
    st.caption(
        "Gain net en fonction du % de patients ciblés et du taux de réduction r. "
        "C_prog interpolé à 60 €/patient."
    )
    plot_sensitivity(df, baseline_lt30)

    st.markdown("---")
    st.markdown("### ⚙️ Simulateur personnalisé")

    scenario_choice = st.selectbox(
        "Charger un scénario de référence",
        options=list(SCENARIOS.keys()),
        index=0,
    )
    s_defaults = SCENARIOS[scenario_choice]
    st.caption(s_defaults["description"])

    colA, colB, colC = st.columns(3)
    with colA:
        top_pct = st.slider(
            "Cibler top X% des patients (selon score P(<30))",
            1, 30, s_defaults["N_pct"],
        )
    with colB:
        cost_per_intervention = st.number_input(
            "Coût intervention / patient (€)",
            min_value=0.0, value=float(s_defaults["C_prog"]), step=10.0,
        )
    with colC:
        cost_readmission = st.number_input(
            "Coût moyen d'une réadmission (€)",
            min_value=0.0, value=float(s_defaults["C_readm"]), step=500.0,
        )

    assumed_rel_reduction = st.slider(
        "Réduction relative supposée de <30 chez les ciblés (%)",
        0, 30, s_defaults["r"],
    )

    n_targeted = int(len(df) * top_pct / 100)
    res = compute_roi(n_targeted, baseline_lt30,
                      assumed_rel_reduction, cost_readmission, cost_per_intervention)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients ciblés",
              f"{n_targeted:,}".replace(",", "\u202f"))
    c2.metric("Réadmissions évitées",     f"{res['evitees']:.1f}")
    c3.metric("Économies estimées",
              f"{res['economies']:,.0f} €".replace(",", "\u202f"))
    delta_color = "normal" if res["gain_net"] >= 0 else "inverse"
    c4.metric(
        "Gain net",
        f"{res['gain_net']:,.0f} €".replace(",", "\u202f"),
        delta=f"ROI {res['roi']:+.1f} %",
        delta_color=delta_color,
    )

    if res["gain_net"] < 0:
        st.info(
            "Le gain net est négatif à ces paramètres — phase pilote normale. "
            "Augmentez N ou r pour atteindre le seuil de rentabilité."
        )
    else:
        st.success(
            f"Programme rentable — gain net {res['gain_net']:+,.0f} € "
            f"pour un ROI de {res['roi']:+.1f} %."
        )

    st.info(
        "Pour affiner : utilisez le modèle final (page 4) pour calculer P(<30) "
        "par patient, puis remplacez top_pct par un seuil de score réel."
    )

elif page.startswith("2)"):
    st.header("EDA — Cohortes & comparaisons simples")
    kpi_block(df)

    st.markdown("### Filtres")
    classes = st.multiselect(
        "Classes",
        options=sorted(df[TARGET_COL].dropna().unique().tolist()),
        default=sorted(df[TARGET_COL].dropna().unique().tolist())
    )
    dff = df[df[TARGET_COL].isin(classes)].copy()

    st.markdown("### Variables numériques (si présentes)")
    num_cols, cat_cols = infer_feature_types(dff, TARGET_COL)
    if len(num_cols) == 0:
        st.warning("Aucune variable numérique détectée.")
    else:
        var = st.selectbox("Choisir une variable numérique", num_cols)
        fig = px.box(dff, x=TARGET_COL, y=var, points="outliers",
                     title=f"{var} par classe de réadmission")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Variables catégorielles (si présentes)")
    if len(cat_cols) == 0:
        st.warning("Aucune variable catégorielle détectée.")
    else:
        cvar = st.selectbox("Choisir une variable catégorielle", cat_cols)
        tmp = dff.groupby([cvar, TARGET_COL]).size().reset_index(name="count")
        fig = px.bar(tmp, x=cvar, y="count", color=TARGET_COL,
                     barmode="group", title=f"{cvar} vs {TARGET_COL}")
        st.plotly_chart(fig, use_container_width=True)

elif page.startswith("3)"):
    st.header("Segments — Clustering des patients")

    CLUSTER_DATA_PATH = os.path.join(
        "data", "processed", "dataset_model_with_clusters.csv"
    )

    if not os.path.exists(CLUSTER_DATA_PATH):
        st.warning(
            "📂 Fichier de clusters introuvable : `data/processed/dataset_model_with_clusters.csv`\n\n"
            "**Étape préalable :** exécutez le notebook `03_clustering.ipynb` "
            "pour générer ce fichier, puis relancez le dashboard."
        )
        st.info(
            "Le notebook effectue automatiquement :\n"
            "- Encodage & normalisation des features\n"
            "- Sélection du k optimal (Elbow + Silhouette)\n"
            "- Entraînement KMeans + labellisation des clusters\n"
            "- Sauvegarde du CSV enrichi"
        )
        st.stop()

    @st.cache_data(show_spinner=False)
    def load_cluster_data(path):
        return pd.read_csv(path)

    dfc = load_cluster_data(CLUSTER_DATA_PATH)

    if "cluster" not in dfc.columns:
        st.error(
            "Colonne `cluster` absente du CSV. Ré-exécutez `03_clustering.ipynb`.")
        st.stop()

    has_label = "cluster_label" in dfc.columns
    cluster_col = "cluster_label" if has_label else "cluster"

    kpi_block(dfc)
    n_clusters = dfc["cluster"].nunique()
    st.caption(
        f"**{n_clusters} clusters** identifiés — dataset `{CLUSTER_DATA_PATH}`")

    st.markdown("---")
    st.markdown("### Profil des clusters")

    dfc["_is_lt30"] = (dfc[TARGET_COL] == "<30").astype(int)
    summary = (
        dfc.groupby(cluster_col)
        .agg(
            n_patients=(TARGET_COL, "count"),
            taux_lt30=("_is_lt30", "mean"),
            taux_gt30=(TARGET_COL, lambda x: (x == ">30").mean()),
            taux_no=(TARGET_COL, lambda x: (x == "NO").mean()),
        )
        .reset_index()
    )
    summary["taux_lt30_%"] = (summary["taux_lt30"] * 100).round(1)
    summary["taux_gt30_%"] = (summary["taux_gt30"] * 100).round(1)
    summary["taux_no_%"] = (summary["taux_no"] * 100).round(1)
    summary["part_dataset_%"] = (
        summary["n_patients"] / len(dfc) * 100).round(1)
    summary = summary.sort_values("taux_lt30_%", ascending=False)

    disp_cols = [cluster_col, "n_patients", "part_dataset_%",
                 "taux_lt30_%", "taux_gt30_%", "taux_no_%"]
    st.dataframe(
        summary[disp_cols].rename(columns={
            cluster_col:      "Segment",
            "n_patients":     "Patients",
            "part_dataset_%": "Part (%)",
            "taux_lt30_%":    "Taux <30 (%)",
            "taux_gt30_%":    "Taux >30 (%)",
            "taux_no_%":      "Taux NO (%)",
        }).reset_index(drop=True),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("### Visualisations")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            summary.sort_values("taux_lt30_%", ascending=True),
            x="taux_lt30_%", y=cluster_col,
            orientation="h",
            title="Taux de réadmission <30j par segment",
            labels={"taux_lt30_%": "Taux <30 (%)", cluster_col: ""},
            color="taux_lt30_%",
            color_continuous_scale=["#27AE60", "#F39C12", "#E74C3C"],
            text="taux_lt30_%",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        melt = dfc.groupby([cluster_col, TARGET_COL]
                           ).size().reset_index(name="n")
        totals = melt.groupby(cluster_col)["n"].transform("sum")
        melt["pct"] = (melt["n"] / totals * 100).round(1)
        fig = px.bar(
            melt, x=cluster_col, y="pct", color=TARGET_COL,
            barmode="stack",
            title="Répartition des classes par segment (%)",
            labels={"pct": "%", cluster_col: "Segment", TARGET_COL: "Classe"},
            color_discrete_map={"<30": "#E74C3C",
                                ">30": "#E67E22", "NO": "#27AE60"},
        )
        fig.update_layout(height=380, xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

    fig = px.pie(
        summary, names=cluster_col, values="n_patients",
        title="Répartition des patients par segment",
        hole=0.4,
    )
    fig.update_traces(textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🖼️ Figures générées par le notebook")

    fig_paths = {
        "Elbow & Silhouette":   "reports/figures/elbow_silhouette.png",
        "Profils clusters":     "reports/figures/cluster_profiles.png",
        "Projection PCA":       "reports/figures/pca_clusters.png",
        "Heatmap clusters":     "reports/figures/cluster_heatmap.png",
    }
    available = {k: v for k, v in fig_paths.items() if os.path.exists(v)}
    missing = {k: v for k, v in fig_paths.items() if not os.path.exists(v)}

    if available:
        tabs = st.tabs(list(available.keys()))
        for tab, (title, path) in zip(tabs, available.items()):
            with tab:
                st.image(path, use_container_width=True)
    if missing:
        st.caption(
            "Figures non trouvées (re-run notebook) : "
            + ", ".join(f"`{v}`" for v in missing.values())
        )

    st.markdown("---")
    st.markdown("### 🔍 Explorer une variable par segment")

    num_cols_c, cat_cols_c = infer_feature_types(
        dfc.drop(columns=["_is_lt30"]), TARGET_COL)

    col_a, col_b = st.columns(2)
    with col_a:
        if num_cols_c:
            num_var = st.selectbox("Variable numérique",
                                   num_cols_c, key="clust_num")
            fig = px.box(
                dfc, x=cluster_col, y=num_var, color=cluster_col,
                points="outliers",
                title=f"{num_var} par segment",
            )
            fig.update_layout(showlegend=False, xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if cat_cols_c:
            cat_var = st.selectbox(
                "Variable catégorielle", cat_cols_c, key="clust_cat")
            tmp = (
                dfc.groupby([cluster_col, cat_var])
                .size()
                .reset_index(name="count")
            )
            fig = px.bar(
                tmp, x=cat_var, y="count", color=cluster_col,
                barmode="group",
                title=f"{cat_var} par segment",
            )
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
    dfc.drop(columns=["_is_lt30"], inplace=True)

elif page.startswith("4)"):
    st.header("Modèle — Entraînement rapide (sklearn) & performance")
    st.write(
        "Cette page entraîne un modèle **à la volée** (baseline) depuis `dataset_model.csv`.\n"
        "Pour le rendu final, vous pouvez :\n"
        "- entraîner dans le notebook\n"
        "- sauvegarder le modèle (joblib)\n"
        "- charger ici le modèle final."
    )

    test_size = st.slider("Taille test", 0.1, 0.4, 0.2, 0.05)
    model_name = st.selectbox(
        "Modèle",
        ["LogisticRegression (multinomial)", "RandomForest",
         "HistGradientBoosting"]
    )

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(str)
    num_cols, cat_cols = infer_feature_types(df, TARGET_COL)
    pre = build_preprocessor(num_cols, cat_cols)

    if model_name.startswith("Logistic"):
        clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    elif model_name.startswith("RandomForest"):
        clf = RandomForestClassifier(
            n_estimators=300, random_state=42,
            class_weight="balanced_subsample", n_jobs=-1)
    else:
        clf = HistGradientBoostingClassifier(random_state=42)

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)

    with st.spinner("Entraînement du modèle..."):
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

    labels = ["<30", ">30", "NO"]

    st.markdown("### Matrice de confusion")
    cm_df = safe_confusion_matrix(y_test, y_pred, labels=labels)
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("### Rapport de classification")
    rep = classification_report(
        y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).T
    st.dataframe(rep_df, use_container_width=True)

elif page.startswith("5)"):
    st.header("Explicabilité — SHAP")
    st.markdown("""
    Cette page présente les résultats d'interprétation du modèle obtenus
    dans le notebook **05_shap.ipynb**.
    """)

    shap_img = os.path.join("reports", "figures", "shap_global_summary.png")
    if os.path.exists(shap_img):
        st.subheader("Importance globale des variables")
        st.image(
            shap_img,
            caption="SHAP Summary Plot — Facteurs influençant la réadmission <30 jours",
            use_container_width=True
        )
    else:
        st.warning(
            "Figure SHAP introuvable. "
            "Exécutez le notebook puis vérifiez que "
            "`reports/figures/shap_global_summary.png` existe."
        )

    st.subheader("Principaux facteurs de risque identifiés")
    importance_df = pd.DataFrame({
        "Feature": [
            "number_inpatient",
            "number_emergency",
            "number_diagnoses",
            "discharge_disposition",
            "num_medications"
        ],
        "Interprétation": [
            "Hospitalisations précédentes fréquentes",
            "Passages répétés aux urgences",
            "Complexité médicale élevée",
            "Modalité de sortie du patient",
            "Poly-médication"
        ]
    })
    st.dataframe(importance_df, use_container_width=True)

    st.subheader("Recommandations opérationnelles")
    st.markdown("""
    ### 1. Programme post-sortie ciblé
    - Identifier les patients les plus à risque selon le score du modèle.
    - Déclencher automatiquement :
        - appel de suivi à J+2 ;
        - téléconsultation à J+7 ;
        - contrôle de l'observance thérapeutique.
    **Objectif :** réduire les réadmissions <30 jours.
    """)
    st.markdown("""
    ### 2. Expérimentation A/B Test
    - Groupe contrôle : prise en charge standard.
    - Groupe test : programme post-sortie.
    Mesurer :
    - taux de réadmission <30 jours ;
    - coût moyen par patient ;
    - retour sur investissement.
    """)
    st.markdown("""
    ### 3. Actions basées sur les drivers SHAP
    **Historique hospitalier**
    - Créer une alerte automatique lorsque
      `number_inpatient >= 2`.
    **Mode de sortie**
    - Vérifier qu'un rendez-vous médical est programmé
      dans les 7 jours suivant la sortie.
    **Poly-médication**
    - Déclencher un bilan pharmaceutique lorsque
      le nombre de médicaments est élevé.
    """)
    st.info(
        "Les variables les plus influentes identifiées dans le notebook "
        "sont : number_inpatient, number_emergency, "
        "number_diagnoses, discharge_disposition et num_medications."
    )
