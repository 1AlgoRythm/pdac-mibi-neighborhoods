"""Step 2: try to reproduce the paper's main claims from the public per-cell table.

Paper: Yeung, Bai, ..., Sivakumar, Jiang. "Spatial Proteomics Uncovers Immune Cell
Neighborhoods and Tumor-Immune Interfaces in Pancreatic Ductal Adenocarcinoma."
Mol Cell Proteomics 2026. PMID 42946214. Abstract claims tested here:

  A. Immunosuppressive myeloid cells are enriched, and CD8 T cells depleted,
     in patients who relapsed.
  B. Topic modeling finds 8 cellular neighborhoods (CNs), and an immune-
     suppressive CN is enriched in relapsed patients.
  C. Neutrophils at the tumor-immune interface differ by relapse status.

This is NOT the paper's exact method. The paper used *spatial* probabilistic
topic modeling; here plain LDA runs on each cell's k-nearest-neighbour window,
which is the simpler idea underneath it. With 8 relapsed vs 4 non-relapsed
patients, every p-value below is fragile; read them as direction, not proof.
"""
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import MiniBatchKMeans

K_NEIGHBOURS = 10   # window size: the cell itself + its 9 nearest neighbours
N_CNS = 8           # the paper reports 8 neighborhoods
SEED = 0

df = pd.read_csv(sys.argv[1], index_col=0)
df = df[df["cell_cluster"] != "Junk"].reset_index(drop=True)
types = sorted(df["cell_cluster"].unique())            # 16 cell types
patient_relapse = df.groupby("Patient")["Relapse"].first()
relapsed = patient_relapse[patient_relapse == "Y"].index
not_relapsed = patient_relapse[patient_relapse == "N"].index
print(f"{len(df)} cells, {len(types)} cell types, "
      f"{len(relapsed)} relapsed vs {len(not_relapsed)} non-relapsed patients\n")


def compare(per_patient: pd.DataFrame, label: str) -> pd.DataFrame:
    """Mann-Whitney U per column: relapsed patients vs non-relapsed patients."""
    rows = []
    for col in per_patient.columns:
        y, n = per_patient.loc[relapsed, col], per_patient.loc[not_relapsed, col]
        p = mannwhitneyu(y, n, alternative="two-sided").pvalue
        rows.append((col, y.median(), n.median(), p))
    out = pd.DataFrame(rows, columns=[label, "median_relapse", "median_no_relapse", "p"])
    return out.sort_values("p")


# ---- A. cell-type composition per patient --------------------------------
comp = pd.crosstab(df["Patient"], df["cell_cluster"], normalize="index")  # (12 patients, 16 types)
print("A. Cell-type fraction per patient, relapsed vs not")
print(compare(comp, "cell_type").round(4).to_string(index=False), "\n")

# ---- B. neighbourhood windows -> topic model -----------------------------
type_idx = df["cell_cluster"].map({t: i for i, t in enumerate(types)}).to_numpy()
counts = np.zeros((len(df), len(types)), dtype=np.int32)  # (cells, types): neighbour counts
for fov, rows in df.groupby("fov").groups.items():
    rows = np.asarray(rows)
    xy = df.loc[rows, ["X_cent", "Y_cent"]].to_numpy()   # (cells_in_fov, 2)
    _, nn = cKDTree(xy).query(xy, k=K_NEIGHBOURS)         # (cells_in_fov, K)
    neigh_types = type_idx[rows][nn]                      # (cells_in_fov, K)
    for j in range(len(types)):
        counts[rows, j] = (neigh_types == j).sum(axis=1)

rng = np.random.default_rng(SEED)
fit_rows = rng.choice(len(df), size=150_000, replace=False)
lda = LatentDirichletAllocation(n_components=N_CNS, random_state=SEED, learning_method="online")
lda.fit(counts[fit_rows])
theta = lda.transform(counts)                             # (cells, 8): topic mix per cell
df["CN"] = theta.argmax(axis=1)

# describe each CN by the average neighbourhood composition of its cells
frac = counts / K_NEIGHBOURS                              # (cells, types)
cn_comp = pd.DataFrame(frac, columns=types).groupby(df["CN"]).mean()  # (8, types)
print("B. What each topic-model neighborhood is made of (top 3 cell types)")
for cn, row in cn_comp.iterrows():
    top = row.sort_values(ascending=False).head(3)
    desc = ", ".join(f"{t} {v:.0%}" for t, v in top.items())
    print(f"  CN{cn}: {desc}  ({(df['CN'] == cn).mean():.1%} of cells)")

cn_frac = pd.crosstab(df["Patient"], df["CN"], normalize="index")  # (12, 8)
cn_frac.columns = [f"CN{c}" for c in cn_frac.columns]
cn_comp.index = [f"CN{c}" for c in cn_comp.index]
pd.concat([patient_relapse.rename("Relapse"), comp.add_prefix("type_"), cn_frac.add_prefix("share_")],
          axis=1).to_csv("per_patient.csv")
cn_comp.to_csv("cn_composition.csv")
print("\nB. Neighborhood fraction per patient, relapsed vs not")
print(compare(cn_frac, "neighborhood").round(4).to_string(index=False), "\n")

# robustness: classic k-means neighborhoods on the same windows
km = MiniBatchKMeans(n_clusters=N_CNS, random_state=SEED, n_init=5).fit(frac[fit_rows])
df["KM"] = km.predict(frac)
km_comp = pd.DataFrame(frac, columns=types).groupby(df["KM"]).mean()
km_frac = pd.crosstab(df["Patient"], df["KM"], normalize="index")
km_frac.columns = [f"KM{c}" for c in km_frac.columns]
print("B'. Same test with k-means neighborhoods (robustness check)")
res = compare(km_frac, "neighborhood")
res["top_types"] = [", ".join(f"{t} {v:.0%}" for t, v in
                    km_comp.loc[int(n[2:])].sort_values(ascending=False).head(3).items())
                    for n in res["neighborhood"]]
print(res.round(4).to_string(index=False), "\n")

# ---- C. neutrophils at the tumour-immune interface -----------------------
markers = ["MPO", "CD15", "CD11b", "CD16", "PD.L1", "HLA.DR", "Ki.67", "CD45RO"]
near = df[(df["cell_cluster"] == "Neutrophil") & df["distance_um"].between(0, 50)]
far = df[(df["cell_cluster"] == "Neutrophil") & (df["distance_um"] > 200) & np.isfinite(df["distance_um"])]
print(f"C. Neutrophils within 50 um of tumour: {len(near)}; beyond 200 um: {len(far)}")
print("   Share of each patient's neutrophils that sit within 50 um of tumour:")
neut = df[df["cell_cluster"] == "Neutrophil"]
share_near = neut.assign(near=neut["distance_um"].between(0, 50)).groupby("Patient")["near"].mean()
print(compare(share_near.to_frame("neutrophil_share_near_tumour"), "measure").round(4).to_string(index=False))
print("   Mean marker level of interface neutrophils, per patient, relapsed vs not:")
print(compare(near.groupby("Patient")[markers].mean(), "marker").round(4).to_string(index=False))
