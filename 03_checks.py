"""Step 3: stress-test the two findings worth asking about.

1. CD8 T cells come out HIGHER in relapsed patients here, while the abstract says
   CD8 T cells were depleted in relapse-prone tumours. Is that a normalisation
   choice (all cells vs immune cells), one outlier patient, or the two cohorts?
2. Does the M2-macrophage neighbourhood's enrichment in relapse survive other
   LDA random seeds?
"""
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
from sklearn.decomposition import LatentDirichletAllocation

df = pd.read_csv(sys.argv[1], index_col=0)
df = df[df["cell_cluster"] != "Junk"].reset_index(drop=True)
rel = df.groupby("Patient")["Relapse"].first()
Y, N = rel[rel == "Y"].index, rel[rel == "N"].index
cohort = pd.Series({p: ("E" if p.startswith("E") else "Pt") for p in rel.index})


def mw(s):
    return mannwhitneyu(s[Y], s[N], alternative="two-sided").pvalue


IMMUNE = ["B", "CD4", "CD8", "M1", "M2", "Monocyte", "NK", "Neutrophil", "Plasmablast", "TReg"]
MYELOID_SUPP = ["M2", "Monocyte", "Neutrophil"]

all_frac = pd.crosstab(df["Patient"], df["cell_cluster"], normalize="index")
imm = df[df["cell_cluster"].isin(IMMUNE)]
imm_frac = pd.crosstab(imm["Patient"], imm["cell_cluster"], normalize="index")

table = pd.DataFrame({
    "relapse": rel,
    "cohort": cohort,
    "CD8_of_all": all_frac["CD8"],
    "CD8_of_immune": imm_frac["CD8"],
    "CD8_per_tumour_cell": all_frac["CD8"] / all_frac["Tumor"],
    "M2_of_immune": imm_frac["M2"],
    "Neut_of_immune": imm_frac["Neutrophil"],
    "supp_myeloid_of_immune": imm_frac[MYELOID_SUPP].sum(axis=1),
}).sort_values(["relapse", "cohort"])
print("1. Per-patient values")
print(table.round(3).to_string(), "\n")
for col in table.columns[2:]:
    print(f"   {col:24s} median Y={table.loc[Y, col].median():.3f}  "
          f"N={table.loc[N, col].median():.3f}  p={mw(table[col]):.3f}")
print("\n   Relapse by cohort:")
print(pd.crosstab(table["cohort"], table["relapse"]), "\n")

# 2. seed stability of the M2 neighbourhood result
types = sorted(df["cell_cluster"].unique())
tidx = df["cell_cluster"].map({t: i for i, t in enumerate(types)}).to_numpy()
counts = np.zeros((len(df), len(types)), dtype=np.int32)       # (cells, types)
for _, rows in df.groupby("fov").groups.items():
    rows = np.asarray(rows)
    _, nn = cKDTree(df.loc[rows, ["X_cent", "Y_cent"]].to_numpy()).query(
        df.loc[rows, ["X_cent", "Y_cent"]].to_numpy(), k=10)   # (cells_in_fov, 10)
    nt = tidx[rows][nn]
    for j in range(len(types)):
        counts[rows, j] = (nt == j).sum(axis=1)
frac = counts / 10
m2 = types.index("M2")
print("2. Seed stability: the neighbourhood with the most M2, relapsed vs not")
for seed in [0, 1, 2, 3, 4]:
    fit = np.random.default_rng(seed).choice(len(df), 150_000, replace=False)
    lda = LatentDirichletAllocation(n_components=8, random_state=seed,
                                    learning_method="online").fit(counts[fit])
    cn = lda.transform(counts).argmax(axis=1)                  # (cells,)
    comp = pd.DataFrame(frac, columns=types).groupby(cn).mean()  # (8, types)
    m2_cn = comp["M2"].idxmax()
    share = pd.Series(cn == m2_cn).groupby(df["Patient"]).mean()
    print(f"   seed {seed}: M2 neighbourhood is {comp.loc[m2_cn, 'M2']:.0%} M2; "
          f"median share Y={share[Y].median():.3f} N={share[N].median():.3f} "
          f"p={mw(share):.3f}; relapse > no-relapse in "
          f"{(share[Y].values[:, None] > share[N].values[None, :]).mean():.0%} of pairs")
