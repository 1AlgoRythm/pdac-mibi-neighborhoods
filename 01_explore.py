"""Step 1: what is in the lab's public per-cell table?

Data: Zenodo record 15109931, file "All_anno_Tumor distance-inf_0328.csv"
(one row per segmented cell, from the MIBI images in Yeung et al., MCP 2026).
"""
import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_csv(path, index_col=0)

print("rows (cells):", len(df))
print("columns:", list(df.columns))
print()
print("patients and relapse status:")
print(df.groupby("Patient")["Relapse"].agg(["first", "nunique", "size"]))
print()
print("FOVs (images) per patient:")
print(df.groupby("Patient")["fov"].nunique())
print()
print("cell_cluster_broad counts:")
print(df["cell_cluster_broad"].value_counts())
print()
print("cell_cluster counts:")
print(df["cell_cluster"].value_counts())
print()
print("distance columns:")
print(df[["distance_to_tumor", "distance_um"]].describe())
