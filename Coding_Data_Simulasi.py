import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(page_title="Dashboard Analisis Hasil Ujian", layout="wide")
st.title("📊 Dashboard Analisis Hasil Ujian Siswa")
st.markdown("Analisis berbasis data untuk evaluasi performa siswa")

# ==========================================================
# UPLOAD DATA (SUPAYA BISA DI STREAMLIT CLOUD)
# ==========================================================
uploaded_file = st.sidebar.file_uploader("Upload File Excel", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
else:
    st.warning("Silakan upload file Excel terlebih dahulu.")
    st.stop()

# Ambil hanya kolom numerik (20 soal)
indikator = df.select_dtypes(include=np.number)

# ==========================================================
# 1️⃣ KPI NILAI RATA-RATA
# ==========================================================
mean_scores = indikator.mean()
nilai_rata2 = mean_scores.mean()

col1, col2, col3 = st.columns(3)
col1.metric("📈 Rata-rata Nilai", f"{nilai_rata2:.2f}")
col2.metric("📚 Jumlah Soal", indikator.shape[1])
col3.metric("👥 Jumlah Siswa", indikator.shape[0])

st.divider()

# ==========================================================
# 2️⃣ ANALISIS SOAL TERSULIT
# ==========================================================
st.header("2️⃣ Analisis Soal Tersulit")

soal_tersulit = mean_scores.idxmin()

fig_gap, ax_gap = plt.subplots(figsize=(8,4))
ax_gap.bar(mean_scores.index, mean_scores.values)
ax_gap.set_ylabel("Rata-rata Nilai")
ax_gap.set_title("Rata-rata Nilai per Soal")
ax_gap.set_xticklabels(mean_scores.index, rotation=90)

st.pyplot(fig_gap)
st.info(f"📌 Soal paling sulit: **{soal_tersulit}**")

st.divider()

# ==========================================================
# 3️⃣ ANALISIS KORELASI
# ==========================================================
st.header("3️⃣ Korelasi Antar Soal")

corr = indikator.corr()

fig_corr, ax_corr = plt.subplots(figsize=(8,6))
im = ax_corr.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax_corr)

ax_corr.set_xticks(range(len(corr.columns)))
ax_corr.set_yticks(range(len(corr.columns)))
ax_corr.set_xticklabels(corr.columns, rotation=90)
ax_corr.set_yticklabels(corr.columns)

st.pyplot(fig_corr)

st.divider()

# ==========================================================
# 4️⃣ ANALISIS REGRESI
# ==========================================================
st.header("4️⃣ Analisis Regresi")

if indikator.shape[1] >= 2:
    X = sm.add_constant(indikator.iloc[:, :-1])
    y = indikator.iloc[:, -1]

    model = sm.OLS(y, X, missing="drop").fit()
    coef = model.params[1:]
    r2 = model.rsquared

    fig_reg, ax_reg = plt.subplots(figsize=(8,4))
    ax_reg.bar(coef.index, coef.values)
    ax_reg.axhline(0, linestyle="--")
    ax_reg.set_title("Koefisien Regresi")

    st.pyplot(fig_reg)
    st.info(f"📈 Nilai R²: {r2:.2f}")

st.divider()

# ==========================================================
# 5️⃣ SEGMENTASI SISWA
# ==========================================================
st.header("5️⃣ Segmentasi Performa Siswa")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(indikator.fillna(indikator.mean()))

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_label = kmeans.fit_predict(X_scaled)

df["Cluster"] = cluster_label

cluster_mean = df.groupby("Cluster")[indikator.columns].mean()
cluster_mean = cluster_mean.sort_values(by=indikator.columns[-1], ascending=False)

st.dataframe(cluster_mean)

st.success("Segmentasi siswa berhasil dilakukan.")
