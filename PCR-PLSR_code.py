import math
import warnings
from datetime import datetime

import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Ellipse
from matplotlib.ticker import FormatStrFormatter, PercentFormatter
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_predict
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.seasonal import seasonal_decompose

# Configuraciones globales para silenciar advertencias de Pandas y Matplotlib
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)

# ==========================================
# 1. CARGA Y PREPROCESAMIENTO DE DATOS
# ==========================================

paths = {
    'estacion': (
        r"D:\MEGA\_Maestría en Ingeniería Civil\03 Semestre 3\Trabajo de"
        r" investigación\Datos\Originales\WeatherStationData.xlsx"
    ),
    'temp_sup': (
        r"D:\MEGA\_Maestría en Ingeniería Civil\03 Semestre 3\Trabajo de"
        r" investigación\Datos\Originales\SurfaceTemperature(CEP,CWP,DEP)_complete.xlsx"
    ),
    'dron_top': (
        r"D:\MEGA\_Maestría en Ingeniería Civil\03 Semestre 3\Trabajo de"
        r" investigación\Datos\Originales\SurfaceTemperatureDroneTop.xlsx"
    ),
    'tc': (
        r"D:\MEGA\_Maestría en Ingeniería Civil\03 Semestre 3\Trabajo de"
        r" investigación\Datos\Originales\InternalTemperature.xlsx"
    ),
}

dfs = {}
for key, path in paths.items():
  df_i = pd.read_excel(path, skiprows=0)
  if key != 'temp_sup':
    df_i.drop(0, inplace=True)
  df_i.rename(
      columns={'Date (YY-MM-DD) - Local Time (UTC -5)': 'Date'}, inplace=True
  )

  if key == 'temp_sup':
    df_i['Date'] = pd.to_datetime(df_i['Date'], unit='D', origin='1899-12-30')
  else:
    df_i['Date'] = pd.to_datetime(df_i['Date'])

  df_i.replace('No data', np.nan, inplace=True)
  df_i = df_i.infer_objects(copy=False)
  df_i['Date'] = df_i['Date'].dt.round('30min')
  dfs[key] = df_i

wind_dirs = [
    'N',
    'NNW',
    'NW',
    'WNW',
    'W',
    'WSW',
    'SW',
    'SSW',
    'S',
    'SSE',
    'SE',
    'ESE',
    'E',
    'ENE',
    'NE',
    'NNE',
]
dir_map = {d: i * np.pi / 8 for i, d in enumerate(wind_dirs)}

dfs['estacion']['Wind Dir'] = (
    dfs['estacion']['Wind Dir'].astype(str).str.strip().replace(dir_map)
)
dfs['estacion']['Wind Dir'] = pd.to_numeric(
    dfs['estacion']['Wind Dir'], errors='coerce'
)
dfs['estacion']['Wind Speed'] = pd.to_numeric(
    dfs['estacion']['Wind Speed'], errors='coerce'
)
dfs['estacion']['Out Hum'] = (
    pd.to_numeric(
        dfs['estacion']['Out Hum'].astype(str).str.strip(), errors='coerce'
    ).astype('Int64')
)

df_tc = dfs['tc'].apply(pd.to_numeric, errors='coerce')
df_tc['Date'] = dfs['tc']['Date']

tc_pairs = {
    'Tv_int1': ['TC4', 'TC5'],
    'Tv_int2': ['TC10', 'TC11'],
    'Tv_int3': ['TC17', 'TC18'],
    'Tv_int4': ['TC17', 'TC18'],
    'Tv_int5': ['TC21', 'TC22'],
    'Tv_int6': ['TC25', 'TC26'],
    'Tv_int7': ['TC28', 'TC29'],
    'Tv_int8': ['TC32', 'TC33'],
    'Tv_int9': ['TC35', 'TC36'],
    'Tv_int10': ['TC39', 'TC40'],
}
df_tv_int = pd.DataFrame({'Date': df_tc['Date']})
for tv, cols in tc_pairs.items():
  df_tv_int[tv] = df_tc[cols].mean(axis=1)
df_tv_int = (
    df_tv_int.drop_duplicates(subset='Date', keep='first').reset_index(
        drop=True
    )
)

fechas = pd.date_range(
    start='2023-07-10 00:00', end='2023-11-06 23:30', freq='30min'
)
df = pd.DataFrame({'Date': fechas})
for key in ['estacion', 'temp_sup', 'dron_top']:
  df = pd.merge(df, dfs[key], on='Date', how='outer')

df = (
    df.drop_duplicates(subset='Date', keep='first')
    .sort_values('Date')
    .reset_index(drop=True)
)
df['Hours'] = np.sin(
    2 * np.pi * (df['Date'].dt.hour + df['Date'].dt.minute / 60) / 24
)

df['WS_NS'] = df['Wind Speed'] * np.cos(df['Wind Dir'])
df['WS_EW'] = -df['Wind Speed'] * np.sin(df['Wind Dir'])

vars_entrada = ['Date', 'Temp Out ', 'Out Hum', 'Wind Speed', 'Solar Rad'] + [
    f'CEP{i}' for i in range(1, 11)
]
vars_salida = [f'Tv_int{i}' for i in range(1, 11)]

for col in [
    'Temp Out ',
    'Solar Rad',
    'Wind Speed',
    'Wind Dir',
    'Out Hum',
    'WS_NS',
    'WS_EW',
]:
  if col in df:
    df[col] = df[col].interpolate(method='linear')

for col in vars_salida:
  df_tv_int[col] = df_tv_int[col].interpolate(method='linear')

df = df[vars_entrada].iloc[925:5457]
horario = (df['Date'].dt.hour + df['Date'].dt.minute / 60 > 6) & (
    df['Date'].dt.hour + df['Date'].dt.minute / 60 < 16.5
)
df = df[horario].dropna().reset_index(drop=True)
df_tv_int = df_tv_int.dropna()

df = df.merge(df_tv_int[['Date']], on='Date', how='inner')
df_tv_int = df_tv_int.merge(df[['Date']], on='Date', how='inner')

df_original2, df_tv_int_original2 = df.copy(), df_tv_int.copy()
df_original, df_tv_int_original = df.copy(), df_tv_int.copy()

# ==========================================
# CONFIGURACIÓN ESTÉTICA GLOBAL (Estándar Código.txt)
# ==========================================

ajust_name = {
    'Temp Out ': r'$T_a$ (°C)',
    'Out Hum': '$RH$ (%)',
    'Solar Rad': '$SR$ (W/m²)',
    'Atm Pressure': '$AP$ (hPa)',
    'Rain': '$P$ (mm)',
    'Wind Speed': '$WS$ (m/s)',
}
ajust_name.update({f'CEP{i}': fr'$T_{{sup,{i}}}$ (°C)' for i in range(1, 11)})
ajust_name.update({f'Tv_int{i}': fr'$T_{{int,{i}}}$ (°C)' for i in range(1, 11)})

round_up_half = lambda x: math.ceil(x * 2) / 2


def set_style(
    fontsize=15, dpi=800, axes_labelsize=None, xtick_size=None, ytick_size=None
):
  plt.rcParams.update({
      'font.family': 'serif',
      'font.serif': ['Times New Roman', 'DejaVu Serif'],
      'mathtext.fontset': 'dejavuserif',
      'font.size': fontsize,
      'axes.labelsize': axes_labelsize if axes_labelsize else fontsize,
      'xtick.labelsize': xtick_size if xtick_size else int(fontsize - 2),
      'ytick.labelsize': ytick_size if ytick_size else int(fontsize - 2),
      'figure.dpi': dpi,
      'text.usetex': False,
  })


# ==========================================
# 2. ANÁLISIS DE VIF (MULTICOLINEALIDAD)
# ==========================================

X_vif = sm.add_constant(df.drop('Date', axis=1).astype('float64'))
vif = pd.DataFrame({
    "Variable": X_vif.columns,
    "VIF": [
        variance_inflation_factor(X_vif.values, i)
        for i in range(X_vif.shape[1])
    ],
})
vif = vif[vif['Variable'] != 'const']
vif['Label_LaTeX'] = vif['Variable'].map(ajust_name).fillna(vif['Variable'])

set_style(
    fontsize=15, dpi=800, axes_labelsize=12, xtick_size=10, ytick_size=10
)
fig, ax = plt.subplots(figsize=(5, 3), dpi=800)
ax.bar(
    vif['Label_LaTeX'],
    vif['VIF'],
    color=sns.color_palette("muted", len(vif)),
    edgecolor='black',
    linewidth=0.7,
    width=0.6,
)
ax.set_yscale('log')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)
ax.set_ylabel('Variance Inflation Factor')
plt.xticks(rotation=45, ha='right')
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.set_ylim(0, 1000)
ax.set_axisbelow(True)
ax.axhline(
    y=10,
    color="#000000",
    linestyle='--',
    linewidth=1.0,
    alpha=1,
    label='Threshold = 10',
)
ax.text(
    0.089,
    0.27,
    "Threshold = 10",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=10.5,
)
plt.tight_layout()
plt.show()

# ==========================================
# 3. DESCOMPOSICIÓN ESTACIONAL (D3C)
# ==========================================


def decompose_df(df_in, cols, date_col, model='additive', period=20):
  df_trend = pd.DataFrame({'Date': df_in[date_col]})
  df_comp = pd.DataFrame({'Date': df_in[date_col]})
  for col in cols:
    if col in ['Hours', 'Date']:
      continue
    series = pd.Series(
        df_in[col].values, index=pd.to_datetime(df_in[date_col])
    ).dropna()
    dec = seasonal_decompose(
        series, model=model, period=period, extrapolate_trend='freq'
    )
    df_trend[col] = dec.trend.values
    df_comp[col] = (dec.resid + dec.seasonal).values
  return df_trend, df_comp


df_tendencia, df = decompose_df(
    df_original, [v for v in vars_entrada if v != 'Date'], 'Date'
)
df_tv_int_tendencia, df_tv_int = decompose_df(
    df_tv_int_original, [v for v in vars_salida if v != 'Date'], 'Date'
)


def plot_decomp_series(
    df_orig, df_trend, df_adj, col_name, ylabel, tag, label_var, dpi=800
):
  set_style(fontsize=15, dpi=dpi)
  fig, ax = plt.subplots(figsize=(6, 3.5), dpi=dpi)
  ax.plot(
      df_orig['Date'],
      df_orig[col_name],
      label=fr'Original time series ({label_var})',
      color='Blue',
      lw=0.8,
      alpha=0.7,
  )
  ax.plot(
      df_trend['Date'],
      df_trend[col_name],
      label='Seasonal components',
      color='Red',
      lw=1.5,
  )
  ax.plot(
      df_adj['Date'],
      df_adj[col_name],
      label=fr'Adjusted time series ({label_var})',
      color='green',
      lw=0.8,
      alpha=0.7,
  )
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
  ax.tick_params(axis='both', labelrotation=0, labelsize=14)
  ax.set_ylabel(ylabel, fontsize=14)
  ax.grid(True, linestyle='--')
  ax.text(-0.1, 1.02, tag, transform=ax.transAxes, fontsize=19)
  plt.tight_layout()
  plt.show()


if 'Temp Out ' in vars_entrada:
  plot_decomp_series(
      df_original2,
      df_tendencia,
      df,
      'Temp Out ',
      'Ambient temperature (°C)',
      '(a)',
      r'${T_a}$',
      dpi=800,
  )
plot_decomp_series(
    df_tv_int_original2,
    df_tv_int_tendencia,
    df_tv_int,
    'Tv_int6',
    r'${T_{{int,6}}}$ (°C)',
    '(b)',
    r'${T_{{int,6}}}$',
    dpi=600,
)

# ==========================================
# 4. ANÁLISIS DE VARIANZA COMPARA PCA VS PLS
# ==========================================

X = df.drop('Date', axis=1).copy()
Y = df_tv_int.drop('Date', axis=1).copy()

set_style(fontsize=17, dpi=800)
tscv = TimeSeriesSplit(n_splits=10)
index_selected = [1, 6, 10]

for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
  X_train, y_train = X.iloc[train_idx], Y.iloc[train_idx]
  X_tr_sc = StandardScaler().fit_transform(X_train)
  y_tr_sc = StandardScaler().fit_transform(y_train)
  n_vars = X_train.shape[1]

  pca = PCA(n_components=n_vars, random_state=22).fit(X_tr_sc)
  pca_var_cum = np.cumsum(pca.explained_variance_ratio_) * 100

  r2x_cum, r2y_cum = [], []
  r2y_cum_ind = {j: [] for j in index_selected}

  for i in range(1, n_vars + 1):
    pls = PLSRegression(n_components=i, scale=False).fit(X_tr_sc, y_tr_sc)
    X_hat = np.dot(pls.x_scores_, pls.x_loadings_.T)
    r2x_cum.append(
        (1 - np.sum((X_tr_sc - X_hat) ** 2) / np.sum(X_tr_sc**2)) * 100
    )
    r2y_cum.append(pls.score(X_tr_sc, y_tr_sc) * 100)

    y_pred = pls.predict(X_tr_sc)
    for j in index_selected:
      r2y_cum_ind[j].append(
          (
              1
              - np.sum((y_tr_sc[:, j - 1] - y_pred[:, j - 1]) ** 2)
              / np.sum(y_tr_sc[:, j - 1] ** 2)
          )
          * 100
      )

  if fold_idx == tscv.n_splits - 1:
    fig, ax1 = plt.subplots(figsize=(4.375, 3.5), dpi=800)
    lvs = np.arange(1, n_vars + 1)
    ms = 4
    ax1.plot(
        lvs,
        pca_var_cum,
        color="red",
        linestyle='-',
        markersize=ms,
        alpha=0.7,
        label="PCA: inputs",
    )
    ax1.plot(
        lvs,
        r2x_cum,
        color="#56B4E9",
        markersize=ms,
        linewidth=1,
        label="PLS: inputs",
    )
    ax1.plot(
        lvs,
        r2y_cum,
        color="#000000",
        marker='s',
        markersize=ms,
        linewidth=1,
        label=r"PLS: $T_{int,all}$",
    )
    for j in index_selected:
      ax1.plot(
          lvs,
          r2y_cum_ind[j],
          marker='o',
          markersize=ms,
          linewidth=1,
          label=fr"PLS: $T_{{int,{j}}}$",
      )

    ax1.set_ylabel("Cumulative Explained Variance", weight='bold', fontsize=12)
    ax1.set_xlabel(
        "Number of Components (PCs / LVs)", weight='bold', fontsize=12
    )
    ax1.set_ylim(0, 105)
    ax1.set_xticks(range(1, len(lvs) + 1, 2))
    ax1.yaxis.set_major_formatter(PercentFormatter())
    ax1.tick_params(axis='both', labelsize=11)
    ax1.axhline(95, color="k", linestyle="-.", linewidth=1.2, alpha=0.6)
    ax1.text(0.35, 96, "95% Threshold", color="k", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.text(
        0.075,
        0.075,
        '(a)',
        transform=ax1.transAxes,
        ha="left",
        va="bottom",
        fontsize=16,
    )
    ax1.legend(frameon=False, fontsize=10.5)
    plt.tight_layout()
    plt.show()

# ==========================================
# 5. GRÁFICA DE SCORES Y BIPLOT (PCA)
# ==========================================

set_style(fontsize=14, dpi=800)
pca_fitted = PCA(n_components=len(vars_entrada) - 1, random_state=22).fit(
    StandardScaler().fit_transform(X)
)
scores = pca_fitted.transform(StandardScaler().fit_transform(X))


def asignar_grupo(dt):
  hora = dt.hour + dt.minute / 60
  if 6.5 <= hora < 8.5:
    return "6:30-8:30"
  elif 9.0 <= hora < 11.0:
    return "9:00-11:00"
  elif 11.5 <= hora < 13.5:
    return "11:30-13:30"
  elif 14.0 <= hora < 16.0:
    return "14:00-16:00"
  return None


grupos = df['Date'].apply(asignar_grupo)
mask = grupos.notna()
scores_filt, grupos_filt = scores[mask], grupos[mask]
colors_dict = {
    "6:30-8:30": "#F28E2B",
    "9:00-11:00": "#4E79A7",
    "11:30-13:30": "#E15759",
    "14:00-16:00": "#76B7B2",
}


def plot_ellipse(ax, x, y, n_std, **kwargs):
  cov = np.cov(x, y)
  eigvals, eigvecs = np.linalg.eigh(cov)
  order = eigvals.argsort()[::-1]
  eigvals, eigvecs = eigvals[order], eigvecs[:, order]
  angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
  width, height = 2 * n_std * np.sqrt(eigvals)
  ax.add_patch(
      Ellipse(
          (np.mean(x), np.mean(y)),
          width,
          height,
          angle=angle,
          fill=False,
          **kwargs,
      )
  )


fig, ax = plt.subplots(figsize=(5, 4), dpi=800)
for grupo, color in colors_dict.items():
  idx = grupos_filt == grupo
  ax.scatter(
      scores_filt[idx, 0],
      scores_filt[idx, 1],
      label=grupo,
      color=color,
      alpha=0.5,
      edgecolor='white',
      s=60,
      linewidth=0.5,
  )

plot_ellipse(
    ax,
    scores_filt[:, 0],
    scores_filt[:, 1],
    2.146,
    linestyle='--',
    color='gray',
    linewidth=1.5,
)
plot_ellipse(
    ax,
    scores_filt[:, 0],
    scores_filt[:, 1],
    2.447,
    linestyle='--',
    color='green',
    linewidth=1.5,
)
plot_ellipse(
    ax,
    scores_filt[:, 0],
    scores_filt[:, 1],
    3.035,
    linestyle='--',
    color='red',
    linewidth=1.5,
)

ax.set_xlabel(
    f"PC1 ({pca_fitted.explained_variance_ratio_[0]*100:.1f}%)",
    weight='bold',
    fontsize=14,
)
ax.set_ylabel(
    f"PC2 ({pca_fitted.explained_variance_ratio_[1]*100:.1f}%)",
    weight='bold',
    fontsize=14,
)
ax.set_ylim(-5, 5)
ax.set_xlim(-12, 12)
ax.axhline(0, color='gray', linewidth=0.8)
ax.axvline(0, color='gray', linewidth=0.8)
ax.grid(True, linestyle='--', alpha=0.4)

leg1 = ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.49, 1.02),
    ncol=4,
    frameon=False,
    fontsize=11.4,
    columnspacing=0.2,
    handletextpad=0,
)
ax.add_artist(leg1)

ellipse_elements = [
    mlines.Line2D(
        [0], [0], color='gray', lw=1.5, linestyle='--', label='90% limit'
    ),
    mlines.Line2D(
        [0], [0], color='green', lw=1.5, linestyle='--', label='95% limit'
    ),
    mlines.Line2D(
        [0], [0], color='red', lw=1.5, linestyle='--', label='99% limit'
    ),
]
ax.legend(
    handles=ellipse_elements,
    loc='lower right',
    frameon=True,
    facecolor='white',
    edgecolor='black',
    fontsize=12,
    title_fontsize=12,
)
ax.tick_params(axis='both', labelsize=14)
ax.text(
    0.075,
    0.075,
    '(b)',
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=18.2,
)
plt.tight_layout()
plt.show()

# Biplot / Loadings
set_style(fontsize=18, dpi=800)
pc1, pc2 = pca_fitted.components_[0], pca_fitted.components_[1]
variable_labels = {i + 1: col for i, col in enumerate(X.columns)}
offsets = {
    1: (0.02, 0.0),
    2: (0.0, -0.06),
    3: (0.0, -0.06),
    4: (0.0, -0.06),
    5: (0.0, 0.05),
    6: (0.02, 0.0),
    7: (0.0, -0.06),
    8: (0.0, 0.06),
    9: (0.02, 0.0),
    10: (0.0, -0.06),
    11: (0.02, 0.0),
    12: (-0.02, -0.06),
    13: (0.02, 0.0),
    14: (0.02, 0.0),
    15: (0.0, 0.0),
}

fig, ax = plt.subplots(figsize=(5.625, 4.5), dpi=800)
for i, var in enumerate(X.columns):
  x, y = pc1[i], pc2[i]
  dx, dy = offsets.get(i + 1, (0.02, 0.02))
  ax.scatter(x, y, color="#0072B2", alpha=0.5, zorder=10, marker='^', s=90)
  ax.text(
      x + dx,
      y + dy,
      str(i + 1),
      fontsize=12,
      color="Black",
      ha='center',
      va='center',
      path_effects=[pe.withStroke(linewidth=3, foreground="white")],
  )

for idx in set([
    np.argmax(pc1),
    np.argmin(pc1),
    np.argmax(pc2),
    np.argmin(pc2),
]):
  ax.plot(
      [0, pc1[idx]],
      [0, pc2[idx]],
      color='red',
      linestyle='--',
      linewidth=1.4,
      zorder=4,
  )

legend_elements = [
    mlines.Line2D(
        [],
        [],
        color='none',
        label=f"[{i}] {ajust_name.get(var, var).split(' (')[0]}",
    )
    for i, var in variable_labels.items()
]
ax.legend(
    handles=legend_elements,
    loc='lower center',
    bbox_to_anchor=(0.5, 1),
    ncol=5,
    frameon=False,
    fontsize=13.2,
    columnspacing=0,
    handletextpad=0.1,
    handlelength=0.5,
)
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel(
    f"PC1 ({pca_fitted.explained_variance_ratio_[0]*100:.1f}%)",
    weight='bold',
    fontsize=14,
)
ax.set_ylabel(
    f"PC2 ({pca_fitted.explained_variance_ratio_[1]*100:.1f}%)",
    weight='bold',
    fontsize=14,
)
ax.set_xlim(-0.35, 0.35)
ax.set_ylim(-0.5, 0.5)
ax.grid(alpha=0.3, linestyle='--')
ax.text(
    0.075,
    0.075,
    '(c)',
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=20,
)
plt.tight_layout()
plt.show()

# ==========================================
# 6. EVALUACIÓN Y VALIDACIÓN CRUZADA DE MODELOS
# ==========================================

test_results_pcr, test_results_pls = [], []
train_results_pcr, train_results_pls = [], []

max_k = len(vars_entrada) - 1

for fold_idx, (train_index, test_index) in enumerate(tscv.split(X)):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

  scaler_X = StandardScaler()
  X_tr_sc = scaler_X.fit_transform(X_train)
  X_te_sc = scaler_X.transform(X_test)

  for n_comp in range(1, max_k + 1):
    pca = PCA(n_components=n_comp)
    X_tr_pca_c = sm.add_constant(pca.fit_transform(X_tr_sc))
    X_te_pca_c = sm.add_constant(pca.transform(X_te_sc))

    for col in y_train.columns:
      y_tr_raw, y_te_raw = y_train[col].values, y_test[col].values
      tendencia_tr = df_tv_int_tendencia.loc[y_train.index, col].values
      y_real_tr = df_tv_int_original2.loc[y_train.index, col].values
      tendencia_te = df_tv_int_tendencia.loc[y_test.index, col].values
      y_real_te = df_tv_int_original2.loc[y_test.index, col].values

      model_pcr = sm.OLS(y_tr_raw, X_tr_pca_c).fit()
      y_pred_tr_pcr = model_pcr.predict(X_tr_pca_c)
      y_pred_te_pcr = model_pcr.predict(X_te_pca_c)

      train_results_pcr.append({
          'n_comp': n_comp,
          'RMSE_train': np.sqrt(
              mean_squared_error(y_real_tr, y_pred_tr_pcr + tendencia_tr)
          ),
          'R2_train': r2_score(y_real_tr, y_pred_tr_pcr + tendencia_tr),
      })
      test_results_pcr.append({
          'n_comp': n_comp,
          'RMSE_test': np.sqrt(
              mean_squared_error(y_real_te, y_pred_te_pcr + tendencia_te)
          ),
          'R2_test': r2_score(y_real_te, y_pred_te_pcr + tendencia_te),
      })

      scaler_y = StandardScaler()
      y_tr_sc = scaler_y.fit_transform(y_tr_raw.reshape(-1, 1)).ravel()
      pls = PLSRegression(n_components=n_comp, scale=False).fit(
          X_tr_sc, y_tr_sc
      )

      y_pred_tr_pls = scaler_y.inverse_transform(
          pls.predict(X_tr_sc).reshape(-1, 1)
      ).ravel()
      y_pred_te_pls = scaler_y.inverse_transform(
          pls.predict(X_te_sc).reshape(-1, 1)
      ).ravel()

      train_results_pls.append({
          'n_comp': n_comp,
          'RMSE_train': np.sqrt(
              mean_squared_error(y_real_tr, y_pred_tr_pls + tendencia_tr)
          ),
          'R2_train': r2_score(y_real_tr, y_pred_tr_pls + tendencia_tr),
      })
      test_results_pls.append({
          'n_comp': n_comp,
          'RMSE_test': np.sqrt(
              mean_squared_error(y_real_te, y_pred_te_pls + tendencia_te)
          ),
          'R2_test': r2_score(y_real_te, y_pred_te_pls + tendencia_te),
      })


def get_optimal_comp(results_list):
  df_res = pd.DataFrame(results_list)
  stats_df = df_res.groupby('n_comp')['RMSE_test'].agg(['mean', 'std', 'count'])
  stats_df['se'] = stats_df['std'] / np.sqrt(stats_df['count'])
  rmse_min = stats_df['mean'].min()
  se_min = stats_df.loc[stats_df['mean'].idxmin(), 'se']
  return stats_df[stats_df['mean'] <= rmse_min + se_min].index.min()


pcr_opt = get_optimal_comp(test_results_pcr)
pls_opt = get_optimal_comp(test_results_pls)
print(f"Número óptimo de componentes: PCR={pcr_opt}, PLS={pls_opt}")


def plot_perf_curves(pcr_data, pls_data, is_test=True, letra=15.5):
  set_style(fontsize=15, dpi=800)
  pcr_df = (
      pd.DataFrame(pcr_data)
      .groupby('n_comp')
      .agg(['mean', 'std'])
      .reset_index()
  )
  pls_df = (
      pd.DataFrame(pls_data)
      .groupby('n_comp')
      .agg(['mean', 'std'])
      .reset_index()
  )

  r2_col = 'R2_test' if is_test else 'R2_train'
  rmse_col = 'RMSE_test' if is_test else 'RMSE_train'

  fig, ax1 = plt.subplots(figsize=(5.3, 3.3), dpi=800)

  ax1.plot(
      pcr_df['n_comp'],
      pcr_df[(r2_col, 'mean')],
      color='#1f77b4',
      lw=2.0,
      linestyle='-',
      label=r'$R^2$ (PCR)',
      zorder=3,
  )
  ax1.fill_between(
      pcr_df['n_comp'],
      pcr_df[(r2_col, 'mean')] - pcr_df[(r2_col, 'std')],
      pcr_df[(r2_col, 'mean')] + pcr_df[(r2_col, 'std')],
      color='#1f77b4',
      alpha=0.15,
      zorder=2,
  )

  ax1.plot(
      pls_df['n_comp'],
      pls_df[(r2_col, 'mean')],
      color="#62b41f",
      lw=2.0,
      linestyle='--',
      label=r'$R^2$ (PLSR)',
      zorder=3,
  )
  ax1.fill_between(
      pls_df['n_comp'],
      pls_df[(r2_col, 'mean')] - pls_df[(r2_col, 'std')],
      pls_df[(r2_col, 'mean')] + pls_df[(r2_col, 'std')],
      color="#62b41f",
      alpha=0.10,
      zorder=2,
  )

  ax1.set_xlabel("Number of Components (PCs / LVs)", fontsize=letra)
  ax1.set_ylabel(r'$R^2$', fontsize=letra)
  ax1.tick_params(labelsize=int(letra - 1))
  ax1.set_ylim(0, 1.05)
  ax1.spines['left'].set_linewidth(1.5)
  ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

  ax2 = ax1.twinx()
  ax2.plot(
      pcr_df['n_comp'],
      pcr_df[(rmse_col, 'mean')],
      color='#d62728',
      lw=2.0,
      linestyle='-',
      label=r'$\mathrm{RMSE}$ (PCR)',
      zorder=3,
  )
  ax2.fill_between(
      pcr_df['n_comp'],
      pcr_df[(rmse_col, 'mean')] - pcr_df[(rmse_col, 'std')],
      pcr_df[(rmse_col, 'mean')] + pcr_df[(rmse_col, 'std')],
      color='#d62728',
      alpha=0.15,
      zorder=2,
  )

  ax2.plot(
      pls_df['n_comp'],
      pls_df[(rmse_col, 'mean')],
      color="#b613cc",
      lw=2.0,
      linestyle='--',
      label=r'$\mathrm{RMSE}$ (PLSR)',
      zorder=3,
  )
  ax2.fill_between(
      pls_df['n_comp'],
      pls_df[(rmse_col, 'mean')] - pls_df[(rmse_col, 'std')],
      pls_df[(rmse_col, 'mean')] + pls_df[(rmse_col, 'std')],
      color="#b613cc",
      alpha=0.10,
      zorder=2,
  )

  ax2.set_ylabel(r'$\mathrm{RMSE (°C)}$', fontsize=letra)
  ax2.tick_params(labelsize=int(letra - 1))
  ax2.spines['right'].set_linewidth(1.5)
  ax2.set_ylim(0.7, 2.055)
  rmse_vals = [0.7, 1.0231, 1.346, 1.6681, 2]
  ax2.set_yticks(rmse_vals)
  ax2.set_yticklabels([f'{v:.2f}' for v in rmse_vals])

  ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.4, color='gray', zorder=1)
  for spine in ['top', 'bottom']:
    ax1.spines[spine].set_linewidth(1.2)

  ax1.text(
      -0.3 if is_test else -0.35,
      0.33,
      "Test set" if is_test else "Training set",
      fontsize=int(letra + 1),
      rotation=90,
  )
  plt.tight_layout()
  plt.show()


plot_perf_curves(test_results_pcr, test_results_pls, is_test=True)
plot_perf_curves(train_results_pcr, train_results_pls, is_test=False)

# ==========================================
# 7. MODELOS FINALES Y PREDICCIONES
# ==========================================


def build_bestmodels_multi_global(modelo='PLSR'):
  k_opt = int(pcr_opt if modelo == 'PCR' else pls_opt)
  scaler_X_full = StandardScaler()
  X_sc_full = scaler_X_full.fit_transform(X.values)
  filas_componentes, filas_originales = [], []
  col_comp = 'PC' if modelo == 'PCR' else 'LV'

  for target_var in Y.columns:
    y_vec = Y[target_var].values
    if modelo == 'PCR':
      pca_full = PCA(n_components=k_opt)
      T_full = pca_full.fit_transform(X_sc_full)
      ols_pc = sm.OLS(y_vec, sm.add_constant(T_full)).fit()
      beta_pc = ols_pc.params[1:]

      fila_comp = {'variable': target_var, 'k_opt': k_opt}
      fila_comp.update(
          {f'{col_comp}{i+1}': abs(beta_pc[i]) for i in range(k_opt)}
      )
      filas_componentes.append(fila_comp)

      coef_orig = (beta_pc @ pca_full.components_) / scaler_X_full.scale_
      intercept_orig = ols_pc.params[0] - np.dot(
          coef_orig, scaler_X_full.mean_
      )
    else:
      scaler_y_full = StandardScaler()
      y_sc = scaler_y_full.fit_transform(y_vec.reshape(-1, 1)).ravel()
      pls_full = PLSRegression(n_components=k_opt, scale=False).fit(
          X_sc_full, y_sc
      )

      ols_lv = sm.OLS(y_sc, sm.add_constant(pls_full.x_scores_)).fit()
      c_lv = ols_lv.params[1:]

      fila_comp = {'variable': target_var, 'k_opt': k_opt}
      fila_comp.update(
          {f'{col_comp}{i+1}': abs(c_lv[i]) for i in range(k_opt)}
      )
      filas_componentes.append(fila_comp)

      coef_orig = (
          pls_full.coef_.ravel() * scaler_y_full.scale_[0]
      ) / scaler_X_full.scale_
      intercept_orig = scaler_y_full.mean_[0] - np.dot(
          coef_orig, scaler_X_full.mean_
      )

    fila_orig = (
        {'variable': target_var, 'k_opt': k_opt, 'Intercept': intercept_orig}
    )
    fila_orig.update({col: val for col, val in zip(X.columns, coef_orig)})
    filas_originales.append(fila_orig)

  comp_cols = ['variable', 'k_opt'] + [
      f'{col_comp}{i+1}' for i in range(k_opt)
  ]
  df_comp = pd.DataFrame(filas_componentes)[comp_cols]
  cols_orig = ['variable', 'k_opt', 'Intercept'] + list(X.columns)
  df_orig = pd.DataFrame(filas_originales)[cols_orig]

  return df_comp, df_orig


_, df_bestmodel_pcr = build_bestmodels_multi_global(modelo='PCR')
_, df_bestmodel_plsr = build_bestmodels_multi_global(modelo='PLSR')


def predict_temp(X_df, df_model, target_var):
  row = df_model.loc[df_model['variable'] == target_var].iloc[0]
  intercept = float(row['Intercept'])
  betas = row[list(X_df.columns)].astype(float).values
  return pd.Series(intercept + np.dot(X_df.values, betas), index=X_df.index)


def compute_metrics(y_obs, y_pred):
  df_eval = (
      pd.concat([y_obs.rename("obs"), y_pred.rename("pred")], axis=1).dropna()
  )
  rmse = float(np.sqrt(mean_squared_error(df_eval["obs"], df_eval["pred"])))
  r2 = float(r2_score(df_eval["obs"], df_eval["pred"]))
  return {"RMSE": rmse, "R2": r2, "data": df_eval}


target_vars = vars_salida

# Pre-cálculo para reutilización en visualizaciones
predicciones = {}
for var in target_vars:
  tend = df_tv_int_tendencia.loc[X.index, var].astype(float)
  y_obs = df_tv_int_original2.loc[X.index, var].astype(float)
  pred_pcr = predict_temp(X, df_bestmodel_pcr, var) + tend
  pred_plsr = predict_temp(X, df_bestmodel_plsr, var) + tend
  predicciones[var] = {
      'obs': y_obs,
      'pcr': pred_pcr,
      'plsr': pred_plsr,
      'm_pcr': compute_metrics(y_obs, pred_pcr),
      'm_plsr': compute_metrics(y_obs, pred_plsr),
  }

# ==========================================
# 8. GRÁFICAS COMPARATIVAS FINALES
# ==========================================

# Gráfica 1: Dispersión Observado vs Predicho
set_style(fontsize=14, dpi=800)
cols = 3
rows = int(np.ceil(len(target_vars) / cols))
fig = plt.figure(figsize=(10, rows * 3.5), dpi=800)

for i, var in enumerate(target_vars):
  res = predicciones[var]
  df_pcr, df_plsr = res['m_pcr']['data'], res['m_plsr']['data']

  xy_min = min(df_pcr.min().min(), df_plsr.min().min())
  xy_max = max(df_pcr.max().max(), df_plsr.max().max())
  lim_min, lim_max = round_up_half(xy_min - 1), round_up_half(xy_max + 1)

  ax = plt.subplot(rows, cols, i + 1)
  ax.scatter(
      df_pcr["obs"],
      df_pcr["pred"],
      alpha=0.7,
      edgecolor='white',
      s=40,
      lw=1,
      color="#000000",
      marker='o',
      label="PCR",
  )
  ax.scatter(
      df_plsr["obs"],
      df_plsr["pred"],
      alpha=0.7,
      edgecolor='white',
      s=60,
      lw=1,
      color="green",
      marker='x',
      label="PLSR",
  )
  ax.plot(
      [lim_min, lim_max], [lim_min, lim_max], color="#F03030", ls="--", lw=1.5
  )

  ax.set_xlabel(fr"Observed {ajust_name.get(var, var)}", fontsize=14)
  ax.set_ylabel(fr"Predicted {ajust_name.get(var, var)}", fontsize=14)
  ax.set_xlim(lim_min, lim_max)
  ax.set_ylim(lim_min, lim_max)
  ax.set_aspect("equal", adjustable="box")
  ax.tick_params(direction="in", top=True, right=True, labelsize=13)
  ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
  ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

  ticks = np.linspace(lim_min, lim_max, 5)
  ax.set_yticks(ticks)
  ax.set_xticks(ticks[1:])
  ax.grid(ls='--', alpha=0.6)

  ax.text(
      0.05,
      0.95,
      f"PCR\nRMSE = {res['m_pcr']['RMSE']:.2f}°C\n$R^2$ = {res['m_pcr']['R2']:.2f}",
      transform=ax.transAxes,
      ha="left",
      va="top",
      fontsize=12.5,
  )
  ax.text(
      0.46,
      0.02,
      f"PLSR\nRMSE = {res['m_plsr']['RMSE']:.2f}°C\n$R^2$ = {res['m_plsr']['R2']:.2f}",
      transform=ax.transAxes,
      ha="left",
      va="bottom",
      fontsize=12.5,
  )

  if i == 1:
    ax.legend(
        loc='upper center',
        frameon=False,
        fontsize=14,
        ncol=2,
        bbox_to_anchor=(0.5, -0.175),
        handletextpad=0.1,
        columnspacing=1,
    )

plt.tight_layout()
plt.show()

# Gráfica 2: Análisis de Residuales
fig = plt.figure(figsize=(12, rows * 3.5), dpi=600)

for i, var in enumerate(target_vars):
  res = predicciones[var]
  label_var = ajust_name.get(var, var)
  y_obs = res['obs']
  resid_pcr = y_obs - res['pcr']
  resid_plsr = y_obs - res['plsr']

  x_min, x_max = y_obs.min(), y_obs.max()
  x_lim_min, x_lim_max = round_up_half(x_min - 1), round_up_half(x_max + 1)
  r_abs_max = max(np.abs(resid_pcr).max(), np.abs(resid_plsr).max())
  y_lim_abs = round_up_half(r_abs_max + 0.5)

  ax = plt.subplot(rows, cols, i + 1)
  A_D_pcr = stats.anderson(resid_pcr.values, dist='norm')
  A_D_plsr = stats.anderson(resid_plsr.values, dist='norm')

  ax.scatter(
      y_obs,
      resid_pcr,
      alpha=1,
      edgecolor='white',
      s=60,
      linewidth=1,
      color="#569BE9",
      marker='^',
      label="PCR",
  )
  ax.scatter(
      y_obs,
      resid_plsr,
      alpha=0.6,
      edgecolor='white',
      s=60,
      linewidth=1,
      color="green",
      marker='x',
      label="PLSR",
  )

  std_resid = np.std(resid_plsr.values)
  ax.axhline(
      2 * std_resid,
      color='k',
      linestyle='--',
      linewidth=1.8,
      label=r'$\pm 2\mathbf{\sigma}$',
  )
  ax.axhline(-2 * std_resid, color='k', linestyle='--', linewidth=1.8)
  ax.axhline(0.0, color="#E21313", linestyle="--", linewidth=1.8)

  ax.set_xlabel(fr"Observed {label_var}")
  ax.set_ylabel(fr"Residual {label_var}")
  ax.set_xlim(x_lim_min, x_lim_max)
  ax.set_ylim(-y_lim_abs, y_lim_abs)
  ax.tick_params(direction="in", top=True, right=True)
  ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
  ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
  ax.grid(linestyle='--', alpha=0.6)

  ax.text(
      0.98,
      0.02,
      fr"$\mathrm{{A-D_{{PCR}}}} = {A_D_pcr.statistic:.2f}$"
      + "\n"
      + fr"$\mathrm{{A-D_{{PLSR}}}} = {A_D_plsr.statistic:.2f}$"
      + "\n"
      + fr"$\mathrm{{Bias}} = {np.mean(resid_pcr):.2f}^\circ C$",
      transform=ax.transAxes,
      ha="right",
      va="bottom",
      fontsize=13,
  )

  if i == 0:
    ax.legend(frameon=False, fontsize=13, loc='upper left')

plt.tight_layout(pad=0.5)
plt.show()

# Gráfica 3: Series Temporales
cols = 2
rows = int(np.ceil(len(target_vars) / cols))
fig = plt.figure(figsize=(12, rows * 3), dpi=500)

for i, var in enumerate(target_vars):
  res = predicciones[var]
  ax = plt.subplot(rows, cols, i + 1)

  ax.plot(
      df_original2['Date'], res['pcr'].values, alpha=0.7, lw=0.5, color="#569BE9"
  )
  ax.plot(
      df_original2['Date'],
      res['obs'].values,
      label='Measured value',
      color='red',
      alpha=1,
      lw=0.7,
  )
  ax.plot(
      df_original2['Date'],
      res['plsr'].values,
      label='PLSR prediction',
      color='green',
      alpha=1,
      lw=0.9,
  )

  ax.set_xlim([datetime(2023, 7, 30, 0, 0), datetime(2023, 11, 1, 0, 0)])
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
  ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
  ax.tick_params(labelsize=12)
  ax.set_ylabel(f"{ajust_name.get(var, var)}", fontsize=13)
  ax.set_yticks(
      np.linspace(
          round_up_half(ax.get_ylim()[0] - 2.5),
          round_up_half(ax.get_ylim()[1]),
          5,
      )
  )
  ax.grid(True, ls='--', color="#CFCFCF")

  if i == len(target_vars) - 2:
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(1.1, -0.4),
        frameon=False,
        fontsize=14,
        ncol=2,
    )

  ax.text(
      0.01,
      0.01,
      f"RMSE = {res['m_plsr']['RMSE']:.2f}°C and $R^2$ = {res['m_plsr']['R2']:.2f}",
      transform=ax.transAxes,
      ha="left",
      va="bottom",
      fontsize=13,
  )

fig.subplots_adjust(hspace=0.3, wspace=0.22)
plt.show()