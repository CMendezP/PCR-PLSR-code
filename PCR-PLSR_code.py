#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> INICIO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
import math
import numpy as np
import pandas as pd
import warnings
import statsmodels.api as sm
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.lines as mlines

from datetime import datetime
from statsmodels.stats.outliers_influence import variance_inflation_factor
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Ellipse
from statsmodels.stats.stattools import durbin_watson
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import TimeSeriesSplit, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.diagnostic import het_breuschpagan
from matplotlib.ticker import FormatStrFormatter
warnings.filterwarnings("ignore", category=UserWarning) #Ignorar los errores de Matplotlib

#   Lectura y ajuste inicial de datos
reader_cam_east = r"...\SurfaceTemperatureCameraEast.xlsx"
reader_cam_west = r"...\SurfaceTemperatureCameraWest.xlsx"
reader_dron_east = r"...\SurfaceTemperatureDroneEast.xlsx"
reader_tc = r"...\InternalTemperature.xlsx"
estacion = r"...\WeatherStationData.xlsx"
temp_sup = r'...\SurfaceTemperature(CEP,CWP,DEP)_complete.xlsx'
reader_dron_top = r"...\SurfaceTemperatureDroneTop.xlsx"

#   Creación de los dataframe
df_estacion = pd.read_excel(estacion, skiprows=0)
df_cam_east = pd.read_excel(reader_cam_east, skiprows = 0)
df_cam_west = pd.read_excel(reader_cam_west, skiprows = 0)
df_dron_east = pd.read_excel(reader_dron_east, skiprows = 0)
df_tc = pd.read_excel(reader_tc, skiprows = 0)
df_temp_sup = pd.read_excel(temp_sup, skiprows = 0)
df_dron_top = pd.read_excel(reader_dron_top, skiprows = 0)

df_estacion.drop(0, inplace = True)
df_cam_east.drop(0, inplace = True)
df_cam_west.drop(0, inplace = True)
df_dron_east.drop(0, inplace = True)
df_tc.drop(0, inplace = True)

# Renombrar la columna de las fechas
df_estacion.rename(columns={'Date (YY-MM-DD) - Local Time (UTC -5)' : 'Date'}, inplace= True)
df_cam_east.rename(columns={'Date (YY-MM-DD) - Local Time (UTC -5)' : 'Date'}, inplace= True)
df_cam_west.rename(columns={'Date (YY-MM-DD) - Local Time (UTC -5)' : 'Date'}, inplace= True)
df_dron_east.rename(columns={'Date (YY-MM-DD) - Local Time (UTC -5)' : 'Date'}, inplace= True)
df_tc.rename(columns={'Date (YY-MM-DD) - Local Time (UTC -5)' : 'Date'}, inplace= True)

# Convertir la columna Date a tipo datetime
df_estacion['Date'] = pd.to_datetime(df_estacion['Date'])
df_cam_east['Date'] = pd.to_datetime(df_cam_east['Date'])
df_cam_west['Date'] = pd.to_datetime(df_cam_west['Date'])
df_dron_east['Date'] = pd.to_datetime(df_dron_east['Date'])
df_tc['Date'] = pd.to_datetime(df_tc['Date'])
df_temp_sup['Date'] = pd.to_datetime(df_temp_sup['Date'], unit="D", origin="1899-12-30")
df_dron_top['Date'] = pd.to_datetime(df_dron_top['Date'])

df_tc_aux=df_tc.copy(["Date"])  #Creado para guardar las fechas
df_estacion_aux=df_estacion.copy(["Date"])  #Creado para guardar las fechas

# Eliminar datos faltantes
df_estacion.replace('No data', np.nan, inplace = True)
df_cam_east.replace('No data', np.nan, inplace = True)
df_cam_west.replace('No data', np.nan, inplace = True)
df_dron_east.replace('No data', np.nan, inplace = True)
df_tc = df_tc.apply(pd.to_numeric, errors='coerce')
df_dron_top.replace('No data', np.nan, inplace = True)

# Los números están escritos como str, esta línea los convierte en tipo float
df_estacion['Out Hum'] = pd.to_numeric(df_estacion['Out Hum'].str.strip(), errors='coerce').astype('Int64') 

df_tc['Date']=df_tc_aux['Date']
df_estacion['Date']=df_estacion_aux['Date']

# Ajuste de hora al minuto
df_estacion['Date']=df_estacion['Date'].dt.round("30min")
df_cam_east['Date']=df_cam_east['Date'].dt.round("30min")
df_cam_west['Date']=df_cam_west['Date'].dt.round("30min")
df_dron_east['Date']=df_dron_east['Date'].dt.round("30min")
df_temp_sup['Date']=df_temp_sup['Date'].dt.round("30min")
df_tc['Date']=df_tc['Date'].dt.round("30min")
df_dron_top['Date']=df_dron_top['Date'].dt.round("30min")

# Crear el DataFrame con la columna 'Date'
fechas = pd.date_range(start="2023-07-10 00:00", end="2023-11-06 23:30", freq="30min")
df = pd.DataFrame({"Date": fechas})

# Creación del dataframe de entrenamiento

otros_df = [df_estacion, df_temp_sup, df_dron_top]
for df_otro in otros_df:
    df = pd.merge(df, df_otro, on='Date', how='outer')

df.apply(pd.to_numeric, errors='coerce')
df = df.drop_duplicates(subset='Date', keep='first').reset_index(drop=True)

#   Ordenar por fecha
df = df.sort_values('Date').reset_index(drop=True)


# Cálculo de la temperatura interna usando los pares de termocuplas de la línea central
datos = {"Tv_int1":  df_tc[["TC4","TC5"]].mean(axis=1),
         "Tv_int2":  df_tc[["TC10","TC11"]].mean(axis=1),
         "Tv_int3":  df_tc[["TC17","TC18"]].mean(axis=1),
         "Tv_int4":  df_tc[["TC17","TC18"]].mean(axis=1),
         "Tv_int5":  df_tc[["TC21","TC22"]].mean(axis=1),
         "Tv_int6":  df_tc[["TC25","TC26"]].mean(axis=1),
         "Tv_int7":  df_tc[["TC28","TC29"]].mean(axis=1),
         "Tv_int8":  df_tc[["TC32","TC33"]].mean(axis=1),
         "Tv_int9":  df_tc[["TC35","TC36"]].mean(axis=1),
         "Tv_int10":  df_tc[["TC39","TC40"]].mean(axis=1),
         }

df_tc_aux['Date']=df_tc_aux['Date'].dt.round("30min")
df_tv_int = pd.DataFrame(datos)
col = df_tc_aux.pop("Date")
df_tv_int.insert(0, "Date", col)
df_tv_int = df_tv_int.drop_duplicates(subset='Date', keep='first').reset_index(drop=True)

vars_entrada = ['Date']
vars_entrada = vars_entrada + ['Temp Out ','Out Hum','Wind Speed','Solar Rad']
vars_entrada = (vars_entrada+ [f'CEP{i}'for i in range(1,11)])
vars_salida = [f"Tv_int{i}" for i in range(1, 11)]

df_tv_int = df_tv_int[['Date'] + vars_salida]

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DEFINICIÓN DE FUNCIONES <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def round_up_five(x):
    return math.ceil(x * 2) / 2

def get_optimal_comp(df, col_n):
    # Agrupar por componentes para obtener media y SE
    stats = df.groupby(col_n)['RMSE_test'].agg(['mean', 'std', 'count'])
    stats['se'] = stats['std'] / np.sqrt(stats['count'])

    rmse_min = stats['mean'].min()
    idx_min = stats['mean'].idxmin()
    se_min = stats.loc[idx_min, 'se']

    # Regla 1-SE: El modelo más simple dentro de 1 SE del mínimo
    umbral = rmse_min + se_min
    optimo = stats[stats['mean'] <= umbral].index.min()  # <= (corrección de &lt;=)
    return optimo, stats


def predict_temp(X: pd.DataFrame, df_bestmodel: pd.DataFrame, target_var: str) -> pd.Series:
    """
    Predice la variable objetivo 'target_var' a partir de:
      - X: DataFrame de predictores
      - df_bestmodel: DataFrame con una fila por variable, columnas:
            Coeficientes reescalados a variables originales.
    Retorna una Serie con ŷ_detrend (sin tendencia), índice = X.index.
    """
    row = df_bestmodel.loc[df_bestmodel['variable'] == target_var]
    if row.empty:
        raise ValueError(f"[predict_temp] '{target_var}' no está en df_bestmodel['variable'].")

    cols_x = list(X.columns)
    missing_in_df = [c for c in cols_x if c not in row.columns]
    
    if missing_in_df:
        raise ValueError(
            "[predict_temp] Faltan en df_bestmodel columnas presentes en X: "
            + ", ".join(missing_in_df)
        )

    intercept = float(row['Intercept'].iloc[0])
    betas = row[cols_x].iloc[0].astype(float).values

    yhat_detrend = intercept + np.dot(X.values, betas)
    return pd.Series(yhat_detrend, index=X.index, name=f"yhat_{target_var}_detrend")


def compute_metrics(y_obs: pd.Series, y_pred: pd.Series) -> dict:
    """
    Devuelve: {'RMSE': float, 'R2': float, 'data': DataFrame(obs, pred)}
    """
    df_eval = pd.concat([y_obs.rename("obs"), y_pred.rename("pred")], axis=1).dropna()
    rmse = float(np.sqrt(mean_squared_error(df_eval["obs"], df_eval["pred"])))
    r2 = float(r2_score(df_eval["obs"], df_eval["pred"]))
    return {"RMSE": rmse, "R2": r2, "data": df_eval}

# Renombres para etiquetas
ajust_name = {'Temp Out ': r'$T_a$ (°C)',
              'Out Hum': '$RH$ (%)',
              'Solar Rad': '$SR$ (W/m²)',
              'Atm Pressure': '$AP$ (hPa)',
              'Rain': '$P$ (mm)',
              'Wind Speed': '$WS$ (m/s)'}
ajust_name.update({f'CEP{i}': fr'$T_{{sup,{i}}}$ (°C)' for i in range(1,11)})
ajust_name.update({f'Tv_int{i}': fr'$T_{{int,{i}}}$ (°C)' for i in range(1,11)})


def build_bestmodels_detrended(modelo='PLSR'):
    """
    Construye dos DataFrames con una fila por cada variable objetivo (Y.columns),
    usando el número de componentes óptimo global: pcr_opt o pls_opt.

    Devuelve
    --------
    df_bestmodel_pcr_plsr : DataFrame
        Filas = Y.columns.
        Columnas = 'variable', 'k_opt', y los coeficientes por componente (|PCi| o |LVi|).
        Los valores de los componentes son magnitudes (valor absoluto).

    df_bestmodel : DataFrame
        Filas = Y.columns.
        Columnas = 'variable', 'k_opt', 'Intercept', y una columna por cada predictor en X.columns
        con los coeficientes reescalados a las variables originales.
    """
    assert modelo in ('PCR', 'PLSR'), "modelo debe ser 'PCR' o 'PLSR'"

    # Asegurar que Y sea DataFrame
    Y_df = Y if isinstance(Y, pd.DataFrame) else pd.DataFrame(Y)

    # Determinar k óptimo global
    k_opt = int(pcr_opt if modelo == 'PCR' else pls_opt)
    if k_opt < 1:
        raise ValueError(f"El número óptimo global de componentes para {modelo} es inválido: {k_opt}")

    # Escalado de X con todos los datos
    scaler_X_full = StandardScaler()
    X_sc_full = scaler_X_full.fit_transform(X.values)

    filas_componentes = []  # magnitudes por PC/LV
    filas_originales  = []  # coeficientes en variables originales + Intercept

    col_comp = 'PC' if modelo == 'PCR' else 'LV'

    for target_var in Y_df.columns:
        # Vector objetivo
        y_vec = Y_df[target_var].values

        if modelo == 'PCR':
            # ===== PCR: PCA + OLS(y ~ scores) =====
            pca_full = PCA(n_components=k_opt)
            T_full = pca_full.fit_transform(X_sc_full)          # Scores PC
            X_T = sm.add_constant(T_full)
            ols_pc = sm.OLS(y_vec, X_T).fit()

            # Coefs en espacio de componentes (sin intercepto)
            beta_pc = ols_pc.params[1:]                         # shape: (k_opt,)
            # Fila de magnitudes por componente
            fila_comp = {'variable': target_var, 'k_opt': k_opt}
            fila_comp.update({f'{col_comp}{i+1}': abs(beta_pc[i]) for i in range(k_opt)})
            filas_componentes.append(fila_comp)

            # Reescalado a variables originales
            # coef_orig = (beta_pc @ pca.components_) / scale_X
            coef_orig = (beta_pc @ pca_full.components_) / scaler_X_full.scale_
            intercept_orig = ols_pc.params[0] - np.dot(coef_orig, scaler_X_full.mean_)

        else:
            # ===== PLSR: ajustar en X escalado e y escalado =====
            scaler_y_full = StandardScaler()
            y_sc = scaler_y_full.fit_transform(y_vec.reshape(-1, 1)).ravel()

            pls_full = PLSRegression(n_components=k_opt, scale=False)
            pls_full.fit(X_sc_full, y_sc)

            # Magnitudes por LV a partir de OLS(y_sc ~ T)
            T_full = pls_full.x_scores_                         # (n_samples, k_opt)
            X_T = sm.add_constant(T_full)
            ols_lv = sm.OLS(y_sc, X_T).fit()
            c_lv = ols_lv.params[1:]                            # (k_opt,)

            fila_comp = {'variable': target_var, 'k_opt': k_opt}
            fila_comp.update({f'{col_comp}{i+1}': abs(c_lv[i]) for i in range(k_opt)})
            filas_componentes.append(fila_comp)

            # Reescalado a variables originales (coincidente con tu CV)
            # coef_orig = (pls.coef_.ravel() * scale_y) / scale_X
            coef_orig = (pls_full.coef_.ravel() * scaler_y_full.scale_[0]) / scaler_X_full.scale_
            intercept_orig = scaler_y_full.mean_[0] - np.dot(coef_orig, scaler_X_full.mean_)

        # Coefs en variables originales (con nombres EXACTOS de X.columns)
        fila_orig = {'variable': target_var, 'k_opt': k_opt, 'Intercept': intercept_orig}
        fila_orig.update({col: val for col, val in zip(X.columns, coef_orig)})
        filas_originales.append(fila_orig)

    # Construcción de DataFrames finales (una fila por Y)
    # Componentes: asegurar todas las columnas hasta k_opt
    comp_cols = ['variable', 'k_opt'] + [f'{col_comp}{i+1}' for i in range(k_opt)]
    df_bestmodel_pcr_plsr = pd.DataFrame(filas_componentes)
    for c in comp_cols:
        if c not in df_bestmodel_pcr_plsr.columns:
            df_bestmodel_pcr_plsr[c] = np.nan
    df_bestmodel_pcr_plsr = df_bestmodel_pcr_plsr[comp_cols]

    # Originales: Intercept + nombres exactos de X
    cols_originales = ['variable', 'k_opt', 'Intercept'] + list(X.columns)
    df_bestmodel = pd.DataFrame(filas_originales)[cols_originales]

    return df_bestmodel_pcr_plsr, df_bestmodel

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> PREPROCESADO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#Interpola linealmente los datos faltantes
horario_6_30am_4pm = (df["Date"].dt.hour + df["Date"].dt.minute/60 > 6) & (df["Date"].dt.hour + df["Date"].dt.minute/60 < 16.5)
horario_4_30pm_6am = ((df["Date"].dt.hour + df["Date"].dt.minute/60 > 16) | (df["Date"].dt.hour + df["Date"].dt.minute/60 < 6.5))

for i in vars_entrada:
    df[i] = df[i].interpolate(method='linear')
    
for i in vars_salida:        
    df_tv_int[i] = df_tv_int[i].interpolate(method='linear')

# Verificar si hay saltos temporales en el dataframe, todos deberían ser de 1800 seg (30 min)
df_time_diff = df["Date"].diff().dt.total_seconds()

df = df[vars_entrada]
df = df[925:5457] # Para la cámara
# df = df[2941:5409]# Para el drone

#PREPARACIÓN DE DATAFRAMES
# Asegurarse de tener índices temporales alineados
df = df.dropna(subset=vars_entrada).copy()
df_tv_int = df_tv_int.dropna().copy()

df_original = df[horario_6_30am_4pm].copy()
df_original.reset_index(inplace=True)
df_original.drop('index', inplace=True, axis= 1)

df_tv_int_original = df_tv_int.merge(df_original[['Date']], on='Date', how='inner').copy()

df = df[horario_6_30am_4pm]
df = df.merge(df_tv_int[['Date']], on='Date', how='inner')
df_tv_int = df_tv_int.merge(df[['Date']], on='Date', how='inner')

df.reset_index(inplace=True)
df.drop('index', inplace=True, axis= 1)

# Horarios de análisis recalculo
horario_6_30am_4pm = (df["Date"].dt.hour + df["Date"].dt.minute/60 > 6) & (df["Date"].dt.hour + df["Date"].dt.minute/60 < 16.5)
horario_4_30pm_6am = ((df["Date"].dt.hour + df["Date"].dt.minute/60 > 16) | (df["Date"].dt.hour + df["Date"].dt.minute/60 < 6.5))

# Extraemos X (predictoras) e Y (multisalida)
X_vif = df.drop('Date', axis=1).copy()
Y_vif = df_tv_int.drop('Date', axis=1).copy()

X_vif = sm.add_constant(X_vif)
X_vif = X_vif.astype('float64')
# Crear DataFrame de VIFs
vif = pd.DataFrame()
vif["Variable"] = X_vif.columns
vif["VIF"] = [variance_inflation_factor(X_vif.values, i)
              for i in range(X_vif.shape[1])]


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DESCOMPOSICIÓN <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

df_original = df.copy()
df_tv_int_original = df_tv_int.copy()
x_train_index = int(len(df)*0.8)

def decomposition_series2(df, df_tv_int, vars_entrada, vars_salida, caso, w):
    df_original = df.copy()
    df_tv_int_original = df_tv_int.copy()

    df = pd.DataFrame(columns=vars_entrada)
    df_tendencia = pd.DataFrame(columns=vars_entrada)
    df_tendencia['Date'] = df_tv_int_original['Date']
    df_tv_int = pd.DataFrame(columns=vars_salida)
    df_tv_int_tendencia = pd.DataFrame(columns=vars_salida)
    df_tv_int_tendencia['Date'] = df_tv_int_original['Date']

    # modelo = 'multiplicative'
    modelo = 'additive'

    #     VARIABLES DE ENTRADA — DESCOMPOSICIÓN D3C
    for i in [var for var in vars_entrada if var not in ['Hours', 'Date']]:
        series = pd.Series(df_original[i].values, index=pd.to_datetime(df_original['Date'])).dropna()

        # if (series > 0).all():
        #     offset = 0
        # else:
        #     offset = abs(series.min()) + 1e-10
        
        # series = series + offset

        decomposition = seasonal_decompose(series, model=modelo, period=w, extrapolate_trend='freq', two_sided= False)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        resid = decomposition.resid
        # --- Reconstrucción ---
        complete = resid + seasonal

        if caso == 1:   #Sin descomposición
            df_tendencia[i] = 0
            df[i] = df_original[i]
            
        else:           #Con descomposición
            df_tendencia[i] = trend.values
            df[i] = complete.values

    #     VARIABLES DE SALIDA — DESCOMPOSICIÓN D3C
    for i in [var for var in vars_salida if var not in ['Date']]:
        
        series = pd.Series(df_tv_int_original[i].values, index=pd.to_datetime(df_tv_int_original['Date'])).dropna()
        decomposition = seasonal_decompose(series, model=modelo, period=w, extrapolate_trend='freq', two_sided= False)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        resid = decomposition.resid
        complete = seasonal.values + resid.values

        if caso == 1:   #Sin descomposición
            df_tv_int_tendencia[i] = 0
            df_tv_int[i] = df_tv_int_original[i]

        elif caso == 2 or caso == 6 or caso == 7:   #Basado en la tendencia real de Tint
            df_tv_int_tendencia[i] = trend.values
            df_tv_int[i] = complete

        elif caso == 3: # Basado en la tendencia de la temperatura ambiente Ta
            df_tv_int_tendencia[i] = df_tendencia['Temp Out ']
            df_tv_int[i] = df_tv_int_original[i] - df_tendencia['Temp Out ']

    mean_trend = df_tv_int_tendencia[['Tv_int1','Tv_int6', 'Tv_int10']].mean(axis=1)     # Para el caso 7   
    for i in range(1, 11):
        if caso == 4: # Basado en la tendencia de cada punto de Tsup
            df_tv_int_tendencia[f'Tv_int{i}'] = df_tendencia[f'CEP{i}']
            df_tv_int[f'Tv_int{i}'] = df_tv_int_original[f'Tv_int{i}'] - df_tv_int_tendencia[f'Tv_int{i}']

        elif caso == 5: # Basado en el promedio de las tendencias de Ta y Tsup
            peso_temp = 0.5
            peso_cep = 1 - peso_temp
            df_tv_int_tendencia[f'Tv_int{i}'] = (df_tendencia['Temp Out '] * peso_temp) + (df_tendencia[f'CEP{i}'] * peso_cep)  
            df_tv_int[f'Tv_int{i}'] = df_tv_int_original[f'Tv_int{i}'] - df_tv_int_tendencia[f'Tv_int{i}']

        elif caso == 6:  #Basado en la tendencia real de 3 puntos (Tint,1 Tint,6 y Tint,10)
            Tint_real = {1:1, 2:1, 3:6, 4:6, 5:6, 6:6, 7:6, 8:10, 9:10, 10:10}
            df_tv_int[f'Tv_int{i}'] = df_tv_int_original[f'Tv_int{i}'] - df_tv_int_tendencia[f'Tv_int{Tint_real[i]}']
            df_tv_int_tendencia[f'Tv_int{i}'] = df_tv_int_tendencia[f'Tv_int{Tint_real[i]}']

        elif caso == 7:  #Basado en la tendencia real de 3 puntos (Tint,1 Tint,6 y Tint,10)
            df_tv_int_tendencia[f'Tv_int{i}'] = mean_trend
            df_tv_int[f'Tv_int{i}'] = df_tv_int_original[f'Tv_int{i}'] - df_tv_int_tendencia[f'Tv_int{i}']

    df['Date'] = df_original['Date'].copy()
    df_tendencia['Date'] = df_original['Date'].copy()
    df_tv_int['Date'] = df_tv_int_original['Date'].copy()
    df_tv_int.dropna(inplace=True)
    
    return df, df_tendencia, df_tv_int, df_tv_int_tendencia


dias = 3
caso = 6
w = int(20*dias)
df, df_tendencia, df_tv_int, df_tv_int_tendencia = decomposition_series2(df, df_tv_int, vars_entrada, vars_salida, caso, w)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ANÁLSIS EXPLORATORIO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
X = df.drop(columns='Date').copy()
Y = df_tv_int.drop(columns='Date').copy()

#Configuración
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.size'] = 17  # tamaño de letra global

# Configuración de la división temporal
n_splits = 10
index = [1, 6, 10]
tscv = TimeSeriesSplit(n_splits=n_splits)
scaler_x = StandardScaler()
scaler_y = StandardScaler()

for fold_idx, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
    y_train, y_test = Y.iloc[train_index].copy(), Y.iloc[test_index].copy() 

    # Escalamiento
    X_train_sc = scaler_x.fit_transform(X_train)
    y_train_sc = scaler_y.fit_transform(y_train)

    n_vars = X_train.shape[1]
    
    # --- 1. CÁLCULO DE PCA (Lógica Original) ---
    pca = PCA(n_components=n_vars, random_state=22)
    pca.fit(X_train_sc)
    pca_var_cum = np.cumsum(pca.explained_variance_ratio_) * 100

    # --- 2. CÁLCULO DE PLS (Lógica Robusta Q1) ---
    r2x_cum, r2y_cum, q2_cum = [], [], []
    r2y_cum_ind = {j: [] for j in index}

    
    for i in range(1, n_vars + 1):
        pls = PLSRegression(n_components=i, scale=False)
        pls.fit(X_train_sc, y_train_sc)
        
        # R2X_cum (Varianza capturada de X por PLS)
        X_hat = np.dot(pls.x_scores_, pls.x_loadings_.T)
        r2x = 1 - (np.sum((X_train_sc - X_hat)**2) / np.sum(X_train_sc**2))
        r2x_cum.append(r2x * 100)
        
        # R2Y_cum (Ajuste Global)
        r2y_cum.append(pls.score(X_train_sc, y_train_sc) * 100)

        # R2Y_cum_ind (Ajuste individual)
        y_pred = pls.predict(X_train_sc)

        for j in index:
            y_true_j = y_train_sc[:, j - 1]
            y_pred_j = y_pred[:, j - 1]
            
            r2_j = 1 - (np.sum((y_true_j - y_pred_j)**2) / np.sum(y_true_j**2))
            r2y_cum_ind[j].append(r2_j * 100)
        
        # Q2 (Predictividad mediante CV interna)
        cv_internal = KFold(n_splits=5, shuffle=True, random_state=22)
        y_cv = cross_val_predict(pls, X_train_sc, y_train_sc, cv=cv_internal)
        q2 = 1 - (np.sum((y_train_sc - y_cv)**2) / np.sum((y_train_sc - y_train_sc.mean())**2))
        q2_cum.append(q2 * 100)


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ENTRENAMIENTO PCR Y PLSR - TSCV <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
X = df.iloc[:x_train_index].drop(columns='Date').copy()
Xtest  = df.iloc[x_train_index:].drop(columns='Date').copy()
Y = df_tv_int.iloc[:x_train_index].drop(columns='Date').copy()

n_splits = 10
tscv = TimeSeriesSplit(n_splits=n_splits)
cont = 0
test_results_pcr = []
test_results_pls = []
# --- NUEVO: listas para guardar resultados de entrenamiento ---
train_results_pcr = []
train_results_pls = []

for fold_idx, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
    y_train, y_test = Y.iloc[train_index].copy(), Y.iloc[test_index].copy()

    # Escalado de predictores
    scaler_X = StandardScaler()
    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc = scaler_X.transform(X_test)

    # Límite quimiométrico: k_max <= min(n/10, p-1)
    #max_k = int(min(len(X_train) / 10, len(X.columns) - 1))  # R E V I S A R
    max_k = len(vars_entrada) - 1
    for n_comp in range(1, max_k + 1):
        # --- BLOQUE PCR  ---
        pca = PCA(n_components=n_comp)
        X_tr_pca = pca.fit_transform(X_train_sc)
        X_te_pca = pca.transform(X_test_sc)
        X_tr_pca_c = sm.add_constant(X_tr_pca)  # Columna de unos para el intercepto
        X_te_pca_c = sm.add_constant(X_te_pca)

        k = 0
        for col in y_train.columns:
            k += 1
            cont += 1
            # Series base
            y_tr_raw = y_train[col].values
            y_te_raw = y_test[col].values

            # Tendencias y series originales completas (para métricas finales)
            tendencia_tr = df_tv_int_tendencia.loc[y_train.index, col].values
            y_real_full_tr = df_tv_int.loc[y_train.index, col].values  + tendencia_tr

            tendencia = df_tv_int_tendencia.loc[y_test.index, col].values
            y_real_full = df_tv_int.loc[y_test.index, col].values + tendencia

            # Estimación PCR con HAC (coeficientes iguales a OLS; HAC corrige varianzas)
            model_pcr = sm.OLS(y_tr_raw, X_tr_pca_c).fit()
            m_hac_pcr = model_pcr.get_robustcov_results(cov_type='HAC', maxlags=1)

            # Desescalado completo de coeficientes hacia variables originales
            beta_pca = m_hac_pcr.params[1:]
            coef_orig_pcr = (beta_pca @ pca.components_) / scaler_X.scale_
            intercept_orig_pcr = m_hac_pcr.params[0] - np.dot(coef_orig_pcr, scaler_X.mean_)

            # Predicciones PCR
            y_pred_tr_detrend = model_pcr.predict(X_tr_pca_c)  # predicción en train (sin tendencia)
            y_pred_te_detrend = model_pcr.predict(X_te_pca_c)

            # --- Métricas de entrenamiento (PCR) ---
            train_results_pcr.append({
                'fold': fold_idx, 'n_comp': n_comp, 'variable': col,
                'RMSE_train': np.sqrt(mean_squared_error(y_real_full_tr, y_pred_tr_detrend + tendencia_tr)),
                'R2_train': r2_score(y_real_full_tr, y_pred_tr_detrend + tendencia_tr),
                'DW_train': durbin_watson(model_pcr.resid),
                'BP_pvalue': het_breuschpagan(model_pcr.resid, X_tr_pca_c)[1],
                'Intercept_orig': intercept_orig_pcr,
                **{f'coef_{v}': c for v, c in zip(X.columns, coef_orig_pcr)}
            })

            # --- Métricas de prueba (PCR) ---
            test_results_pcr.append({
                'fold': fold_idx, 'n_comp': n_comp, 'variable': col,
                'RMSE_test': np.sqrt(mean_squared_error(y_real_full, y_pred_te_detrend + tendencia)),
                'R2_test': r2_score(y_real_full, y_pred_te_detrend + tendencia),
                'DW_train': durbin_watson(model_pcr.resid),
                'BP_pvalue': het_breuschpagan(model_pcr.resid, X_tr_pca_c)[1],
                'Intercept_orig': intercept_orig_pcr,
                **{f'coef_{v}': c for v, c in zip(X.columns, coef_orig_pcr)}
            })

            # --- BLOQUE PLS (Enfoque Predictivo) ---

            scaler_y = StandardScaler()
            y_tr_sc = scaler_y.fit_transform(y_tr_raw.reshape(-1, 1)).ravel()

            pls = PLSRegression(n_components=n_comp, scale=False)
            pls.fit(X_train_sc, y_tr_sc)

            # Predicción y back-transform (test)
            y_pred_te_sc = pls.predict(X_test_sc).ravel()
            y_pred_te_detrend_pls = scaler_y.inverse_transform(y_pred_te_sc.reshape(-1, 1)).ravel()

            # Predicción y back-transform (train)
            y_pred_tr_sc = pls.predict(X_train_sc).ravel()  # NUEVO
            y_pred_tr_detrend_pls = scaler_y.inverse_transform(y_pred_tr_sc.reshape(-1, 1)).ravel()  # NUEVO

            # Coeficientes PLS desescalados
            coef_orig_pls = (pls.coef_.ravel() * scaler_y.scale_[0]) / scaler_X.scale_
            intercept_orig_pls = scaler_y.mean_[0] - np.dot(coef_orig_pls, scaler_X.mean_)

            # --- NUEVO: guardar métricas de entrenamiento (PLS) ---
            train_results_pls.append({
                'fold': fold_idx, 'n_comp': n_comp, 'variable': col,
                'RMSE_train': np.sqrt(mean_squared_error(y_real_full_tr, y_pred_tr_detrend_pls + tendencia_tr)),
                'R2_train': r2_score(y_real_full_tr, y_pred_tr_detrend_pls + tendencia_tr),
                'R2_train_scaled': pls.score(X_train_sc, y_tr_sc),  # se mantiene
                'Intercept_orig': intercept_orig_pls,
                **{f'coef_{v}': c for v, c in zip(X.columns, coef_orig_pls)}})

            # Resultados de prueba (PLS) - SIN CAMBIOS
            test_results_pls.append({
                'fold': fold_idx, 'n_comp': n_comp, 'variable': col,
                'RMSE_test': np.sqrt(mean_squared_error(y_real_full, y_pred_te_detrend_pls + tendencia)),
                'R2_test': r2_score(y_real_full, y_pred_te_detrend_pls + tendencia),
                'R2_train_scaled': pls.score(X_train_sc, y_tr_sc),
                'Intercept_orig': intercept_orig_pls,
                **{f'coef_{v}': c for v, c in zip(X.columns, coef_orig_pls)}})

# DataFrames finales
test_results_pcr = pd.DataFrame(test_results_pcr)
test_results_pls = pd.DataFrame(test_results_pls)

# --- NUEVO: DataFrames para resultados de entrenamiento ---
train_results_pcr = pd.DataFrame(train_results_pcr)
train_results_pls = pd.DataFrame(train_results_pls)

pcr_opt, pcr_stats = get_optimal_comp(test_results_pcr, 'n_comp')
pls_opt, pls_stats = get_optimal_comp(test_results_pls, 'n_comp')
print(f"Número óptimo de componentes (Regla 1-SE): PCR={pcr_opt}, PLS={pls_opt}")

df_bestmodel_pcr, df_bestmodel_ori_pcr = build_bestmodels_detrended(modelo='PCR')
df_bestmodel_plsr, df_bestmodel_ori_plsr = build_bestmodels_detrended(modelo='PLSR')


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> PREDICCIONES Y MÉTRICAS EN GRUPO INDEPENDIENTE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Configuración de estilo
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.size'] = 15  # tamaño de letra global

target_vars = vars_salida

xlim_start = df_tv_int_original.loc[x_train_index, 'Date']
xlim_end =  df_tv_int_original.loc[df_tv_int_original.shape[0] - 1,'Date']

start_num = pd.Timestamp(xlim_start).timestamp()
end_num = pd.Timestamp(xlim_end).timestamp()

cols = 2
rows = int(np.ceil(len(target_vars) / cols))

for i, var in enumerate(target_vars):
    matriz_aux = pd.DataFrame(columns=vars_salida)
    ajust_label = ajust_name.get(var, var)

    # 1) Observado
    y_obs_full = df_tv_int_original.loc[Xtest.index, var].astype(float).rename(f"yobs_{var}")# - df_tendencia.loc[Xtest.index, 'Temp Out '].astype(float)

    # ===== PCR =====
    # 2) Predicción sin tendencia (PCR)
    y_pred_det_pcr = predict_temp(Xtest, df_bestmodel_ori_pcr, var)
    # 3) Suma de tendencia y métricas (PCR)
    tendencia = df_tv_int_tendencia.loc[Xtest.index, var].astype(float)
    y_pred_full_pcr = (y_pred_det_pcr + tendencia).rename(f"yhat_{var}_pcr")
    y_pred_full_pcr = y_pred_full_pcr.rename(f"yhat_{var}_pcr")
    metrics_pcr = compute_metrics(y_obs_full, y_pred_full_pcr)
    df_eval_pcr = metrics_pcr["data"]  # Contiene columnas: 'obs', 'pred'

    # ===== PLSR =====
    # 5) Predicción sin tendencia (PLSR)
    y_pred_det_plsr = predict_temp(Xtest, df_bestmodel_ori_plsr, var)
    # 6) Suma de tendencia y métricas (PLSR)
    y_pred_full_plsr = (y_pred_det_plsr + tendencia).rename(f"yhat_{var}_plsr")
    metrics_plsr = compute_metrics(y_obs_full, y_pred_full_plsr)
    df_eval_plsr = pd.concat([y_obs_full.rename("obs"), y_pred_full_plsr.rename("pred")], axis=1).dropna()
