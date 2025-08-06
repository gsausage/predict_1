# predict_1
!pip install arch optuna
import yfinance as yf
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from arch import arch_model
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, classification_report
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import optuna

warnings.filterwarnings("ignore")

# 설정
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

analysis_start_date = "2003-08-01"
end_date = "2023-08-01"
buffer_days = 150
window_size = 100
forecast_horizon = 1
window_len = 30

download_start_date = (pd.to_datetime(analysis_start_date) - pd.tseries.offsets.BDay(buffer_days)).strftime('%Y-%m-%d')
assets = {
    'KOSPI': '^KS11', 'S&P500': '^GSPC', 'SSE': '000001.SS', 'Gold': 'GC=F',
    'CrudeOil': 'CL=F', 'USD_KRW': 'KRW=X', 'VIX': '^VIX', 'US10Y': '^TNX'
}

# 데이터 다운로드
raw_data = yf.download(list(assets.values()), start=download_start_date, end=end_date)['Close']
raw_data.columns = assets.keys()
data = raw_data.dropna()
returns = np.log(data).diff().dropna()

from scipy.stats import norm

def compute_var_es(series, alpha=0.01):  # 99% 기준
    mu = series.mean()
    sigma = series.std()
    var = norm.ppf(alpha) * sigma
    es = mu - sigma * norm.pdf(norm.ppf(alpha)) / alpha
    return var, es

# 30일 이동 윈도우 기준으로 VaR & ES 계산
window = 30
var_list, es_list, idx_list = [], [], []
for i in range(window, len(returns)):
    window_data = returns['KOSPI'].iloc[i - window:i]
    var, es = compute_var_es(window_data)
    var_list.append(var)
    es_list.append(es)
    idx_list.append(returns.index[i])

var_es_df = pd.DataFrame({
    'VaR': var_list,
    'ES': es_list
}, index=idx_list)


# GJR-GARCH 표준화 잔차 계산
def get_standardized_resid(returns):
    std_resids = {}
    for col in returns.columns:
        am = arch_model(returns[col]*100, vol='Garch', p=1, o=1, q=1, dist='t')
        res = am.fit(disp='off')
        std_resids[col] = res.resid / res.conditional_volatility
    return pd.DataFrame(std_resids).dropna()

standardized_resid = get_standardized_resid(returns)

# DCC-GARCH 상관계수 예측 함수
def dcc_forecast_estimated(eps_window):
    T, N = eps_window.shape
    Qbar = np.cov(eps_window.T)
    def dcc_likelihood(params):
        a, b = params
        if a < 0 or b < 0 or a + b >= 1: return 1e6
        Qt = Qbar.copy()
        loglik = 0
        for t in range(1, T):
            et = eps_window[t-1].reshape(-1, 1)
            Qt = (1 - a - b) * Qbar + a * (et @ et.T) + b * Qt
            D = np.diag(1 / np.sqrt(np.diag(Qt)))
            Rt = D @ Qt @ D
            eps_t = eps_window[t].reshape(-1, 1)
            loglik += np.log(np.linalg.det(Rt)) + eps_t.T @ np.linalg.inv(Rt) @ eps_t
        return loglik.flatten()[0]
    from scipy.optimize import minimize
    res = minimize(dcc_likelihood, x0=[0.03, 0.94], bounds=[(0,1), (0,1)])
    a_hat, b_hat = res.x
    Qt = Qbar.copy()
    for t in range(T):
        et = eps_window[t].reshape(-1, 1)
        Qt = (1 - a_hat - b_hat) * Qbar + a_hat * (et @ et.T) + b_hat * Qt
    et = eps_window[-1].reshape(-1, 1)
    Q_forecast = (1 - a_hat - b_hat) * Qbar + a_hat * (et @ et.T) + b_hat * Qt
    D = np.diag(1 / np.sqrt(np.diag(Q_forecast)))
    return D @ Q_forecast @ D

# DCC 예측 상관계수 계산
import os
import pickle

# 체크포인트 파일 경로
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
dcc_path = os.path.join(checkpoint_dir, "dcc_forecast.pkl")

# 1. 체크포인트 로드 여부 확인
if os.path.exists(dcc_path):
    with open(dcc_path, "rb") as f:
        dcc_corrs, forecast_dates = pickle.load(f)
    print("✅ DCC forecast 로드 완료 (checkpoint 사용됨)")
else:
      dcc_corrs, forecast_dates = {}, []
      extra_pairs = [('S&P500', 'VIX'), ('USD_KRW', 'US10Y')]
      for i in tqdm(range(window_size, len(standardized_resid) - forecast_horizon)):
          window = standardized_resid.iloc[i - window_size:i]
          R = dcc_forecast_estimated(window.values)
          forecast_date = standardized_resid.index[i + forecast_horizon]
          for col in returns.columns:
              if col == 'KOSPI': continue
              key = f'KOSPI-{col}'
              dcc_corrs.setdefault(key, []).append(R[returns.columns.get_loc('KOSPI'), returns.columns.get_loc(col)])
          for a1, a2 in extra_pairs:
              idx1, idx2 = returns.columns.get_loc(a1), returns.columns.get_loc(a2)
              key = f'{a1}-{a2}'
              dcc_corrs.setdefault(key, []).append(R[idx1, idx2])
          forecast_dates.append(forecast_date)
# 3. 체크포인트 저장
      with open(dcc_path, "wb") as f:
          pickle.dump((dcc_corrs, forecast_dates), f)
          print("✅ DCC forecast 계산 완료 및 checkpoint 저장됨")
# 2. forecast_df 초기화
forecast_df = pd.DataFrame(dcc_corrs, index=forecast_dates)
forecast_df = forecast_df[forecast_df.index >= analysis_start_date]

# 3. VaR & ES 병합
forecast_df = forecast_df.merge(var_es_df, left_index=True, right_index=True, how='left')
forecast_df = forecast_df.dropna()

#  KOSPI 수익률 기준 충격일 정의 (-2% 이상 하락)
kospi_returns = returns['KOSPI']
shock_dates = kospi_returns[kospi_returns <= -0.02].index
print(f" 충격일 개수 (KOSPI -2% 이상 하락): {len(shock_dates)}일")

# 5. 상관계수 중요 변수 선택 (RandomForest)
y_label = forecast_df.index.isin(shock_dates).astype(int)
df_clean = forecast_df.dropna()
y_label = y_label[-len(df_clean):]
X_scaled = StandardScaler().fit_transform(df_clean.values)
rf = RandomForestClassifier(n_estimators=200, random_state=SEED).fit(X_scaled, y_label)
importances = rf.feature_importances_# 6. 주요 상관계수 + VaR + ES 포함한 컬럼만 선택
top_corr_cols = df_clean.columns[np.argsort(importances)][-5:]
final_cols = list(top_corr_cols) + ['VaR', 'ES']
forecast_df = forecast_df[final_cols]

# LSTM 입력 데이터 구성 및 시계열 기반 자동 분할
shock_extended = set()
for date in shock_dates:
    for offset in range(-2, 1):
        shock_extended.add(date + pd.Timedelta(days=offset))
shock_dates_extended = pd.DatetimeIndex(sorted(shock_extended))

forecast_idx = forecast_df.index
X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []

for idx in range(window_len, len(forecast_df)):
    current_date = forecast_idx[idx]
    window = forecast_df.iloc[idx - window_len: idx].values
    if window.shape[0] != window_len:
        continue
    label = 1 if current_date in shock_dates_extended else 0

    # 날짜 기준으로 자동 분할
    if current_date <= pd.Timestamp('204-12-31'):
        X_train.append(window)
        y_train.append(label)
    elif current_date <= pd.Timestamp('2016-06-30'):
        X_val.append(window)
        y_val.append(label)
    else:
        X_test.append(window)
        y_test.append(label)

# 리스트를 numpy array로 변환
X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_test, y_test = np.array(X_test), np.array(y_test)

# Jitter를 활용한 증강
import numpy as np
np.random.seed(SEED)

def jitter(x, sigma=0.01):
    noise = np.random.normal(loc=0.0, scale=sigma, size=x.shape)
    return x + noise

augmented_X, augmented_y = [], []
for xi, yi in zip(X_train, y_train):
    if yi == 1:
        for _ in range(10):  # 5배 증강
            augmented_X.append(jitter(xi))
            augmented_y.append(1)

# numpy array로 변환 후 병합
X_train = np.concatenate([X_train, np.array(augmented_X)], axis=0)
y_train = np.concatenate([y_train, np.array(augmented_y)], axis=0)

import tensorflow.keras.backend as K
import tensorflow as tf

def focal_loss(gamma=1.5, alpha=0.9):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        loss = -alpha_t * K.pow(1 - p_t, gamma) * K.log(p_t)
        return K.mean(loss)
    return focal_loss_fixed


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# ✅ focal loss 사용 (클래스 불균형 대응)
model.compile(
    optimizer='adam',
    loss=focal_loss(gamma=2.0, alpha=0.75),
    metrics=['accuracy']
)



class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    class_weight=class_weight_dict,
    verbose=1
)

# 예측 및 평가
pred_dates, pred_probs = [], []
for i in range(window_len, len(forecast_df)):
    window = forecast_df.iloc[i - window_len:i].values
    if window.shape[0] != window_len:
        continue
    prob = model.predict(window.reshape(1, window_len, -1), verbose=0)[0][0]
    pred_probs.append(prob)
    pred_dates.append(forecast_df.index[i])
pred_result_df = pd.DataFrame({'date': pred_dates, 'prob': pred_probs}).set_index('date')
pred_result_df['true_label'] = pred_result_df.index.isin(shock_dates_extended).astype(int)

# 7. Optuna + F2 최적화
import optuna
from sklearn.metrics import recall_score

# 🔁 검증셋 범위 (validation set: 2023-01-01 ~ 2024-06-30)
val_mask = (pred_result_df.index >= pd.Timestamp("2014-01-01")) & (pred_result_df.index <= pd.Timestamp("2016-06-30"))
val_df = pred_result_df[val_mask].copy()

from sklearn.metrics import fbeta_score
def filter_consecutive_ones(series, min_days):
    result = pd.Series(0, index=series.index)
    count = 0

    for i in range(len(series)):
        if series.iloc[i] == 1:
            count += 1
        else:
            if count >= min_days:
                result.iloc[i - count:i] = 1
            count = 0
    # 마지막 시퀀스가 조건 만족하는지 확인
    if count >= min_days:
        result.iloc[len(series) - count:] = 1

    return result

def objective(trial):
    threshold = trial.suggest_float("threshold", 0.01, 0.6)
    delta = trial.suggest_float("delta", 0, 0.1)
    corr_thresh = trial.suggest_float("corr_thresh", 0, 0.02)
    min_consecutive_days = trial.suggest_int("min_consecutive_days", 1, 5)  # 추가됨

    val_df['prev_mean'] = val_df['prob'].rolling(5, min_periods=1).mean()
    rule_filter = (
        (val_df['prob'] > threshold) &
        ((val_df['prob'] - val_df['prev_mean']) > delta)
    )

    dcc_vol = forecast_df.diff().abs().mean(axis=1)
    val_df['dcc_volatility'] = dcc_vol.reindex(val_df.index)
    dcc_filter = val_df['dcc_volatility'] > corr_thresh

    val_df['predicted_raw'] = (rule_filter & dcc_filter).astype(int)

    # ✅ 연속된 1이 min_consecutive_days 이상인 경우만 경고로 인정
    val_df['predicted_label'] = filter_consecutive_ones(val_df['predicted_raw'], min_consecutive_days)

    return fbeta_score(val_df['true_label'], val_df['predicted_label'], beta=1.5)


# ✅ Optuna 실행
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1000)
best_params = study.best_params
print("\n📌 Optuna Best Params (from Validation Set):", best_params)

# 전체 데이터에 best_params 적용하여 라벨링
pred_result_df['prev_mean'] = pred_result_df['prob'].rolling(5, min_periods=1).mean()
rule_filter = (
    (pred_result_df['prob'] > best_params['threshold']) &
    ((pred_result_df['prob'] - pred_result_df['prev_mean']) > best_params['delta'])
)
dcc_cols = [col for col in forecast_df.columns if '-' in col]
rolling_std = forecast_df[dcc_cols].rolling(window=5, min_periods=1).std()
pred_result_df['dcc_volatility'] = rolling_std.mean(axis=1).reindex(pred_result_df.index)
dcc_filter = pred_result_df['dcc_volatility'] > best_params['corr_thresh']
pred_result_df['predicted_label'] = (rule_filter & dcc_filter).astype(int)


# 테스트셋 마스크 생성
test_mask = pred_result_df.index >= pd.Timestamp("2016-07-01")
test_df = pred_result_df[test_mask]

print("\n📊 테스트셋 평가 결과:")
print(classification_report(test_df['true_label'], test_df['predicted_label']))
print("AUC:", roc_auc_score(test_df['true_label'], test_df['prob']))

# 마지막 날짜 정보 출력 (테스트셋 내에서)
last_date = test_df.index.max()
row = test_df.loc[last_date]
print(f"\n마지막 테스트셋 날짜: {last_date.date()}")
print(f"예측 확률: {row['prob']:.4f}")
print(f"이전 평균: {row['prev_mean']:.4f}")
print(f"DCC 변동성: {row['dcc_volatility']:.4f}")
print(f"예측 라벨: {'경고' if row['predicted_label']==1 else '비경고'}")
print(f"실제 라벨: {'충격일' if row['true_label']==1 else '비충격일'}")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# Confusion Matrix (테스트셋 기준)
cm = confusion_matrix(test_df['true_label'], test_df['predicted_label'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["비충격일", "충격일"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (테스트셋 기준)")
plt.grid(False)
plt.tight_layout()
plt.show()

# 1. KOSPI 수익률 정렬 (pred_result_df에 맞춤)
returns_kospi_bt = returns['KOSPI'].reindex(pred_result_df.index).fillna(0)

# 2. 포지션 설정: 예측 경보 없을 때만 주식 보유
position = (~pred_result_df['predicted_label'].astype(bool)).astype(int)

# 3. 전략 수익률 계산
strategy_returns = returns_kospi_bt * position

# 4. 누적 수익률 계산
cum_strategy = (strategy_returns + 1).cumprod()
cum_bh = (returns_kospi_bt + 1).cumprod()

# 5. 백테스트 결과 정리
bt_result_df = pd.DataFrame({
    "Buy & Hold": cum_bh,
    "Model Strategy": cum_strategy
})
import matplotlib.pyplot as plt

# 테스트셋만 필터링
test_df = pred_result_df[pred_result_df.index >= pd.Timestamp("2016-07-01")].copy()

plt.figure(figsize=(15, 5))

# 예측 확률 라인
plt.plot(test_df.index, test_df['prob'], label='Predicted Probability', color='black', linewidth=1)

# Threshold 라인 (Optuna에서 찾은 best threshold)
plt.axhline(best_params['threshold'], color='red', linestyle='--', label=f"Threshold = {best_params['threshold']:.2f}")

# 충격일 표시
shock_dates = test_df[test_df['true_label'] == 1].index
plt.scatter(shock_dates, test_df.loc[shock_dates, 'prob'], color='blue', label='Actual Shock', marker='x', s=80)

# 예측된 경고일 표시
predicted_shocks = test_df[test_df['predicted_label'] == 1].index
plt.scatter(predicted_shocks, test_df.loc[predicted_shocks, 'prob'], color='orange', label='Predicted Warning', marker='o', facecolors='none', edgecolors='orange', s=80)

# 시각적 요소
plt.title("테스트셋 예측 확률 시계열")
plt.xlabel("Date")
plt.ylabel("Predicted Probability")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
pred_result_df['prob'].hist(bins=50, color='skyblue', edgecolor='black')
plt.title("LSTM 예측 확률 분포")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(bt_result_df.index, bt_result_df["Buy & Hold"], label="Buy & Hold", linestyle='--')
plt.plot(bt_result_df.index, bt_result_df["Model Strategy"], label="Model Strategy", linewidth=2)
plt.title(" 누적 수익률 비교 (백테스트)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

def compare_hedged_vs_original_portfolio_script(pred_result_df, forecast_df, returns, window=60):
    import matplotlib.pyplot as plt

    def construct_partial_cov_hedge_portfolio(forecast_df, returns, target_asset='KOSPI', date=None, window=60):
        if date is None:
            date = forecast_df.index[-1]
        else:
            date = pd.to_datetime(date)
            if date not in forecast_df.index:
                date = forecast_df.index[forecast_df.index.get_indexer([date], method='nearest')[0]]

        hedge_assets = [col.split('-')[1] for col in forecast_df.columns if col.startswith(f'{target_asset}-')]
        all_assets = [target_asset] + hedge_assets

        aligned_returns = returns[all_assets].dropna()
        recent_returns = aligned_returns.loc[:date].iloc[-window:]
        std_devs = recent_returns.std()

        corr_matrix = np.eye(len(all_assets))
        asset_idx = {a: i for i, a in enumerate(all_assets)}
        for col in forecast_df.columns:
            a1, a2 = col.split('-')
            if a1 == target_asset and a2 in hedge_assets:
                i, j = asset_idx[a1], asset_idx[a2]
                corr = forecast_df.loc[date, col]
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

        cov_matrix = np.outer(std_devs.values, std_devs.values) * corr_matrix

        sigma_kk = cov_matrix[0, 0]
        sigma_kw = cov_matrix[0, 1:]
        sigma_ww = cov_matrix[1:, 1:]

        opt_w = -np.linalg.pinv(sigma_ww) @ sigma_kw.T
        var_hedged = sigma_kk + 2 * opt_w.T @ sigma_kw + opt_w.T @ sigma_ww @ opt_w

        return {
            'date': date,
            'target_variance': sigma_kk,
            'hedged_variance': float(var_hedged),
            'reduction_ratio': 1 - float(var_hedged) / sigma_kk,
            'weights': dict(zip(hedge_assets, opt_w))
        }

    shock_dates = pred_result_df[pred_result_df['predicted_label'] == 1].index
    hedge_returns = []
    kospi_returns = []

    for shock_day in shock_dates:
        if shock_day not in forecast_df.index or shock_day not in returns.index:
            continue

        hedge_result = construct_partial_cov_hedge_portfolio(
            forecast_df=forecast_df,
            returns=returns,
            target_asset='KOSPI',
            date=shock_day,
            window=window
        )

        weights = hedge_result['weights']
        aligned_assets = ['KOSPI'] + list(weights.keys())

        try:
            one_day_returns = returns.loc[shock_day, aligned_assets]
        except KeyError:
            continue

        hedge_r = one_day_returns['KOSPI'] + sum(
            one_day_returns[asset] * w for asset, w in weights.items()
        )

        hedge_returns.append(hedge_r)
        kospi_returns.append(one_day_returns['KOSPI'])

    hedge_returns = pd.Series(hedge_returns, index=shock_dates[:len(hedge_returns)])
    kospi_returns = pd.Series(kospi_returns, index=shock_dates[:len(kospi_returns)])

    # 누적 수익률
    cum_kospi = (1 + kospi_returns).cumprod()
    cum_hedged = (1 + hedge_returns).cumprod()

    # 시각화: 누적 수익률
    plt.figure(figsize=(12, 6))
    plt.plot(cum_kospi, label="KOSPI 단독 (충격일)", linewidth=2)
    plt.plot(cum_hedged, label="헷지 포트폴리오 (충격일)", linewidth=2, linestyle='--')
    plt.title("충격일 누적 수익률 비교")
    plt.ylabel("누적 수익률")
    plt.xlabel("날짜")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 시각화: 분산 비교
    plt.figure(figsize=(10, 4))
    plt.bar(['KOSPI'], [kospi_returns.var()], label='KOSPI')  
    plt.bar(['헷지 포트폴리오'], [hedge_returns.var()], label='헷지 포트폴리오')
    plt.title("충격일 수익률 분산 비교")
    plt.ylabel("분산")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    return kospi_returns.describe(), hedge_returns.describe()






