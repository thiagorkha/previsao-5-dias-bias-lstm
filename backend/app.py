# app.py (ou stock_analysis_backend.py)
from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import logging
import pickle
import os
from datetime import datetime, timedelta
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, field
import sys
import json
import traceback

# Importações específicas para LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Importar apenas se TensorFlow estiver instalado e for necessário
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    import tensorflow as tf
    LSTM_AVAILABLE = True
except ImportError:
    print("TensorFlow/Keras não encontrado. Funções LSTM serão desativadas.")
    LSTM_AVAILABLE = False


from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD

# Importações específicas para Bias-Variance
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.base import clone
from mlxtend.evaluate import bias_variance_decomp
from statsmodels.tsa.stattools import adfuller
from itertools import product
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app) # Habilita CORS para permitir requisições do frontend React

# --- NOVA ROTA PARA ACESSO DIRETO VIA NAVEGADOR ---
@app.route('/', methods=['GET'])
def home():
    return "Bem-vindo ao serviço de Análise de Ações! Use o endpoint /analyze_stock com um método POST para análises."
# --- FIM DA NOVA ROTA ---

# Configure logging to go to a file and stderr, not stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_predictor.log"),
        logging.StreamHandler(sys.stderr) # Explicitly send to stderr
    ]
)
logger = logging.getLogger(__name__)


# --- CLASSE DE CONFIGURAÇÃO INTEGRADA ---
@dataclass
class Config:
    TICKER: str = "b3sa3.SA"
    PERIOD: str = "1y"
    DIAS_PREVISAO: int = 5
    LOOKBACK: int = 60
    EPOCHS: int = 100
    BATCH_SIZE: int = 32
    START_DATE_OFFSET_YEARS: int = 5
    FEATURES_LSTM: List[str] = field(default_factory=lambda: [
        "Close", "SMA_20", "RSI", "MACD", "BollingerBands_Upper", "BollingerBands_Lower"
    ])
    MODEL_PATH: str = "lstm_model.h5"
    DATA_PATH: str = "dados_acao.csv"
    PLOT_PATH: str = "previsao_acao.png"
    TRAIN_SPLIT_RATIO: float = 0.8
    PLOT_LAST_DAYS: int = 100
    CACHE_DIR: str = "cache"
    LOG_FILE: str = "stock_predictor.log"
    EARLY_STOPPING_PATIENCE: int = 15
    LSTM_UNITS: List[int] = field(default_factory=lambda: [100, 50])
    DENSE_UNITS: List[int] = field(default_factory=lambda: [50])
    DROPOUT_RATE: float = 0.3

    WINDOW_SIZE_BIAS_VARIANCE: int = 250
    STEP_SIZE_BIAS_VARIANCE: int = 1
    N_COMPONENTS_PCA: int = 4
    NUM_ROUNDS_BIAS_VARIANCE: int = 100
    INITIAL_CAPITAL_BACKTEST: int = 10000
    CAPITAL_DEPLOYED_PER_TRADE: float = 0.2

app_config = Config() # Global config instance


@dataclass
class ModelMetrics:
    rmse: float
    mae: float
    r2: float

    def __str__(self) -> str:
        return f"RMSE: {self.rmse:.4f}, MAE: {self.mae:.4f}, R²: {self.r2:.4f}"
    
    def to_dict(self):
        return {"rmse": self.rmse, "mae": self.mae, "r2": self.r2}


class DataManager:
    def __init__(self, config: Config):
        self.config = config
        self.cache_dir = self.config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def download_stock_data(self, ticker: str, start_date: str, end_date: str,
                            use_cache: bool = True) -> pd.DataFrame:
        cache_file = os.path.join(self.cache_dir, f"{ticker}_{start_date}_{end_date}.pkl")

        if use_cache and os.path.exists(cache_file):
            logger.info(f"Carregando dados do cache: {cache_file}")
            try:
                df = pd.read_pickle(cache_file)
                if not isinstance(df, pd.DataFrame) or df.empty:
                    logger.warning(f"Cache '{cache_file}' inválido (não é DataFrame ou está vazio). Baixando novamente.")
                    os.remove(cache_file)
                    return self.download_stock_data(ticker, start_date, end_date, use_cache=False)

                last_cache_date = df.index.max().date()

                current_date_for_comparison = datetime.now().date()
                if current_date_for_comparison > last_cache_date + timedelta(days=1):
                     logger.warning("Dados do cache estão desatualizados, baixando novamente.")
                     os.remove(cache_file)
                     return self.download_stock_data(ticker, start_date, end_date, use_cache=False)

                logger.debug(f"Cache de dados carregado. Shape: {df.shape}, Colunas: {df.columns.tolist()}")

                if isinstance(df.columns, pd.MultiIndex):
                    logger.warning("Cache carregado com MultiIndex de colunas. Achatando...")
                    df.columns = df.columns.droplevel(1)
                    df = df.loc[:,~df.columns.duplicated()].copy()

                df.columns.name = None
                return df
            except Exception as e:
                logger.warning(f"Erro ao carregar cache '{cache_file}': {str(e)}. Baixando dados novamente.")
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                return self.download_stock_data(ticker, start_date, end_date, use_cache=False)

        try:
            logger.info(f"Baixando dados para {ticker} de {start_date} até {end_date}")
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

            if not isinstance(df, pd.DataFrame) or df.empty:
                raise ValueError(f"Não foi possível obter dados válidos para {ticker} no período especificado. DataFrame vazio ou inválido.")

            if isinstance(df.columns, pd.MultiIndex):
                logger.info("DataFrame baixado possui MultiIndex nas colunas. Achatando...")
                df.columns = df.columns.droplevel(1)

            df.columns.name = None

            logger.debug(f"DataFrame recém-baixado (APÓS achatar MultiIndex). Shape: {df.shape}, Colunas: {df.columns.tolist()}, Dtypes: \\n{df.dtypes}")
            logger.debug(f"Head do DataFrame baixado (APÓS achatar MultiIndex): \\n{df.head()}")

            if use_cache:
                df.to_pickle(cache_file)
                logger.info(f"Dados salvos no cache: {cache_file}")

            return df.copy()

        except Exception as e:
            logger.error(f"Erro ao baixar dados: {str(e)}")
            raise ValueError(f"Falha no download de dados para {ticker}: {str(e)}")

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Calculando indicadores técnicos...")
        df = df.copy()

        if 'Close' not in df.columns or 'High' not in df.columns or 'Low' not in df.columns or 'Volume' not in df.columns:
            logger.error("Colunas essenciais (Close, High, Low, Volume) não encontradas no DataFrame para cálculo de indicadores.")
            raise ValueError("Colunas de preço/volume ausentes, não é possível calcular indicadores.")

        for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        if df['Close'].isnull().all() or df['Close'].empty:
            logger.warning("Coluna 'Close' está toda NaN ou vazia após limpeza inicial. Indicadores podem falhar.")

        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
        macd_indicator = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
        df["MACD"] = macd_indicator.macd()
        bollinger_bands = BollingerBands(close=df["Close"], window=20, window_dev=2)
        df["BollingerBands_Upper"] = bollinger_bands.bollinger_hband()
        df["BollingerBands_Lower"] = bollinger_bands.bollinger_lband()
        df['Volatility'] = df['Close'].rolling(window=20).std()
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()

        price = df['Close']
        high = df['High']
        low = df['Low']
        close = df['Close']
        volume = df['Volume']

        df['sma_5'] = price.rolling(window=5).mean()
        df['sma_10'] = price.rolling(window=10).mean()
        df['ema_5'] = price.ewm(span=5).mean()
        df['ema_10'] = price.ewm(span=10).mean()
        df['momentum_5'] = price - price.shift(5)
        df['momentum_10'] = price - price.shift(10)
        df['roc_5'] = price.pct_change(5)
        df['roc_10'] = price.pct_change(10)
        df['std_5'] = price.rolling(window=5).std()
        df['std_10'] = price.rolling(window=10).std()

        # Added check for typical_price to avoid division by zero if volume is all zero or NaN.
        typical_price = (high + low + close) / 3
        if volume.sum() == 0: # Avoid division by zero if volume is all 0
            df['vwap'] = np.nan
        else:
            vwap = (typical_price * volume).cumsum() / volume.cumsum()
            df['vwap'] = vwap

        obv = pd.Series(0, index=close.index)
        price_change = close.diff()
        obv[price_change > 0] = volume[price_change > 0]
        obv[price_change < 0] = -volume[price_change < 0]
        df['obv'] = obv.cumsum().fillna(0)

        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        # Handle cases where atr might be zero or NaN
        if atr.isnull().all() or (atr == 0).all():
            plus_di = pd.Series(np.nan, index=df.index)
            minus_di = pd.Series(np.nan, index=df.index)
            dx = pd.Series(np.nan, index=df.index)
        else:
            plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
            minus_di = 100 * (-minus_dm.ewm(alpha=1/14).mean() / atr)
            # Handle division by zero for dx
            denominator_dx = (plus_di + minus_di)
            dx = 100 * abs(plus_di - minus_di) / (denominator_dx.replace(0, np.nan) + 1e-10) # Add epsilon and replace 0 with nan
        
        df['adx_14'] = dx.ewm(alpha=1/14).mean()

        df['atr_14'] = atr

        tp = (high + low + close) / 3
        # Handle cases where std might be zero or NaN
        cci_std = tp.rolling(20).std()
        if cci_std.isnull().all() or (cci_std == 0).all():
            cci = pd.Series(np.nan, index=df.index)
        else:
            cci = (tp - tp.rolling(20).mean()) / (0.015 * cci_std)
        df['cci_20'] = cci

        highest_high = high.rolling(14).max()
        lowest_low = low.rolling(14).min()
        # Handle division by zero for Williams %R
        denominator_williams = (highest_high - lowest_low)
        df['williams_r'] = -100 * (highest_high - close) / (denominator_williams.replace(0, np.nan) + 1e-10)

        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        # Handle division by zero for Stochastic Oscillator %K
        denominator_stochastic = (high14 - low14)
        df['stochastic_k'] = 100 * (close - low14) / (denominator_stochastic.replace(0, np.nan) + 1e-10)

        logger.info("Indicadores técnicos calculados")
        return df

    def validate_data(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        logger.info("Iniciando validação de dados...")

        if df.empty:
            raise ValueError("DataFrame está vazio após o download ou processamento.")

        missing_columns_before_fill = [col for col in required_columns if col not in df.columns]
        if missing_columns_before_fill:
            logger.error(f"COLUNAS CRÍTICAS FALTANDO no DataFrame antes do tratamento de NaN: {missing_columns_before_fill}")
            raise ValueError(f"Colunas requeridas faltando no DataFrame: {missing_columns_before_fill}")

        for col in required_columns:
            current_col_series = df[col]
            col_data = current_col_series.values

            if np.isnan(col_data).any():
                initial_nans = np.isnan(col_data).sum()
                logger.warning(f"Coluna '{col}' tem {initial_nans} NaNs iniciais ANTES do preenchimento coletivo.")

            if not np.isfinite(col_data).all():
                logger.warning(f"Coluna '{col}' contém valores infinitos ou muito grandes. Eles serão convertidos para NaN e tratados.")
                df[col].replace([np.inf, -np.inf], np.nan, inplace=True)

        initial_df_len = len(df)
        df[required_columns] = df[required_columns].fillna(method='ffill')
        df[required_columns] = df[required_columns].fillna(method='bfill')

        original_rows_before_final_dropna = len(df)
        df.dropna(subset=required_columns, inplace=True)
        rows_dropped_final = original_rows_before_final_dropna - len(df)
        if rows_dropped_final > 0:
            logger.warning(f"Removidas {rows_dropped_final} linhas devido a NaNs remanescentes nas features requeridas APÓS preenchimento.")

        if df.empty:
            raise ValueError(f"DataFrame ficou vazio após limpeza de NaNs nas features requeridas.")

        if len(df) < self.config.LOOKBACK + self.config.DIAS_PREVISAO:
            raise ValueError(f"Dados insuficientes após limpeza: {len(df)} registros. "
                             f"Necessário pelo menos {self.config.LOOKBACK + self.config.DIAS_PREVISAO}")

        for col in required_columns:
            if np.isnan(df[col].values).any():
                logger.warning(f"Atenção: A coluna '{col}' ainda contém valores NaN após a validação final. Isso pode afetar o escalonamento e o treinamento.")

        logger.info("Validação de dados concluída.")
        return True


class LSTMModel:
    def __init__(self, input_shape: Tuple[int, int], config: Config):
        self.config = config
        self.input_shape = input_shape
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False

    def build_model(self) -> None:
        if not LSTM_AVAILABLE:
            raise RuntimeError("TensorFlow/Keras não está disponível. Não é possível construir o modelo LSTM.")
        logger.info("Construindo modelo LSTM...")

        self.model = Sequential()

        self.model.add(LSTM(
            self.config.LSTM_UNITS[0],
            return_sequences=len(self.config.LSTM_UNITS) > 1,
            input_shape=self.input_shape
        ))
        self.model.add(Dropout(self.config.DROPOUT_RATE))

        for i, units in enumerate(self.config.LSTM_UNITS[1:], 1):
            return_sequences = i < len(self.config.LSTM_UNITS) - 1
            self.model.add(LSTM(units, return_sequences=return_sequences))
            self.model.add(Dropout(self.config.DROPOUT_RATE))

        for units in self.config.DENSE_UNITS:
            self.model.add(Dense(units, activation='relu'))
            self.model.add(Dropout(self.config.DROPOUT_RATE))

        self.model.add(Dense(1))

        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        logger.info(f"Modelo construído com {self.model.count_params()} parâmetros")
        # self.model.summary(print_fn=logger.info) # Removed for web app output

    def prepare_data(self, df: pd.DataFrame, features: List[str],
                     lookback: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logger.info("Preparando dados para LSTM...")

        data_for_scaling = df[features].values
        scaled_data = self.scaler.fit_transform(data_for_scaling)

        X, y = [], []
        close_idx = features.index("Close")
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, close_idx])

        X, y = np.array(X), np.array(y)

        train_size = int(len(X) * self.config.TRAIN_SPLIT_RATIO)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        logger.info(f"Dados preparados - Treino: {X_train.shape}, Teste: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        if not LSTM_AVAILABLE:
            raise RuntimeError("TensorFlow/Keras não está disponível. Não é possível treinar o modelo LSTM.")
        if self.model is None:
            raise ValueError("Modelo não foi construído. Chame build_model() primeiro.")

        logger.info("Iniciando treinamento do modelo...")

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=0 # Changed to 0 for web app
            ),
            ModelCheckpoint(
                self.config.MODEL_PATH,
                monitor='val_loss',
                save_best_only=True,
                verbose=0 # Changed to 0 for web app
            )
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=0 # Changed to 0 for web app
        )

        self.is_trained = True
        logger.info("Treinamento concluído")

        return history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not LSTM_AVAILABLE:
            raise RuntimeError("TensorFlow/Keras não está disponível. Não é possível fazer previsões com o modelo LSTM.")
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ou carregado.")

        return self.model.predict(X, verbose=0)

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> ModelMetrics:
        if not LSTM_AVAILABLE:
            return ModelMetrics(rmse=np.nan, mae=np.nan, r2=np.nan)
        y_pred = self.predict(X_test).flatten()

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return ModelMetrics(rmse=rmse, mae=mae, r2=r2)

    def save_model(self, filepath: str) -> None:
        if not LSTM_AVAILABLE:
            logger.warning("TensorFlow/Keras não está disponível. Não é possível salvar o modelo LSTM.")
            return
        if self.model is None:
            raise ValueError("Modelo não foi construído ou treinado.")

        self.model.save(filepath)

        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        logger.info(f"Modelo e scaler salvos em {filepath} e {scaler_path}")

    def load_model(self, filepath: str) -> None:
        if not LSTM_AVAILABLE:
            raise RuntimeError("TensorFlow/Keras não está disponível. Não é possível carregar o modelo LSTM.")
        try:
            self.model = load_model(filepath, custom_objects={
                'mse': tf.keras.losses.MeanSquaredError(),
                'mae': tf.keras.metrics.MeanAbsoluteError()
            })
            scaler_path = filepath.replace('.h5', '_scaler.pkl')
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

            self.is_trained = True
            logger.info(f"Modelo e scaler carregados de {filepath}")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo ou scaler de {filepath}: {str(e)}")
            raise


class StockPredictor:
    def __init__(self, config: Config):
        self.config = config
        self.data_manager = DataManager(self.config)
        self.model: Optional[LSTMModel] = None
        self.df: Optional[pd.DataFrame] = None

    def predict_stock_price(self, ticker: str,
                            dias_previsao: int = None,
                            lookback: int = None,
                            use_cache: bool = True,
                            retrain: bool = False) -> Dict[str, Any]:
        if not LSTM_AVAILABLE:
            return {
                'error': "TensorFlow/Keras não está disponível no ambiente do backend. Previsão LSTM desativada.",
                'ticker': ticker,
                'predictions': [],
                'dates': [],
                'metrics': {"rmse": None, "mae": None, "r2": None},
                'current_price': None,
                'data_shape': None
            }

        dias_previsao = dias_previsao if dias_previsao is not None else self.config.DIAS_PREVISAO
        lookback = lookback if lookback is not None else self.config.LOOKBACK

        try:
            today = datetime.now()
            end_date = today.strftime('%Y-%m-%d')
            start_date = (today - timedelta(days=self.config.START_DATE_OFFSET_YEARS * 365)).strftime('%Y-%m-%d')

            # Use self.df if already populated, otherwise download
            if self.df is None:
                self.df = self.data_manager.download_stock_data(
                    ticker, start_date, end_date, use_cache
                )
                self.df = self.data_manager.calculate_technical_indicators(self.df)


            self.data_manager.validate_data(self.df, self.config.FEATURES_LSTM)

            logger.debug(f"DataFrame FINAL antes de preparar dados para o modelo. Shape: {self.df.shape}, Colunas: {self.df.columns.tolist()}")
            if self.df.empty:
                raise ValueError("DataFrame está vazio após o tratamento final de NaNs para as features.")

            missing_features_final = [f for f in self.config.FEATURES_LSTM if f not in self.df.columns]
            if missing_features_final:
                logger.error(f"ERRO CRÍTICO: As seguintes features ainda estão faltando após validação: {missing_features_final}")
                raise KeyError(f"Features obrigatórias não encontradas: {missing_features_final}")


            self.model = LSTMModel(input_shape=(lookback, len(self.config.FEATURES_LSTM)), config=self.config)

            X_train, X_test, y_train, y_test = self.model.prepare_data(
                self.df, self.config.FEATURES_LSTM, lookback
            )

            model_exists = os.path.exists(self.config.MODEL_PATH) and os.path.exists(self.config.MODEL_PATH.replace('.h5', '_scaler.pkl'))

            force_retrain_due_to_mismatch = False
            if model_exists:
                temp_model = None
                try:
                    temp_model = load_model(self.config.MODEL_PATH)
                    loaded_input_features = temp_model.input_shape[2]
                    current_expected_features = len(self.config.FEATURES_LSTM)

                    if loaded_input_features != current_expected_features:
                        logger.warning(f"O modelo carregado espera {loaded_input_features} features, mas a configuração atual tem {current_expected_features}. Forçando retreinamento.")
                        force_retrain_due_to_mismatch = True
                except Exception as e:
                    logger.warning(f"Não foi possível inspecionar o modelo existente em {self.config.MODEL_PATH} ({e}). Forçando retreinamento para garantir compatibilidade.")
                    force_retrain_due_to_mismatch = True
                finally:
                    if temp_model:
                        del temp_model
                        tf.keras.backend.clear_session()


            if retrain or not model_exists or force_retrain_due_to_mismatch:
                if retrain:
                    logger.info("Opção 'retrain' ativada. Retreinando o modelo.")
                elif not model_exists:
                    logger.info("Modelo não encontrado ou scaler ausente. Treinando novo modelo.")
                elif force_retrain_due_to_mismatch:
                    logger.info("Forçando retreinamento devido a incompatibilidade de features do modelo.")

                self.model.build_model()

                val_split = int(len(X_train) * self.config.TRAIN_SPLIT_RATIO)
                X_val = X_train[val_split:]
                y_val = y_train[val_split:]
                X_train_final = X_train[:val_split]
                y_train_final = y_train[:val_split]

                history = self.model.train(X_train_final, y_train_final, X_val, y_val)
                self.model.save_model(self.config.MODEL_PATH)
            else:
                logger.info("Carregando modelo e scaler existentes.")
                self.model.load_model(self.config.MODEL_PATH)

            metrics = self.model.evaluate_model(X_test, y_test)
            logger.info(f"Métricas do modelo: {metrics}")

            predictions, actual_last_price = self._predict_future_prices(dias_previsao, lookback)

            future_dates = [
                (today + timedelta(days=i)).strftime('%Y-%m-%d')
                for i in range(1, dias_previsao + 1)
            ]

            results = {
                'ticker': ticker,
                'predictions': [float(p) for p in predictions], # Ensure serializable
                'dates': future_dates,
                'metrics': metrics.to_dict(), # Convert ModelMetrics to dict
                'current_price': float(actual_last_price), # Ensure serializable
                'data_shape': self.df.shape
            }

            return results

        except Exception as e:
            logger.error(f"Erro fatal na previsão: {str(e)}", exc_info=True)
            raise

    def _predict_future_prices(self, dias_previsao: int, lookback: int) -> Tuple[List[float], float]:
        if not LSTM_AVAILABLE:
            raise RuntimeError("TensorFlow/Keras não está disponível. Não é possível fazer previsões futuras com o modelo LSTM.")
        if self.df is None or self.model is None or not self.model.is_trained:
            raise ValueError("DataFrame ou modelo não estão prontos para previsão futura.")

        if len(self.df) < lookback:
            raise ValueError(f"Dados insuficientes para previsão futura. Necessário pelo menos {lookback} dias, mas tem {len(self.df)}.")

        missing_features_for_predict = [f for f in self.config.FEATURES_LSTM if f not in self.df.columns]
        if missing_features_for_predict:
            raise ValueError(f"Features faltando para previsão: {missing_features_for_predict}")

        last_data_point_scaled = self.model.scaler.transform(self.df[self.config.FEATURES_LSTM].tail(lookback).values)

        actual_last_price = self.df['Close'].iloc[-1]

        predictions = []
        current_batch = last_data_point_scaled

        for i in range(dias_previsao):
            predicted_scaled_price = self.model.predict(current_batch[np.newaxis, :, :])[0, 0]

            dummy_row_for_inverse = np.zeros((1, len(self.config.FEATURES_LSTM)))
            close_idx = self.config.FEATURES_LSTM.index("Close")
            dummy_row_for_inverse[0, close_idx] = predicted_scaled_price

            predicted_price = self.model.scaler.inverse_transform(dummy_row_for_inverse)[0, close_idx]

            predictions.append(predicted_price)

            next_day_features_scaled = current_batch[-1].copy()
            next_day_features_scaled[close_idx] = predicted_scaled_price

            current_batch = np.vstack([current_batch[1:], next_day_features_scaled[np.newaxis, :]])

        return predictions, actual_last_price


class Visualizer:
    def __init__(self, config: Config):
        self.config = config

    def plot_predictions(self, df: pd.DataFrame, predictions: List[float],
                         future_dates: List[str], ticker: str,
                         save_path: Optional[str] = None) -> None:
        pass

    @staticmethod
    def plot_training_history(history: Dict[str, List[float]]) -> None:
        pass


# --- Funções da análise de Bias-Variance ---

def compute_target(df: pd.DataFrame) -> pd.Series:
    return df['Close'].pct_change(periods=5).shift(-5)

def walk_forward_with_pca_vif(data, model, window_size, step_size, n_components):
    predictions = []
    prediction_indices = []
    last_scaler = None
    last_pca = None
    last_vif_scores = None
    last_trained_model = None
    last_selected_pc_indices = None

    for i in range(window_size, len(data) - 5, step_size):
        train_data = data.iloc[i - window_size:i]
        test_row = data.iloc[i]

        X_raw_train = train_data.drop(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Target'], errors='ignore')
        y_train = train_data['Target']

        if X_raw_train.empty or y_train.empty:
            continue

        scaler = StandardScaler()
        X_scaled_train = scaler.fit_transform(X_raw_train)

        pca = PCA(n_components=n_components)
        X_pca_train = pca.fit_transform(X_scaled_train)

        vif_data_train = pd.DataFrame(X_pca_train, columns=[f'PC{j+1}' for j in range(n_components)])
        if vif_data_train.shape[1] > 1:
            vif_scores = [variance_inflation_factor(vif_data_train.values, j) for j in range(vif_data_train.shape[1])]
            selected_pc_indices = np.array(vif_scores) < 3
            selected_pcs_train = vif_data_train.iloc[:, selected_pc_indices]
        else:
            selected_pc_indices = np.array([True] * vif_data_train.shape[1])
            selected_pcs_train = vif_data_train

        test_raw = test_row.drop(labels=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Target'], errors='ignore').values.reshape(1, -1)
        test_scaled = scaler.transform(test_raw)
        test_pca = pca.transform(test_scaled)

        test_vif_df = pd.DataFrame(test_pca, columns=[f'PC{j+1}' for j in range(n_components)])
        test_selected = test_vif_df.iloc[:, selected_pc_indices]

        if selected_pcs_train.empty or test_selected.empty:
            continue

        model_clone = clone(model)
        model_clone.fit(selected_pcs_train, y_train)
        prediction = model_clone.predict(test_selected)[0]
        predictions.append(prediction)
        prediction_indices.append(i)

        last_scaler = scaler
        last_pca = pca
        last_vif_scores = vif_scores if 'vif_scores' in locals() else None
        last_trained_model = model_clone
        last_selected_pc_indices = selected_pc_indices

    pred_series = pd.Series(data=np.nan, index=data.index)
    if prediction_indices:
        pred_series.iloc[prediction_indices] = predictions

    return pred_series, last_scaler, last_pca, last_vif_scores, last_trained_model, last_selected_pc_indices


def backtest_strategy(final_data, initial_capital, capital_deployed):
    trade_log = []
    for i in range(len(final_data) - 5):
        prediction = final_data.iloc[i]['Predictions']
        if pd.isna(prediction) or prediction == 0:
            continue

        direction = 'Long' if prediction > 0 else 'Short'
        entry_price = final_data.iloc[i + 1]['Open']
        exit_price = final_data.iloc[i + 5]['Close']

        if direction == 'Long':
            trade_return = (exit_price - entry_price) / entry_price
        else:
            trade_return = (entry_price - exit_price) / entry_price

        capital_change = capital_deployed * initial_capital * trade_return

        trade_log.append({
            'Signal_Day': final_data.iloc[i]['Date'].strftime('%Y-%m-%d'),
            'Entry_Day': final_data.iloc[i + 1]['Date'].strftime('%Y-%m-%d'),
            'Exit_Day': final_data.iloc[i + 5]['Date'].strftime('%Y-%m-%d'),
            'Direction': direction,
            'Entry_Price': float(entry_price),
            'Exit_Price': float(exit_price),
            'Return': float(trade_return),
            'Capital_Change': float(capital_change)
        })
    return pd.DataFrame(trade_log)

def sharpe_ratio(equity_curve, risk_free_rate=0.0):
    daily_returns = equity_curve.pct_change().dropna()
    if daily_returns.std() == 0:
        return 0.0
    excess_returns = daily_returns - (risk_free_rate / 252)
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def max_drawdown(equity_curve):
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min(), drawdown

def get_models():
    return {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'DecisionTree': DecisionTreeRegressor(),
        'Bagging': BaggingRegressor(n_estimators=100, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

def evaluate_bias_variance_all(models, X, y, test_size=0.2, num_rounds=100):
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    results = {}
    for name, model in models.items():
        if len(y_train) == 0 or len(y_test) == 0:
            results[name] = {'Total Error': np.nan, 'Bias': np.nan, 'Variance': np.nan, 'Irreducible Error': np.nan}
            continue

        try:
            avg_loss, bias, var = bias_variance_decomp(
                model,
                X_train.values, y_train.values,
                X_test.values, y_test.values,
                loss='mse',
                num_rounds=num_rounds,
                random_seed=42
            )
            results[name] = {
                'Total Error': float(avg_loss),
                'Bias': float(bias),
                'Variance': float(var),
                'Irreducible Error': float(avg_loss - bias - var)
            }
        except Exception as e:
            results[name] = {'Total Error': np.nan, 'Bias': np.nan, 'Variance': np.nan, 'Irreducible Error': np.nan}
    return pd.DataFrame(results).T

def find_integration_order(series):
    d = 0
    current_series = series.copy()
    while True:
        if len(current_series.dropna()) < 20:
            return d
        result = adfuller(current_series.dropna(), autolag='AIC')
        if result[1] <= 0.05:
            return d
        current_series = current_series.diff().dropna()
        d += 1
        if d >= 2:
            return d

def run_bias_variance_analysis(df: pd.DataFrame, config: Config):
    results_bv = {}

    data_for_bv = df.copy()
    data_for_bv['Target'] = compute_target(data_for_bv)
    data_for_bv.dropna(inplace=True)

    all_indicator_columns = [col for col in data_for_bv.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Target']]
    X_bv = data_for_bv[all_indicator_columns]
    y_bv = data_for_bv['Target']

    X_bv = X_bv.dropna()
    y_bv = y_bv.loc[X_bv.index]

    models = get_models()
    if not X_bv.empty and not y_bv.empty:
        bv_results = evaluate_bias_variance_all(models, X_bv, y_bv, num_rounds=config.NUM_ROUNDS_BIAS_VARIANCE)
        results_bv['bias_variance_decomposition'] = bv_results.to_dict('index')
    else:
        results_bv['bias_variance_decomposition'] = "Dados insuficientes para a decomposição Bias-Variância (sem estacionarização)."

    integration_orders = {}
    for col in all_indicator_columns:
        if col in df.columns:
            integration_orders[col] = find_integration_order(df[col].copy())

    integration_orders_df = pd.DataFrame.from_dict(integration_orders, orient='index', columns=['Integration Order'])
    integration_orders_df.index.name = 'Indicator'
    results_bv['integration_orders'] = integration_orders_df.to_dict('index')

    differenced_indicators = df[all_indicator_columns].copy()
    for indicator in integration_orders_df.index:
        order = integration_orders_df.loc[indicator, 'Integration Order']
        if order > 0:
            differenced_indicators[indicator] = differenced_indicators[indicator].diff(order)

    differenced_indicators.dropna(inplace=True)

    data_merged_differenced = df.iloc[-len(differenced_indicators):].reset_index(drop=True)
    data_merged_differenced = pd.concat([data_merged_differenced, differenced_indicators.reset_index(drop=True)], axis=1)
    data_merged_differenced['Target'] = compute_target(data_merged_differenced)
    data_merged_differenced.dropna(inplace=True)

    model_bv = GradientBoostingRegressor(n_estimators=100, random_state=42)
    pred_series_differenced, last_scaler, last_pca, last_vif_scores, last_trained_model, last_selected_pc_indices = \
        walk_forward_with_pca_vif(
            data_merged_differenced, model_bv,
            window_size=config.WINDOW_SIZE_BIAS_VARIANCE,
            step_size=config.STEP_SIZE_BIAS_VARIANCE,
            n_components=config.N_COMPONENTS_PCA
        )

    data_merged_differenced['Predictions'] = pred_series_differenced
    data_merged_differenced.dropna(subset=['Predictions'], inplace=True)

    final_data_differenced = data_merged_differenced[['Date', 'Open', 'Close', 'Target', 'Predictions']].copy()
    final_data_differenced['Date'] = pd.to_datetime(final_data_differenced['Date'])
    trades_df_differenced = backtest_strategy(
        final_data_differenced,
        initial_capital=config.INITIAL_CAPITAL_BACKTEST,
        capital_deployed=config.CAPITAL_DEPLOYED_PER_TRADE
    )
    results_bv['trade_log'] = trades_df_differenced.to_dict('records')

    initial_capital = config.INITIAL_CAPITAL_BACKTEST
    if not trades_df_differenced.empty:
        trades_df_differenced['Cumulative_Capital_Strategy'] = initial_capital + trades_df_differenced['Capital_Change'].cumsum()
        results_bv['cumulative_capital_strategy'] = trades_df_differenced[['Exit_Day', 'Cumulative_Capital_Strategy']].to_dict('records')

    first_trade_date = trades_df_differenced['Entry_Day'].min() if not trades_df_differenced.empty else None
    last_trade_date = trades_df_differenced['Exit_Day'].max() if not trades_df_differenced.empty else None

    bh_data = pd.DataFrame()
    if first_trade_date and last_trade_date:
        bh_data = df[(df['Date'] >= first_trade_date) &
                     (df['Date'] <= last_trade_date)].copy()

    if not bh_data.empty:
        bh_initial_price = bh_data.iloc[0]['Close']
        bh_data['Cumulative_Capital_BH'] = initial_capital * (bh_data['Close'] / bh_initial_price)
        results_bv['cumulative_capital_bh'] = bh_data[['Date', 'Cumulative_Capital_BH']].to_dict('records')
    else:
        results_bv['cumulative_capital_bh'] = "Não há dados suficientes para calcular o Buy and Hold no período dos trades."

    # Métricas da Estratégia
    if not trades_df_differenced.empty:
        final_capital_strategy = trades_df_differenced['Cumulative_Capital_Strategy'].iloc[-1]
        total_return_strategy = (final_capital_strategy - initial_capital) / initial_capital

        trading_days = (datetime.strptime(trades_df_differenced['Exit_Day'].iloc[-1], '%Y-%m-%d') - datetime.strptime(trades_df_differenced['Entry_Day'].iloc[0], '%Y-%m-%d')).days
        num_years = trading_days / 365.25
        if num_years > 0:
            cagr_strategy = ((final_capital_strategy / initial_capital) ** (1 / num_years)) - 1
        else:
            cagr_strategy = 0

        sharpe_strategy = sharpe_ratio(trades_df_differenced['Cumulative_Capital_Strategy'])
        max_dd_strategy, _ = max_drawdown(trades_df_differenced['Cumulative_Capital_Strategy'])

        profitable_trades = trades_df_differenced[trades_df_differenced['Return'] > 0]
        if len(trades_df_differenced) > 0:
            hit_ratio_strategy = len(profitable_trades) / len(trades_df_differenced)
        else:
            hit_ratio_strategy = 0

        avg_return_per_trade_strategy = trades_df_differenced['Return'].mean()

        results_bv['strategy_metrics'] = {
            "Capital Final": float(final_capital_strategy),
            "Retorno Total": float(total_return_strategy),
            "CAGR": float(cagr_strategy),
            "Sharpe Ratio": float(sharpe_strategy),
            "Max Drawdown": float(max_dd_strategy),
            "Hit Ratio": float(hit_ratio_strategy),
            "Retorno Médio por Trade": float(avg_return_per_trade_strategy)
        }
    else:
        results_bv['strategy_metrics'] = "Não há trades para a Estratégia de Trading para calcular as métricas."

    # Métricas de Buy and Hold
    if not bh_data.empty:
        final_capital_bh = bh_data['Cumulative_Capital_BH'].iloc[-1]
        total_return_bh = (final_capital_bh - initial_capital) / initial_capital

        if num_years > 0:
            cagr_bh = ((final_capital_bh / initial_capital) ** (1 / num_years)) - 1
        else:
            cagr_bh = 0

        sharpe_bh = sharpe_ratio(bh_data['Cumulative_Capital_BH'])
        max_dd_bh, _ = max_drawdown(bh_data['Cumulative_Capital_BH'])

        results_bv['buy_and_hold_metrics'] = {
            "Capital Final": float(final_capital_bh),
            "Retorno Total": float(total_return_bh),
            "CAGR": float(cagr_bh),
            "Sharpe Ratio": float(sharpe_bh),
            "Max Drawdown": float(max_dd_bh)
        }
    else:
        results_bv['buy_and_hold_metrics'] = "Não há dados para o Buy and Hold para calcular as métricas."

    # Previsão Futura para Bias-Variance
    if last_trained_model is not None and last_scaler is not None and last_pca is not None and last_selected_pc_indices is not None:
        if len(df) >= 30:
            temp_df_for_latest_indicators = df.iloc[-30:].copy()
        else:
            temp_df_for_latest_indicators = df.copy()

        latest_indicators_full_df = DataManager(config).calculate_technical_indicators(temp_df_for_latest_indicators)

        if not latest_indicators_full_df.empty:
            latest_full_indicators_row = latest_indicators_full_df.iloc[-1]

            latest_differenced_indicators_for_prediction = latest_full_indicators_row.copy()
            for indicator_name, order in integration_orders.items():
                if order > 0 and indicator_name in latest_full_indicators_row.index:
                    if indicator_name in df.columns:
                        temp_series = df[indicator_name]
                        for d_order in range(order):
                            temp_series = temp_series.diff().dropna()

                        if not temp_series.empty:
                            latest_differenced_indicators_for_prediction[indicator_name] = temp_series.iloc[-1]
                        else:
                            latest_differenced_indicators_for_prediction.drop(indicator_name, inplace=True, errors='ignore')
                    else:
                        latest_differenced_indicators_for_prediction.drop(indicator_name, inplace=True, errors='ignore')
                elif indicator_name in latest_full_indicators_row.index:
                    latest_differenced_indicators_for_prediction[indicator_name] = latest_full_indicators_row[indicator_name]
                else:
                    latest_differenced_indicators_for_prediction.drop(indicator_name, inplace=True, errors='ignore')

            X_latest_for_prediction = pd.DataFrame([latest_differenced_indicators_for_prediction]).dropna(axis=1)

            training_feature_columns = data_merged_differenced.drop(
                columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Predictions'],
                errors='ignore'
            ).columns

            missing_cols = set(training_feature_columns) - set(X_latest_for_prediction.columns)
            for c in missing_cols:
                X_latest_for_prediction[c] = 0

            extra_cols = set(X_latest_for_prediction.columns) - set(training_feature_columns)
            X_latest_for_prediction = X_latest_for_prediction.drop(columns=list(extra_cols))

            X_latest_for_prediction = X_latest_for_prediction[training_feature_columns]

            X_latest_raw_for_transform = X_latest_for_prediction.values.reshape(1, -1)

            X_latest_scaled = last_scaler.transform(X_latest_raw_for_transform)
            X_latest_pca = last_pca.transform(X_latest_scaled)

            X_latest_selected_pcs = pd.DataFrame(X_latest_pca, columns=[f'PC{j+1}' for j in range(last_pca.n_components_)])
            X_latest_selected_for_prediction = X_latest_selected_pcs.iloc[:, last_selected_pc_indices]

            future_prediction = last_trained_model.predict(X_latest_selected_for_prediction)[0]

            latest_date_available = df['Date'].iloc[-1].strftime('%Y-%m-%d')
            results_bv['future_prediction'] = {
                "date_available": latest_date_available,
                "prediction_value": float(future_prediction),
                "direction": "ALTA" if future_prediction > 0 else ("BAIXA" if future_prediction < 0 else "NEUTRO")
            }
        else:
            results_bv['future_prediction'] = "Não foi possível gerar indicadores para a previsão futura (dados insuficientes ou erro na criação)."
    else:
        results_bv['future_prediction'] = "Não foi possível gerar a previsão futura. Verifique se o pipeline de dados foi executado corretamente e gerou modelos/transformadores válidos."

    return results_bv


def run_lstm_prediction(df: pd.DataFrame, config: Config):
    results_lstm = {}
    predictor = StockPredictor(config)

    try:
        predictor.df = df.copy()
        results = predictor.predict_stock_price(
            ticker=config.TICKER,
            dias_previsao=config.DIAS_PREVISAO,
            lookback=config.LOOKBACK,
            retrain=False
        )
        results_lstm = results
    except Exception as e:
        results_lstm['error'] = f"Erro fatal na execução principal da previsão LSTM: {str(e)}"
        logger.error(results_lstm['error'], exc_info=True)
    return results_lstm


@app.route('/analyze_stock', methods=['POST'])
def analyze_stock():
    data = request.get_json()
    ticker_input = data.get('ticker', 'b3sa3.SA')
    period_input = data.get('period', '1y')

    app_config.TICKER = ticker_input
    app_config.PERIOD = period_input

    today = datetime.now()
    if period_input.endswith('y'):
        years = int(period_input[:-1])
        start_date_offset = years * 365
    elif period_input.endswith('mo'):
        months = int(period_input[:-2])
        start_date_offset = months * 30
    else:
        start_date_offset = app_config.START_DATE_OFFSET_YEARS * 365

    app_config.START_DATE_OFFSET_YEARS = start_date_offset / 365

    data_manager = DataManager(app_config)
    end_date = today.strftime('%Y-%m-%d')
    start_date = (today - timedelta(days=start_date_offset)).strftime('%Y-%m-%d')

    full_results = {}

    try:
        raw_df = data_manager.download_stock_data(app_config.TICKER, start_date, end_date, use_cache=True)
        
        if raw_df.empty:
            raise ValueError(f"Não foi possível baixar dados para {app_config.TICKER} no período {app_config.PERIOD}. DataFrame vazio.")

        processed_df = data_manager.calculate_technical_indicators(raw_df.copy())
        
        if processed_df.empty:
            raise ValueError(f"DataFrame vazio após o cálculo de indicadores para {app_config.TICKER}.")

        processed_df['Date'] = processed_df.index
        processed_df.reset_index(drop=True, inplace=True)

        all_indicator_columns_for_validation = [col for col in processed_df.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        data_manager.validate_data(processed_df, all_indicator_columns_for_validation)
        data_manager.validate_data(processed_df, app_config.FEATURES_LSTM)

        # Execute both analyses
        lstm_output = run_lstm_prediction(processed_df.copy(), app_config)
        bias_variance_output = run_bias_variance_analysis(processed_df.copy(), app_config)

        full_results['lstm_results'] = lstm_output
        full_results['bias_variance_results'] = bias_variance_output

        return jsonify(full_results)

    except Exception as e:
        error_message = f"Erro ao preparar os dados ou executar a análise: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message, exc_info=True)
        return jsonify({"error": error_message}), 500

if __name__ == '__main__':
    # Para execução local, use: python app.py
    # Para implantação em nuvem, o provedor de serviços pode gerenciar a execução.
    # Exemplo para Flask local:
    # app.run(debug=True, host='0.0.0.0', port=5000)
    # Use um servidor de produção como Gunicorn para produção: gunicorn -w 4 app:app
    print("Iniciando Flask app. Para testar localmente, acesse http://127.0.0.1:5000/analyze_stock com um POST request.")
    print("Certifique-se de ter as dependências instaladas: pip install -r requirements.txt")
    print("Exemplo de POST request (usando curl):")
    print("curl -X POST -H \"Content-Type: application/json\" -d '{\"ticker\": \"b3sa3.SA\", \"period\": \"1y\"}' http://127.0.0.1:5000/analyze_stock")
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
