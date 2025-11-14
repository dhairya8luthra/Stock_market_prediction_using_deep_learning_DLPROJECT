"""
Data Preparation and Feature Engineering Module
Improved preprocessing pipeline for stock market data
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import ta


class StockDataPreprocessor:
    """
    Advanced stock data preprocessing with multiple improvements
    """
    
    def __init__(self, window_size=60, prediction_horizon=1, scaler_type='standard'):
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.scaler_type = scaler_type
        self.scalers = {}
        self.feature_names = []
        
    def add_technical_indicators(self, df):
        """
        Add comprehensive technical indicators
        
        IMPROVEMENT: More indicators than baseline
        """
        df = df.copy()
        
        # Basic returns
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving Averages
        for window in [5, 10, 20, 50]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
            df[f'Volume_MA_{window}'] = df['Volume'].rolling(window=window).mean()
        
        # Momentum Indicators
        df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['RSI_21'] = ta.momentum.RSIIndicator(df['Close'], window=21).rsi()
        
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        df['ROC'] = ta.momentum.ROCIndicator(df['Close']).roc()
        df['Williams_R'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
        
        # Trend Indicators
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()
        
        df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
        df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
        
        # Volatility Indicators
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_High'] = bollinger.bollinger_hband()
        df['BB_Low'] = bollinger.bollinger_lband()
        df['BB_Mid'] = bollinger.bollinger_mavg()
        df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['BB_Mid']
        df['BB_Pct'] = (df['Close'] - df['BB_Low']) / (df['BB_High'] - df['BB_Low'])
        
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        kc = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'])
        df['KC_High'] = kc.keltner_channel_hband()
        df['KC_Low'] = kc.keltner_channel_lband()
        
        # Volume Indicators
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume']).chaikin_money_flow()
        df['MFI'] = ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume']).money_flow_index()
        
        # Price-based features
        df['HL_Range'] = (df['High'] - df['Low']) / df['Close']
        df['OC_Range'] = abs(df['Open'] - df['Close']) / df['Close']
        df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Volatility (Rolling Std)
        for window in [5, 10, 20]:
            df[f'Volatility_{window}'] = df['Returns'].rolling(window=window).std()
            df[f'Volume_Std_{window}'] = df['Volume'].rolling(window=window).std()
        
        # Volume ratios
        df['Volume_Ratio_5'] = df['Volume'] / df['Volume_MA_5']
        df['Volume_Ratio_20'] = df['Volume'] / df['Volume_MA_20']
        
        # Lag features (previous day values)
        for lag in [1, 2, 3, 5]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
        
        return df
    
    def create_sequences(self, df, stock_name=None):
        """
        Create sequences for time series prediction
        
        IMPROVEMENT: More robust sequence creation with proper scaling
        """
        # Add technical indicators
        df = self.add_technical_indicators(df)
        
        # Drop NaN values
        df = df.dropna().reset_index(drop=True)
        
        if len(df) < self.window_size + self.prediction_horizon:
            raise ValueError(f"Insufficient data: {len(df)} rows")
        
        # Select features (exclude date and stock columns)
        feature_columns = [col for col in df.columns 
                          if col not in ['Date', 'Stock', 'Symbol']]
        
        self.feature_names = feature_columns
        
        # Get target (Close price)
        target_col = 'Close'
        
        # Scale features
        scaler_name = stock_name if stock_name else 'default'
        if scaler_name not in self.scalers:
            if self.scaler_type == 'standard':
                self.scalers[scaler_name] = StandardScaler()
            elif self.scaler_type == 'minmax':
                self.scalers[scaler_name] = MinMaxScaler()
            elif self.scaler_type == 'robust':
                self.scalers[scaler_name] = RobustScaler()
        
        # Fit and transform
        scaled_data = self.scalers[scaler_name].fit_transform(df[feature_columns])
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.window_size - self.prediction_horizon + 1):
            X.append(scaled_data[i:i + self.window_size])
            
            # Target: percentage change from last value in window
            current_price = df.iloc[i + self.window_size - 1][target_col]
            future_price = df.iloc[i + self.window_size + self.prediction_horizon - 1][target_col]
            y.append((future_price - current_price) / current_price)
        
        return np.array(X), np.array(y)
    
    def create_multi_stock_dataset(self, stock_data_dict):
        """
        Create dataset from multiple stocks
        
        IMPROVEMENT: Multi-stock learning for better generalization
        """
        all_X, all_y = [], []
        stock_indices = []
        
        for stock_name, df in stock_data_dict.items():
            try:
                X, y = self.create_sequences(df, stock_name)
                all_X.append(X)
                all_y.append(y)
                stock_indices.extend([stock_name] * len(X))
                print(f"{stock_name}: {len(X)} sequences created")
            except ValueError as e:
                print(f"Skipping {stock_name}: {e}")
        
        if not all_X:
            raise ValueError("No valid sequences created")
        
        X_combined = np.vstack(all_X)
        y_combined = np.hstack(all_y)
        
        return X_combined, y_combined, stock_indices
    
    def inverse_transform_predictions(self, predictions, stock_name='default'):
        """
        Convert normalized predictions back to original scale
        """
        # This is percentage change, so no inverse transform needed
        return predictions


class DataAugmentation:
    """
    Data augmentation techniques for time series
    
    IMPROVEMENT: Augmentation for robust training
    """
    
    @staticmethod
    def add_noise(X, noise_level=0.01):
        """Add Gaussian noise"""
        noise = np.random.normal(0, noise_level, X.shape)
        return X + noise
    
    @staticmethod
    def time_warping(X, sigma=0.2):
        """Apply time warping"""
        warped = X.copy()
        n_samples, n_timesteps, n_features = X.shape
        
        for i in range(n_samples):
            # Generate random warping curve
            warp = np.random.normal(1.0, sigma, n_timesteps)
            warp = np.cumsum(warp)
            warp = warp / warp[-1] * n_timesteps
            
            # Interpolate
            for f in range(n_features):
                warped[i, :, f] = np.interp(
                    np.arange(n_timesteps),
                    warp,
                    X[i, :, f]
                )
        
        return warped
    
    @staticmethod
    def magnitude_scaling(X, sigma=0.1):
        """Scale magnitude"""
        factors = np.random.normal(1.0, sigma, (X.shape[0], 1, X.shape[2]))
        return X * factors
    
    @staticmethod
    def augment_dataset(X, y, augmentation_factor=2):
        """
        Augment dataset with multiple techniques
        
        Returns augmented X and y
        """
        X_aug_list = [X]
        y_aug_list = [y]
        
        for _ in range(augmentation_factor - 1):
            # Apply random augmentation
            aug_type = np.random.choice(['noise', 'warp', 'scale'])
            
            if aug_type == 'noise':
                X_aug = DataAugmentation.add_noise(X)
            elif aug_type == 'warp':
                X_aug = DataAugmentation.time_warping(X)
            else:
                X_aug = DataAugmentation.magnitude_scaling(X)
            
            X_aug_list.append(X_aug)
            y_aug_list.append(y)
        
        X_augmented = np.vstack(X_aug_list)
        y_augmented = np.hstack(y_aug_list)
        
        # Shuffle
        indices = np.random.permutation(len(X_augmented))
        return X_augmented[indices], y_augmented[indices]


if __name__ == "__main__":
    # Test preprocessing
    print("Testing data preprocessing...")
    
    # Create dummy data
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    df = pd.DataFrame({
        'Date': dates,
        'Open': np.random.randn(1000).cumsum() + 100,
        'High': np.random.randn(1000).cumsum() + 102,
        'Low': np.random.randn(1000).cumsum() + 98,
        'Close': np.random.randn(1000).cumsum() + 100,
        'Volume': np.random.randint(1000000, 5000000, 1000)
    })
    
    preprocessor = StockDataPreprocessor(window_size=60)
    X, y = preprocessor.create_sequences(df)
    
    print(f"Created sequences: X shape = {X.shape}, y shape = {y.shape}")
    print(f"Number of features: {len(preprocessor.feature_names)}")
    print(f"First 10 features: {preprocessor.feature_names[:10]}")
