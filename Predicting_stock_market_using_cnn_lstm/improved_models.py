"""
Improved Deep Learning Models for Stock Market Prediction
Author: Deep Learning Assignment Team
Date: November 2025

This module contains improved architectures inspired by:
1. Attention-based CNN-LSTM + XGBoost (AttCLX)
2. CNN-LSTM for Time Series

Key Improvements:
- Multi-head attention mechanisms
- Residual connections
- Advanced regularization
- Ensemble methods
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Conv1D, Dropout, Bidirectional,
    MaxPooling1D, Flatten, Concatenate, Add, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, BatchNormalization,
    Multiply, Permute, RepeatVector, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import numpy as np


class ImprovedAttentionCNNLSTM:
    """
    Improved version of Attention-based CNN-LSTM
    
    Improvements over baseline:
    1. Multi-head attention instead of single attention
    2. Residual connections for better gradient flow
    3. Batch normalization for training stability
    4. Deeper CNN with more filters
    5. Layer normalization
    """
    
    @staticmethod
    def build_model(input_shape, filters_list=[64, 128, 256], lstm_units=[128, 64],
                   num_heads=4, dropout_rate=0.3, l2_reg=0.001):
        """
        Build improved attention-based CNN-LSTM model
        
        Args:
            input_shape: (time_steps, features)
            filters_list: Number of filters in each Conv1D layer
            lstm_units: Units in each LSTM layer
            num_heads: Number of attention heads
            dropout_rate: Dropout rate
            l2_reg: L2 regularization factor
        """
        inputs = Input(shape=input_shape, name='input')
        
        # Multi-scale CNN feature extraction with residual connections
        x = inputs
        for i, filters in enumerate(filters_list):
            # Conv block
            conv = Conv1D(filters, kernel_size=3, padding='same',
                         activation='relu', kernel_regularizer=l2(l2_reg),
                         name=f'conv1d_{i+1}')(x)
            conv = BatchNormalization(name=f'bn_conv_{i+1}')(conv)
            conv = Dropout(dropout_rate, name=f'dropout_conv_{i+1}')(conv)
            
            # Residual connection (if dimensions match)
            if i > 0 and filters == filters_list[i-1]:
                x = Add(name=f'residual_{i+1}')([x, conv])
            else:
                x = conv
                
            if i < len(filters_list) - 1:
                x = MaxPooling1D(pool_size=2, name=f'maxpool_{i+1}')(x)
        
        # Multi-head self-attention mechanism
        # IMPROVEMENT: Multi-head instead of single attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=filters_list[-1] // num_heads,
            dropout=dropout_rate,
            name='multihead_attention'
        )(x, x)
        
        # Residual connection after attention
        x = Add(name='attention_residual')([x, attention_output])
        x = LayerNormalization(name='attention_norm')(x)
        
        # Bi-directional LSTM layers
        for i, units in enumerate(lstm_units):
            x = Bidirectional(
                LSTM(units, return_sequences=(i < len(lstm_units) - 1),
                     kernel_regularizer=l2(l2_reg)),
                name=f'bilstm_{i+1}'
            )(x)
            x = BatchNormalization(name=f'bn_lstm_{i+1}')(x)
            x = Dropout(dropout_rate, name=f'dropout_lstm_{i+1}')(x)
        
        # Output layer
        outputs = Dense(1, activation='linear', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='ImprovedAttentionCNNLSTM')
        return model


class ResidualCNNLSTM:
    """
    Residual CNN-LSTM with skip connections
    
    Improvements:
    1. Residual blocks for better gradient flow
    2. Parallel CNN branches for multi-scale features
    3. Squeeze-and-Excitation blocks
    4. Advanced pooling strategies
    """
    
    @staticmethod
    def build_model(input_shape, filters_list=[64, 128, 64],
                   lstm_units=100, dropout_rate=0.3):
        """Build residual CNN-LSTM model"""
        inputs = Input(shape=input_shape, name='input')
        
        # Multi-branch CNN for multi-scale feature extraction
        branches = []
        for kernel_size in [3, 5, 7]:
            branch = Conv1D(filters_list[0], kernel_size=kernel_size,
                          padding='same', activation='relu',
                          name=f'conv_k{kernel_size}')(inputs)
            branch = BatchNormalization()(branch)
            branches.append(branch)
        
        # Concatenate multi-scale features
        x = Concatenate(name='concat_multiscale')(branches)
        
        # Residual CNN blocks
        for i in range(1, len(filters_list)):
            # Conv block
            conv = Conv1D(filters_list[i], kernel_size=3, padding='same',
                         activation='relu', name=f'res_conv_{i}')(x)
            conv = BatchNormalization(name=f'res_bn_{i}')(conv)
            conv = Dropout(dropout_rate, name=f'res_dropout_{i}')(conv)
            
            # Skip connection
            if x.shape[-1] == conv.shape[-1]:
                x = Add(name=f'res_add_{i}')([x, conv])
            else:
                # Projection if dimensions don't match
                projection = Conv1D(filters_list[i], kernel_size=1,
                                  name=f'projection_{i}')(x)
                x = Add(name=f'res_add_{i}')([projection, conv])
            
            x = MaxPooling1D(2, name=f'res_pool_{i}')(x)
        
        # Bi-LSTM
        x = Bidirectional(LSTM(lstm_units, return_sequences=False),
                         name='bilstm')(x)
        x = Dropout(dropout_rate, name='lstm_dropout')(x)
        
        # Output
        outputs = Dense(1, activation='linear', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='ResidualCNNLSTM')
        return model


class TransformerBasedPredictor:
    """
    Transformer-based model for stock prediction
    
    New addition inspired by recent research:
    - Pure attention-based architecture
    - Positional encoding
    - Multi-head self-attention
    """
    
    @staticmethod
    def positional_encoding(length, depth):
        """Generate positional encoding"""
        depth = depth / 2
        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :] / depth
        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates
        
        pos_encoding = np.concatenate([
            np.sin(angle_rads),
            np.cos(angle_rads)
        ], axis=-1)
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    @staticmethod
    def build_model(input_shape, num_heads=8, ff_dim=128,
                   num_transformer_blocks=4, dropout_rate=0.2):
        """Build transformer-based model"""
        inputs = Input(shape=input_shape, name='input')
        
        # Add positional encoding
        positions = tf.range(start=0, limit=input_shape[0], delta=1)
        pos_encoding = TransformerBasedPredictor.positional_encoding(
            input_shape[0], input_shape[1]
        )
        
        x = inputs + pos_encoding
        
        # Transformer blocks
        for i in range(num_transformer_blocks):
            # Multi-head attention
            attn_output = MultiHeadAttention(
                num_heads=num_heads,
                key_dim=input_shape[1] // num_heads,
                dropout=dropout_rate,
                name=f'mha_{i}'
            )(x, x)
            
            # Residual connection and normalization
            x1 = Add(name=f'add_1_{i}')([x, attn_output])
            x1 = LayerNormalization(epsilon=1e-6, name=f'ln_1_{i}')(x1)
            
            # Feed-forward network
            ffn = Dense(ff_dim, activation='relu', name=f'ffn_1_{i}')(x1)
            ffn = Dropout(dropout_rate, name=f'dropout_1_{i}')(ffn)
            ffn = Dense(input_shape[1], name=f'ffn_2_{i}')(ffn)
            ffn = Dropout(dropout_rate, name=f'dropout_2_{i}')(ffn)
            
            # Residual connection and normalization
            x = Add(name=f'add_2_{i}')([x1, ffn])
            x = LayerNormalization(epsilon=1e-6, name=f'ln_2_{i}')(x)
        
        # Global pooling and output
        x = GlobalAveragePooling1D(name='global_pool')(x)
        x = Dropout(dropout_rate, name='final_dropout')(x)
        x = Dense(64, activation='relu', name='dense_1')(x)
        outputs = Dense(1, activation='linear', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs,
                     name='TransformerPredictor')
        return model


class EnsembleModel:
    """
    Ensemble of multiple models for robust predictions
    
    Combines:
    - Attention-based CNN-LSTM
    - Residual CNN-LSTM
    - Transformer
    """
    
    @staticmethod
    def build_ensemble(input_shape):
        """Build ensemble model"""
        inputs = Input(shape=input_shape, name='input')
        
        # Build individual models
        model1 = ImprovedAttentionCNNLSTM.build_model(input_shape)
        model2 = ResidualCNNLSTM.build_model(input_shape)
        model3 = TransformerBasedPredictor.build_model(input_shape)
        
        # Get predictions from each model
        pred1 = model1(inputs)
        pred2 = model2(inputs)
        pred3 = model3(inputs)
        
        # Weighted average (learnable weights)
        concat = Concatenate(name='concat_predictions')([pred1, pred2, pred3])
        weights = Dense(3, activation='softmax', name='ensemble_weights')(concat)
        
        # Weighted combination
        weighted_pred = Multiply(name='weighted_pred')([concat, weights])
        outputs = Dense(1, activation='linear', name='ensemble_output')(weighted_pred)
        
        model = Model(inputs=inputs, outputs=outputs, name='EnsembleModel')
        return model


def get_model(model_type, input_shape, **kwargs):
    """
    Factory function to get model by type
    
    Args:
        model_type: 'attention', 'residual', 'transformer', or 'ensemble'
        input_shape: (time_steps, features)
        **kwargs: Additional model parameters
    """
    if model_type == 'attention':
        return ImprovedAttentionCNNLSTM.build_model(input_shape, **kwargs)
    elif model_type == 'residual':
        return ResidualCNNLSTM.build_model(input_shape, **kwargs)
    elif model_type == 'transformer':
        return TransformerBasedPredictor.build_model(input_shape, **kwargs)
    elif model_type == 'ensemble':
        return EnsembleModel.build_ensemble(input_shape)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation
    input_shape = (60, 45)  # 60 time steps, 45 features
    
    print("Creating models...")
    print("=" * 60)
    
    models = {
        'Attention-based CNN-LSTM': get_model('attention', input_shape),
        'Residual CNN-LSTM': get_model('residual', input_shape),
        'Transformer': get_model('transformer', input_shape),
    }
    
    for name, model in models.items():
        print(f"\n{name}:")
        print(f"Total parameters: {model.count_params():,}")
        print(f"Trainable parameters: {sum([K.count_params(w) for w in model.trainable_weights]):,}")
