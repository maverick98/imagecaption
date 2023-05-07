from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras import Sequential




def point_wise_feed_forward_network(d_model, dff):
    return Sequential([
        Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


