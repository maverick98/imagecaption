#https://github.com/maverick98/Group4Capstone/blob/main/encoder_demo.ipynb
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LayerNormalization
from tensorflow import shape
from src.model.transformer.positional_encoding import positional_encoding_2d
from src.model.transformer.multihead_attention import MultiHeadAttention
from src.model.transformer.feedforward import point_wise_feed_forward_network



    

class EncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training, mask=None):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

    

       

class Encoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,row_size,col_size,rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = Dense(self.d_model,activation='relu')
        self.pos_encoding = positional_encoding_2d(row_size,col_size, self.d_model)


        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = Dropout(rate)
        
    def call(self, x, training, mask=None):
        # shape(x) = (batch_size,seq_len(H*W),features)
        seq_len = shape(x)[1]
        #print("Inside encoder Training mode ",training)
        #if mask is not None:
        #    print("Inside encoder mask ",mask)
        #print("Inside encoder x",x.shape)
        #print("Inside encoder seq_len",seq_len)
        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len(H*W), d_model)
        #print("Inside encoder x after embedding",x.shape)
        #print("Inside encoder pos_encoding is ",self.pos_encoding.shape)
        x += self.pos_encoding[:, :seq_len, :]
        #print("Inside encoder x after pos encoding",x.shape)

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)
