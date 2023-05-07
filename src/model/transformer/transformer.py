#https://github.com/maverick98/Group4Capstone/blob/main/TransformerModel_demo.ipynb
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras import Model
# import pathlib,sys,os
# src_path = pathlib.Path(__file__).parents[2]
# print(src_path)
# sys.path.append(os.path.dirname(src_path))
from src.model.transformer.encoder import Encoder
from src.model.transformer.decoder import Decoder
from src.model.transformer.positional_encoding import create_masks_decoder





class Transformer(Model):
    def __init__(self,model_params,**kwargs):
        super(Transformer, self).__init__(**kwargs)

        #####################Model params######################
        self.model_params=model_params
        target_vocab_size=model_params.target_vocab_size
        max_pos_encoding=model_params.max_pos_encoding
        row_size=model_params.attn_row_size
        col_size=model_params.attn_col_size
        num_heads=model_params.h
        dropout_rate=model_params.dropout_rate
        d_model=model_params.d_model
        dff=model_params.d_ff
        num_layers=model_params.num_layers
        
        ########################Model params###################

        



        self.encoder = Encoder(num_layers, d_model, num_heads, dff,row_size,col_size, dropout_rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,target_vocab_size,max_pos_encoding, None,dropout_rate)

        self.final_layer = Dense(target_vocab_size)

    def call(self, encoder_input, decoder_input, training, dec_padding_mask=None,enc_padding_mask=None):
        #print("Inside transformer encoder_input shape ",encoder_input.shape)
        #print("Inside transformer decoder_input shape ",decoder_input.shape)
        #print("Inside transformer encoder_input ",encoder_input)
        #print("Inside transformer dec_padding_mask ",dec_padding_mask)
        #print("Inside transformer enc_padding_mask ",enc_padding_mask)
        enc_output = self.encoder(encoder_input, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        look_ahead_mask=create_masks_decoder(decoder_input)
        dec_output, attention_weights = self.decoder(decoder_input, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights
