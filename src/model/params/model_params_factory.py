from src.model.params.model_params import TransformerModelParams
import os
import numpy as np


class ModelParamsFactory:
    def __init__(self,**kwargs):
          super().__init__(**kwargs)
    
    
    def create(self):
        model_params=TransformerModelParams()

        num_words = 8357  # We know this from EDA 
        attn_row_size=8
        attn_col_size=8
        target_vocab_size=num_words+1
        max_pos_encoding=num_words+1
        h=8
        d_k=64
        d_v=64
        d_ff=2048
        d_model=512
        num_layers=6 #Think of reducing to 4 if required.
        dropout_rate=0.1
        caption_max_len=33 #Need to revisit this.
       

        model_params.set_attn_row_size(attn_row_size)\
                    .set_attn_col_size(attn_col_size)\
                    .set_h(h)\
                    .set_d_k(d_k)\
                    .set_d_v(d_v)\
                    .set_d_model(d_model)\
                    .set_d_ff(d_ff)\
                    .set_num_layers(num_layers)\
                    .set_target_vocab_size(target_vocab_size)\
                    .set_max_pos_encoding(max_pos_encoding)\
                    .set_dropout_rate(dropout_rate)\
                    .set_caption_max_len(caption_max_len)
                    
        return model_params




