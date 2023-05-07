import json
class TransformerModelParams:
      def __init__(self,**kwargs):
          super().__init__(**kwargs)
      def toJSON(self):
          return json.dumps(self, default=lambda o: o.__dict__,sort_keys=True, indent=4)        
      def set_attn_row_size(self,value):
          self.attn_row_size=value
          return self
      def set_attn_col_size(self,value):
          self.attn_col_size=value
          return self    
      def set_h(self,value):
          self.h=value
          return self
      def set_d_k(self,value):
          self.d_k=value
          return self
      def set_d_v(self,value):
          self.d_v=value
          return self
      def set_d_model(self,value):
          self.d_model=value
          return self
      def set_d_ff(self,value):
          self.d_ff=value
          return self
      def set_num_layers(self,value):
          self.num_layers=value
          return self 
      def set_target_vocab_size(self,value):
          self.target_vocab_size=value
          return self
      def set_max_pos_encoding(self,value):
          self.max_pos_encoding=value
          return self 
      def set_dropout_rate(self,value):
          self.dropout_rate=value
          return self
      def set_caption_max_len(self,value):
          self.caption_max_len=value
          return self
      
      
                        