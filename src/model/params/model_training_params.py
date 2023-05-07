import json
class TransformerTrainingParams:
      def __init__(self,**kwargs):
          super().__init__(**kwargs)
      def toJSON(self):
          return json.dumps(self, default=lambda o: o.__dict__,sort_keys=True, indent=4)    
      def set_epochs(self,value):
          self.epochs=value
          return self
      def set_batch_size(self,value):
          self.batch_size=value
          return self
      def set_beta_1(self,value):
          self.beta_1=value
          return self
      def set_beta_2(self,value):
          self.beta_2=value
          return self
      def set_epsilon(self,value):
          self.epsilon=value
          return self
      
      def set_checkpoints_path(self,value):
          self.checkpoints_path=value
          return self
      def set_max_to_keep(self,value):
          self.max_to_keep=value
          return self
      def set_train_image_caption_path(self,value):
          self.train_image_caption_path=value
          return self
      def set_test_image_caption_path(self,value):
          self.test_image_caption_path=value
          return self
      def set_image_path(self,value):
          self.image_path=value
          return self 
      def set_token_file(self,value):
          self.token_file=value
          return self
      def set_token_file_train(self,value):
          self.token_file_train=value
          return self
      def set_token_file_val(self,value):
          self.token_file_val=value
          return self
      def set_token_file_test(self,value):
          self.token_file_test=value
          return self
      
      def set_data_limit_train(self,value):
          self.data_limit_train=value
          return self
      def set_data_limit_val(self,value):
          self.data_limit_val=value
          return self
      def set_data_limit_test(self,value):
          self.data_limit_test=value
          return self
      def set_num_words(self,value):
          self.num_words=value
          return self
      def set_caption_max_len(self,value):
          self.caption_max_len=value
          return self
      def set_use_train_val_test_split(self,value): 
          self.use_train_val_test_split=value
          return self
      def set_glove_dir(self,value): 
          self.glove_dir=value
          return self
      def set_glove_embedding_file(self,value): 
          self.glove_embedding_file=value
          return self
      def set_min_delta(self,value): 
          self.min_delta=value
          return self
      def set_patience(self,value): 
          self.patience=value
          return self
      def set_monitor(self,value): 
          self.monitor=value
          return self
      def set_model_output(self,value): 
          self.model_output=value
          return self
      
       
      
      
      
            