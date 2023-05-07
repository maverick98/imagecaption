#
#Ideally both RNN and Transformer should use the same model params and training class.
#There is no time for Software Engineering here.
#Thus this extra class
#
class DatasetInput:
      def __init__(self,**kwargs):
          super().__init__(**kwargs)
      def set_cnn_model(self,value):
          self.cnn_model=value
          return self
      def set_image_path(self,value):
          self.image_path=value
          return self
      def set_token_file(self,value):
          self.token_file=value
          return self
      def set_train_images_text_file(self,value):
          self.train_images_text_file=value
          return self
      def set_dev_images_text_file(self,value):
          self.dev_images_text_file=value
          return self
      def set_test_images_text_file(self,value):
          self.test_images_text_file=value
          return self    
      def set_descriptions_file(self,value):
          self.descriptions_file=value
          return self   
      def set_encoded_train_images_pkl(self,value):
          self.encoded_train_images_pkl=value
          return self 
      def set_encoded_dev_images_pkl(self,value):
          self.encoded_dev_images_pkl=value
          return self 
      def set_encoded_test_images_pkl(self,value):
          self.encoded_test_images_pkl=value
          return self
      def set_data_limit(self,value):
          self.data_limit=value
          return self  
      def set_glove_dir(self,value):
          self.glove_dir=value
          return self
      def set_glove_embedding_file(self,value):
          self.glove_embedding_file=value
          return self
      def set_high_frequency_threshold(self,value):
          self.high_frequency_threshold=value 
          return self
      def set_epochs(self,value):
          self.epochs=value
          return self
      def set_num_photos_per_batch(self,value):    
          self.num_photos_per_batch=value
          return self
      def set_embedding_dim(self,value):
          self.embedding_dim=value
          return self
      def set_model_weights_path(self,value):
          self.model_weights_path=value
          return self
      def set_test_image_caption_path(self,value):
          self.set_test_image_caption_path=value
          return self
      def set_train_image_caption_path(self,value):
          self.train_image_caption_path=value
          return self
      def set_test_image_caption_path(self,value):
          self.test_image_caption_path=value
          return self         
                                           
