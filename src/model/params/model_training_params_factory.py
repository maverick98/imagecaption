from src.model.params.model_training_params import TransformerTrainingParams

class ModelTrainingParamsFactory:
    def __init__(self,**kwargs):
          super().__init__(**kwargs)

    def create(self):
        model_training_params=TransformerTrainingParams()

        batch_size=128

        epochs=100
        beta_1=0.9
        beta_2=0.98
        epsilon=1e-9
        checkpoints_path='./checkpoints' 
        max_to_keep=3
        train_image_caption_path='/content/drive/MyDrive/Capstone/train_captions'
        test_image_caption_path='/content/drive/MyDrive/Capstone/test_captions'
       
        image_path = "/content/dataset/images/Flicker8k_Dataset/"
        #token_file='/content/dataset/texts/Flickr8k.lemma.token.txt'
        token_file='/content/dataset/texts/Flickr8k.token.txt'
        token_file_train='/content/dataset/texts/Flickr_8k.trainImages.txt'
        token_file_val='/content/dataset/texts/Flickr_8k.devImages.txt'
        token_file_test='/content/dataset/texts/Flickr_8k.testImages.txt'
        data_limit_train=30000 # We know this from EDA
        data_limit_val=5000 # We know this from EDA
        data_limit_test=5000 # We know this from EDA

        #data_limit_train=10 # We know this from EDA
        #data_limit_val=5 # We know this from EDA
        #data_limit_test=5 # We know this from EDA

        num_words = 8357  # We know this from EDA 
        caption_max_len=33 #Need to revisit this. 
        use_train_val_test_split=True #Use the train-dev-test split of Flickr
        glove_dir = '/content/dataset/glove'
        glove_embedding_file='glove.6B.300d.txt'
        monitor="val_loss"
        min_delta=0.001
        patience=5
        model_output='/content/drive/MyDrive/Capstone/weights/saved_model'


        model_training_params.set_epochs(epochs)\
                     .set_batch_size(batch_size)\
                     .set_beta_1(beta_1)\
                     .set_beta_2(beta_2)\
                     .set_epsilon(epsilon)\
                     .set_checkpoints_path(checkpoints_path)\
                     .set_max_to_keep(max_to_keep)\
                     .set_train_image_caption_path(train_image_caption_path)\
                     .set_test_image_caption_path(test_image_caption_path)\
                     .set_image_path(image_path)\
                     .set_token_file(token_file)\
                     .set_token_file_train(token_file_train)\
                     .set_token_file_val(token_file_val)\
                     .set_token_file_test(token_file_test)\
                     .set_data_limit_train(data_limit_train)\
                     .set_data_limit_val(data_limit_val)\
                     .set_data_limit_test(data_limit_test)\
                     .set_num_words(num_words)\
                     .set_caption_max_len(caption_max_len)\
                     .set_use_train_val_test_split(use_train_val_test_split)\
                     .set_glove_dir(glove_dir)\
                     .set_glove_embedding_file(glove_embedding_file)\
                     .set_monitor(monitor)\
                     .set_min_delta(min_delta)\
                     .set_patience(patience)\
                     .set_model_output(model_output)
        return model_training_params




