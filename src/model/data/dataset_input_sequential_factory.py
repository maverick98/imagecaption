from dataset_input_sequential import DatasetInput
from cnn_model_sequence import CNN_Model
class DatasetInputSequentialFactory:
    def __init__(self,**kwargs):
          super().__init__(**kwargs)
    def create(self):      
        image_path = "/content/dataset/images/Flicker8k_Dataset/"
        token_file='/content/dataset/texts/Flickr8k.token.txt'
        #token_file='/content/dataset/texts/Flickr8k.lemma.token.txt'

        train_images_text_file='/content/dataset/texts/Flickr_8k.trainImages.txt'
        dev_images_text_file='/content/dataset/texts/Flickr_8k.devImages.txt'
        test_images_text_file='/content/dataset/texts/Flickr_8k.testImages.txt'
        descriptions_file='/content/drive/MyDrive/Capstone/sequential/descriptions.txt'
        data_limit=100
        high_frequency_threshold=10

        glove_dir = '/content/dataset/glove'
        glove_embedding_file='glove.6B.300d.txt'
        embedding_dim=300

        encoded_train_images_pkl='/content/drive/MyDrive/Capstone/sequential/encoded_train_images.pkl'
        encoded_dev_images_pkl='/content/drive/MyDrive/Capstone/sequential/encoded_dev_images.pkl'
        encoded_test_images_pkl='/content/drive/MyDrive/Capstone/sequential/encoded_test_images.pkl'

        epochs = 30
        num_photos_per_batch = 3

        model_weights_path='/content/drive/MyDrive/Capstone/sequential/model_weights'
        test_image_caption_path='/content/drive/MyDrive/Capstone/sequential/test_captions'
        cnn_model=CNN_Model(image_path)
        datasetInput=DatasetInput()
        datasetInput.set_cnn_model(cnn_model)\
                    .set_image_path(image_path)\
                    .set_token_file(token_file)\
                    .set_train_images_text_file(train_images_text_file)\
                    .set_dev_images_text_file(dev_images_text_file)\
                    .set_test_images_text_file(test_images_text_file)\
                    .set_descriptions_file(descriptions_file)\
                    .set_data_limit(data_limit)\
                    .set_encoded_train_images_pkl(encoded_train_images_pkl)\
                    .set_encoded_dev_images_pkl(encoded_dev_images_pkl)\
                    .set_encoded_test_images_pkl(encoded_test_images_pkl)\
                    .set_glove_dir(glove_dir)\
                    .set_glove_embedding_file(glove_embedding_file)\
                    .set_embedding_dim(embedding_dim)\
                    .set_high_frequency_threshold(high_frequency_threshold)\
                    .set_epochs(epochs)\
                    .set_num_photos_per_batch(num_photos_per_batch)\
                    .set_model_weights_path(model_weights_path)\
                    .set_test_image_caption_path(test_image_caption_path)
         
        #print(datasetInput.cnn_model)

        return datasetInput