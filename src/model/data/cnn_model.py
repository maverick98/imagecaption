import numpy as np
import tensorflow as tf
from time import time



from tqdm import tqdm
#import gensim

class CNN_Model:
    def __init__(self,cnn_type, **kwargs):
        super().__init__(**kwargs)
        self.cnn_type=cnn_type
        self.image_features_extract_model=None

    def load_image_model(self):
        start_time=time()
        if self.image_features_extract_model == None:
            if self.cnn_type == 'Inception':
                image_model = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')
                new_input = image_model.input
                hidden_layer = image_model.layers[-1].output
                self.image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
                #print("Total time taken for load_image_model: %.2fs" % (time() - start_time))              
        return self.image_features_extract_model

    def load_image(self,image_path):
        start_time=time()
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        #print("Total time taken for load_image: %.2fs" % (time() - start_time))   
        return img, image_path

    def extract_feature(self,img):
        start_time=time()
        image_features_extract_model=self.load_image_model()
        
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features,(batch_features.shape[0], -1, batch_features.shape[3]))
        #print("Total time taken for extract_feature: %.2fs" % (time() - start_time))   
        return batch_features

    def preprocess_images(self,img_name_vector,enable_tqdm=True):
        
        start_time=time()
        # Get unique images
        encode_train = sorted(set(img_name_vector))

        # Feel free to change batch_size according to your system configuration
        image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
        image_dataset = image_dataset.map(self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)
        
        #Enable tqdm in Collab
        image_dataset_to_iterate=None
        if enable_tqdm == True:
            image_dataset_to_iterate=tqdm(image_dataset)
        else:
            image_dataset_to_iterate=image_dataset  
        for img, path in image_dataset_to_iterate:
            print("Loading img of shape ",img.shape)
            batch_features = self.extract_feature(img)
            

            for bf, p in zip(batch_features, path):
                path_of_feature = p.numpy().decode("utf-8")
                #print('saving at ',path_of_feature)
                np.save(path_of_feature, bf.numpy())
        print("Total time taken for preprocess_image: %.2fs" % (time() - start_time))           