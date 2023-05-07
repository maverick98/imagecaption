import numpy as np
import tensorflow as tf
from time import time
from PIL import Image
from keras.applications.inception_v3 import preprocess_input
from pickle import dump, load

#
#Ideally both RNN and Transformer should use the same cnn_model class.
#There is no time for Software Engineering here.
#Thus this extra class
#
class CNN_Model:
    def __init__(self,image_path, **kwargs):
        super().__init__(**kwargs)
        self.model=None
        self.image_path=image_path
          
    def load_model(self): 
        # Load the inception v3 model
        model = tf.keras.applications.InceptionV3(weights='imagenet')
        # Create a new model, by removing the last layer (output layer) from the inception v3
        self.model = tf.keras.Model(model.input, model.layers[-2].output)
        return self.model
    def show_summary(self):
        print(self.model.summary() )
    def preprocess(self,image):  
        # Convert all the images to size 299x299 as expected by the inception v3 model
        #img = Image.load_img(image_path, target_size=(299, 299))
        img=Image.open(image)
        img = img.resize((299, 299))
        # Convert PIL image to numpy array of 3-dimensions
        x = tf.keras.utils.img_to_array(img)
        # Add one more dimension
        x = np.expand_dims(x, axis=0)
        # preprocess the images using preprocess_input() from inception module
        x = preprocess_input(x)
        return x
    # Function to encode a given image into a vector of size (2048, )
    def encode(self,image):
        image = self.preprocess(image) # preprocess the image
        fea_vec = self.model.predict(image) # Get the encoding vector for the image
        fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
        return fea_vec
    # Call the funtion to encode all the train images
    # This will take a while on CPU - Execute this only once
    def encode_images(self,imgs):
        
        #start = time.time()
        encoding_imgs = {}
        count=0;
        for img in imgs:
            count=count+1
            encoding_imgs[img[len(self.image_path):]] = self.encode(img)
            print('Processed image ',count)
        #print("Time taken in seconds =", time.time()-start)
        return encoding_imgs
    def save_encoded_imgs(self,images_pkl_file,encoded_images):
        # Save the bottleneck train features to disk
        with open(images_pkl_file, "wb") as encoded_pickle:
            dump(encoded_images, encoded_pickle)             
