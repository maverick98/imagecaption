from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import LayerNormalization



#import gensim


class AddNormalization(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm=LayerNormalization()
    def call(self, x,sublayer_x):
        return self.layer_norm(x+sublayer_x)