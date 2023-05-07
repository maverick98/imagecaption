import pickle
import tensorflow as tf
import properties
import sys
import os
# from happytransformer import HappyTextToText
# from happytransformer import TTSettings
from gingerit.gingerit import GingerIt


# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
# print(SCRIPT_DIR)

import pathlib
src_path = pathlib.Path(__file__).parents[2]
print(src_path)
sys.path.append(os.path.dirname(src_path))

from src.evaluate.inference import Inference

def load_pretrained_files():
    with open(properties.CAPTION_PROCESSOR_PICKLE_FILE, 'rb') as handle:
        cp = pickle.load(handle)

    with open(properties.INCEPTION_MODEL_PICKLE_FILE, 'rb') as handle:
        im = pickle.load(handle)

    model = tf.saved_model.load(properties.MODEL_PATH, tags=None, options=None)
    return cp, im, model

def inference_function(input_image_path):
    cp, im, model = load_pretrained_files()
    inference= Inference(model,im,cp)
    caption_words,_,_=inference.extract_caption(input_image_path,False)
    
    return caption_words


def grammar_correction(input_text):
    parser = GingerIt()
    # happy_tt = HappyTextToText("T5",  "prithivida/grammar_error_correcter_v1")

    # settings = TTSettings(do_sample=True, top_k=10, temperature=0.5,  min_length=1, max_length=100)

    # result = happy_tt.generate_text(input_text, args=settings)
    res = parser.parse(input_text)['result']
    res_list = res.split(' ')
    previous_value = None
    new_lst = []

    for elem in res_list:
        if elem != previous_value:
            new_lst.append(elem)
            previous_value = elem

    res = ' '.join(new_lst)
    return res