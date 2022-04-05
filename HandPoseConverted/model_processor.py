import os
import cv2
import numpy as np
import sys

sys.path.append('../')

from acl_model import Model


class ModelProcessor:
    def __init__(self, acl_resource, params):
        self._acl_resource = acl_resource
        self.params = params
        self._model_width = params['width']
        self._model_height = params['height']

        assert 'model_dir' in params and params['model_dir'] is not None, 'Review your param: model_dir'
        assert os.path.exists(params['model_dir']), "Model directory doesn't exist {}".format(params['model_dir'])
            
        # load model from path, and get model ready for inference
        self.model = Model(acl_resource, params['model_dir'])

    def predict(self, img_original):
        # preprocess image to get 'model_input'
        model_input = self.preprocess(img_original)

        # execute model inference
        infer_output = self.model.execute([model_input]) 

        # postprocessing: 
        categories = self.post_process(infer_output)

        return categories

    def preprocess(self, img_original):
        """
        preprocessing: resize image to model required size, and normalize value
        """
        # scaled_img_data = cv2.resize(img_original, (self._model_width, self._model_height))
        # normalized_img = scaled_img_data - np.array([123, 117, 104])

        image_expanded = np.expand_dims(img_original, axis=0)
        # Caffe model after conversion, need input to be NCHW; the orignal image is NHWC,
        # need to be transposed (use .copy() to change memory format)
        # preprocessed_img = np.asarray(normalized_img, dtype=np.float16).transpose([2,0,1]).copy()
        
        return image_expanded

    def post_process(self, infer_output):
        print("we reached post-processing")
        print(self._model_width)
        # data = infer_output[0]
        # vals = data.flatten()
        # top_k = vals.argsort()[-1:-6:-1]
        return infer_output, 1


