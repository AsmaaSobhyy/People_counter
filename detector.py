
import os 
# import pathlib

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

# import io
# import scipy.misc
# import numpy as np
# from six import BytesIO
# from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

# from object_detection.utils import label_map_util
from object_detection.utils import config_util
# from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

class DetectorAPI:
    def __init__(self, PATH_TO_MODEL_DIR):
        self.PATH_TO_MODEL_DIR = PATH_TO_MODEL_DIR
        self.pipeline_config = os.path.join(self.PATH_TO_MODEL_DIR+ 'pipeline.config')
        self.model_dir = self.PATH_TO_MODEL_DIR+'checkpoint/'
        # Load pipeline config and build a detection model
        self.configs = config_util.get_configs_from_pipeline_file(self.pipeline_config)
        self.model_config = self.configs['model']
        self.detection_model = model_builder.build(model_config=self.model_config, is_training=False)

        # Restore checkpoint
        self.ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        self.ckpt.restore(os.path.join(self.model_dir, 'ckpt-0')).expect_partial()

        # self.PATH_TO_LABELS= self.PATH_TO_MODEL_DIR+"mscoco_label_map.pbtxt"
        # self.category_index = label_map_util.create_category_index_from_labelmap(self.PATH_TO_LABELS,use_display_name=True)

    @tf.function
    def detect_fn(self,image):
        """Detect objects in image."""

        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    def filter_pred(self,detections,threshold):
        classes = detections['detection_classes'][0]
        boxes= detections['detection_boxes'][0]
        scores = detections['detection_scores'][0]
        scores = scores[classes==0]
        boxes = boxes[classes==0]
        return boxes[scores >= threshold]




# def get_keypoint_tuples(eval_config):
#   """Return a tuple list of keypoint edges from the eval config.
  
#   Args:
#     eval_config: an eval config containing the keypoint edges
  
#   Returns:
#     a list of edge tuples, each in the format (start, end)
#   """
#   tuple_list = []
#   kp_list = eval_config.keypoint_edge
#   for edge in kp_list:
#     tuple_list.append((edge.start, edge.end))
#   return tuple_list




