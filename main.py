from detector import *
from read_save_images import *
from videomaker import *
from skimage import io, transform, color
from object_detection.utils import label_map_util
import numpy as np
import cv2

PATH_TO_LABELS= "mscoco_label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,use_display_name=True)


model = DetectorAPI("efficientdet_d2_coco17_tpu-32/")


_,images_paths = get_paths("S1_L1/Crowd_PETS09/")
create_hir(images_paths)

threshold = 0.4
ROIs=[[10,8,757,566],[287,156,711,431],[27,129,230,289]]

for img_path in images_paths:
    images,names = read_images(img_path)
    for ind,img in enumerate(images):
        image_path = os.path.join(img_path,names[ind])
        image_np = io.imread(image_path)
        input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.float32)

        detections, predictions_dict, shapes=model.detect_fn(input_tensor)
        boxes = model.filter_pred(detections,threshold)
        boxes = to_pixel_coords(np.asarray(boxes),image_np.shape)

        people_count = count_people(ROIs,boxes)
        print(people_count)
        for i,r in enumerate(ROIs):
            image_np = cv2.rectangle(image_np, (r[0],r[1]), (r[2],r[3]), (255, 0, 0), 2)
            image_np = cv2.putText(image_np, str(people_count[i]), (r[0]+10,r[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imwrite("out/"+image_path,image_np)







        
        


# def detect_all(path):

ROIs=[[10,8,757,566],[287,156,711,431],[27,129,230,289]]

     


# print(to_pixel_coords(test_coord)) # prints (960, 324)


image_dir = 'test'
image_path = os.path.join(image_dir, 'frame_0012.jpg')
image_np= io.imread(image_path)


input_tensor = tf.convert_to_tensor(
    np.expand_dims(image_np, 0), dtype=tf.float32)

detections, predictions_dict, shapes=model.detect_fn(input_tensor)
classes=detections['detection_classes'][0]
boxes= detections['detection_boxes'][0]
scores = detections['detection_scores'][0]
scores=scores[classes==0]
boxes = boxes[classes==0]
    # print("True")
# print(classes[classes==0])
# print(boxes[scores>0.5])
# print(scores[classes==0])





label_id_offset = 1
image_np_with_detections = image_np.copy()