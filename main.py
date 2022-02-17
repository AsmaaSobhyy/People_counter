from detector import *
from helper_functions import *
from videomaker import *

from skimage import io, transform, color
from object_detection.utils import label_map_util
import numpy as np
import cv2





  

if __name__ == "__main__":

    threshold = 0.4
    ROIs=[[10,8,757,566],[287,156,711,431],[27,129,230,289]]
    model_path = "model/"
    data_path = "S1_L1/"

    model = DetectorAPI(model_path)
    # paths to all images folders
    _,images_paths = get_paths(data_path)
    # create the same hierarchy for the outputted images
    create_hierarchy(images_paths)


    for img_path in images_paths:
        print(img_path) # to make sure everything is working
        # do this for each folder containing images at a time
        images,names = read_images(img_path)
        # read each image at a time, detect human then save the image with
        #  the total number of people drawn on it in each roi
        for ind,img in enumerate(images):
            
            image_path = os.path.join(img_path,names[ind])
            image_np = io.imread(image_path)
            input_tensor = tf.convert_to_tensor(
            np.expand_dims(image_np, 0), dtype=tf.float32)

            detections, predictions_dict, shapes =model.detect_fn(input_tensor)
            boxes = model.filter_pred(detections,threshold)
            boxes = to_pixel_coords(np.asarray(boxes),image_np.shape)

            people_count = count_people(ROIs,boxes)
            # draw ROI boxes and counts over the image then saving it 
            for i,r in enumerate(ROIs):
                image_np = cv2.rectangle(image_np, (r[0],r[1]), (r[2],r[3]), (255, 0, 0), 2)
                image_np = cv2.putText(image_np, str(people_count[i]), (r[0]+10,r[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.imwrite("out/"+image_path,image_np)

    #Save to videos
    print("------ Saving to videos ------")
    speed = 5
    shape=(768,576,3)
    videomaker = VideoMaker(images_paths,shape,speed)
    videomaker.save_videos()