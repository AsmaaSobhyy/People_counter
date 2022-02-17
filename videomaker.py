
import cv2
import os
import numpy as np
from skimage import io, transform, color



class VideoMaker:
    def __init__(self, paths,shape,speed):
        self.images_paths = paths
        self.shape = shape
        self.speed = speed



    def save_to_video(self,path,target_path):
        h,w,c = self.shape
        images=[]
        number=[]
        for img in os.listdir(path):
            # print(img)
            image= io.imread(os.path.join(path,img))
            images.append(image)
            number.append(img)

        number, images = (list(t) for t in zip(*sorted(zip(number, images))))

        out = cv2.VideoWriter(target_path+'.avi',cv2.VideoWriter_fourcc(*'DIVX'),self.speed,(h, w))

        for i in range(len(images)):
            out.write(images[i])

        out.release()
        return out


    def save_videos(self):
        # create the final path if it's not 
        try:
            os.makedirs('final_videos/')
        except:
            print("path already exists")

        target_path = 'final_videos/'

        for i,path in enumerate(self.images_paths):
            target=target_path+str(i)
            out = self.save_to_video('out/'+path,target)


# test_paths = ['S1_L1/Crowd_PETS09/S1/L1/Time_13-57/View_001','S1_L1/Crowd_PETS09/S1/L1/Time_13-57/View_002']

# videomaker = VideoMaker(test_paths,(768,576,3),5)
# videomaker.save_videos()
