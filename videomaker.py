
import cv2
import os
import numpy as np
from skimage import io, transform, color

def saveto_video(path):
    images=[]
    number=[]
    for img in os.listdir(path):
        # print(img)
        image= io.imread(os.path.join(path,img))
        images.append(image)
        number.append(img)

    number, images = (list(t) for t in zip(*sorted(zip(number, images))))

    out = cv2.VideoWriter('test.avi',cv2.VideoWriter_fourcc(*'DIVX'), 5,(768, 576))

    for i in range(len(images)):
        out.write(images[i])
    out.release()
    return out


path ="S1_L1/Crowd_PETS09/S1/L1/Time_13-59/View_001"
# saveto_video(path)

# ROIs=[[10,8,757,566],[287,156,711,431],[27,129,230,289]]
# detection=[3,1,5]
# image_dir = 'test'
# image_path = os.path.join(image_dir, 'frame_0012.jpg')
# image_np= io.imread(image_path)
# for i,r in enumerate(ROIs):
#     image_np = cv2.rectangle(image_np, (r[0],r[1]), (r[2],r[3]), (255, 0, 0), 2)
#     image_np = cv2.putText(image_np, str(detection[i]), (r[0]+10,r[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 
#                    1, (255, 0, 0), 1, cv2.LINE_AA)

coor = [[0.40133488, 0.66025114, 0.58037764, 0.7347317 ],
[0.25122064, 0.11024494, 0.35075065, 0.14044665]]

def to_pixel_coords(relative_coords,shape):
    # print(relative_coords)
    #reorder to xmin,ymin,xmax,ymax and map back to original pixels
    h,w,_ = shape
    new_coord = relative_coords.copy()
    for i,cor in enumerate(relative_coords):
        new_coord[i][1] = int(cor[0]*h)
        new_coord[i][0] = int(cor[1]*w)
        new_coord[i][3] = int(cor[2]*h )
        new_coord[i][2] = int(cor[3]*w)
    return new_coord

def isOverlapping(box1,box2):
    # If one rectangle is on left side of other
    # xmin,ymin,xmax,ymax
    # xmin1 > xmax2 or xmin2 > xmax1
    # print(box1)
    # print(box2)
    if box1[0] > box2[2] or box2[0] > box1[2]:
        return False
    # If one rectangle is above other
    # ymax1 > ymin2
    if (box1[3] < box2[1]) or( box2[3] < box1[1]):
        return False
    return True

def count_people(ROIs,boxes):

    count = [0 for x in range(len(ROIs))]

    for box in boxes:
        for ind,roi in enumerate(ROIs):
            if isOverlapping(box,roi):
                count[ind]+=1
    return count
            


ROIs=[[10,8,757,566],[287,156,711,431],[27,129,230,289]]

# coor = to_pixel_coords(coor,image_np.shape)
# print(coor)
# print(ROIs)
# print(count_people(ROIs,coor))

# h,w,_ = image_np.shape
# print(int(coor[0]*w))
# for cor in coor:
#  image_np = cv2.rectangle(image_np, (int(cor[0]),int(cor[1])), (int(cor[2]),int(cor[3])), (255, 0, 0), 2)
# for roi in ROIs:
#     image_np = cv2.rectangle(image_np, (int(roi[0]),int(roi[1])), (int(roi[2]),int(roi[3])), (255, 0, 0), 2)


# cv2.imwrite("test.jpg",image_np )
