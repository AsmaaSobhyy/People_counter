import os
from skimage import io




# a function to collect all paths recursively
def get_paths(path,memo=[]):
    if "jpg" in os.listdir(path)[0]:
        memo.append(path)
        return path,memo
    for p in os.listdir(path):
        if "." not in p:
            subpath = os.path.join(path, p)
            subpath,memo = get_paths(subpath,memo)
    return path,memo



# create the same hierarchy as the input images paths
def create_hierarchy(paths):
    try:
        for p in paths:
            os.makedirs('out/'+p)
    except:
        print("path already exists")



# read images in a single foldes
def read_images(path):
    images=[]
    number=[]
    for img in os.listdir(path):
        image= io.imread(os.path.join(path,img))
        images.append(image)
        number.append(img)
    return images, number



# convert normalized coordinates into original ones and
 #reorder to xmin,ymin,xmax,ymax 
def to_pixel_coords(relative_coords,shape):
    h,w,_ = shape
    new_coord = relative_coords.copy()
    for i,cor in enumerate(relative_coords):
        new_coord[i][1] = int(cor[0]*h)
        new_coord[i][0] = int(cor[1]*w)
        new_coord[i][3] = int(cor[2]*h )
        new_coord[i][2] = int(cor[3]*w)
    return new_coord



# check if 2 boxes areoverlapping
def isOverlapping(box1,box2):
    # xmin,ymin,xmax,ymax

    # If one rectangle is on left side of other
    if box1[0] > box2[2] or box2[0] > box1[2]:
        return False
    # If one rectangle is above other
    # ymax1 > ymin2
    if (box1[3] < box2[1]) or( box2[3] < box1[1]):
        return False
    return True



# count people within an ROI
def count_people(ROIs,boxes):
    count = [0 for x in range(len(ROIs))]
    for box in boxes:
        for ind,roi in enumerate(ROIs):
            if isOverlapping(box,roi):
                count[ind]+=1
    return count





