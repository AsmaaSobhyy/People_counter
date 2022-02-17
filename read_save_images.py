import os
from skimage import io

# a function to collect all paths recursively
def get_paths(path,memo=[]):
    # print(os.listdir(path)[0])
    if "jpg" in os.listdir(path)[0]:
        memo.append(path)
        return path,memo
    for p in os.listdir(path):
        if "." not in p:
            subpath = os.path.join(path, p)
            subpath,memo = get_paths(subpath,memo)
    return path,memo

def create_hir(paths):
    for p in paths:
        os.makedirs('out/'+p)

def read_images(path):
    images=[]
    number=[]
    for img in os.listdir(path):
        # print(img)
        image= io.imread(os.path.join(path,img))
        images.append(image)
        number.append(img)
    return images, number


# def count_people(ROIs,predictions):

# def save_image(ROIs,detections):

# _,paths = get_paths("S1_L1/Crowd_PETS09/")



