import pickle
import numpy as np
import glob
import json

file=glob.glob("*.pickle")
a=[]
for fi in file:
    with open(fi, 'rb') as f:
         ddd=pickle.load(f)
         temp={}
         temp["image_id"]=ddd["file_path"].replace("../data/scene_classification/scene_test_a_images_20170922/","")
         temp["label_id"]=list(map(int, ddd["top3"]))
         a.append(temp)
with open('result.json', 'w') as outfile:
    json.dump(a, outfile)