# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 10:02:33 2017

@author: user98
"""

import json
datajson="../data/scene_validation_annotations_20170908.json"
traindir="../data/scene_train_images_20170904"
testdir="../data/scene_validation_images_20170908"
with open(datajson) as json_data:
    d = json.load(json_data)

dic={}
for i in range(len(d)):
    dic[d[i].get('image_id')]=int(d[i].get('label_id'))

test=np.empty((len(d),2),dtype="<U50")
for i in range(len(d)):
    test[i,0]=d[i].get('image_id')
    test[i,1]=d[i].get('label_id')