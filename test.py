# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 10:02:33 2017

@author: user98
"""

import json
datajson="../data/scene_train_annotations_20170904.json"
with open(datajson) as json_data:
    d = json.load(json_data)

dic={}
for i in range(len(d)):
    dic[d[i].get('image_id')]=int(d[i].get('label_id'))