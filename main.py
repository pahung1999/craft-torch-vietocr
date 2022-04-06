import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import cv2
import numpy as np
import craft_utils
import imgproc
import file_utils
from ocr import *
from craft import CRAFT
from test_copy import *


#Convertdata
import pandas as pd
from dataclasses import dataclass
from collections import namedtuple
from ast import literal_eval
from itertools import islice
from functools import partial
import ast

#Craft config
trained_model='weights/craft_mlt_25k.pth'
text_threshold=0.7
low_text=0.4
link_threshold=0.4
cuda=True
canvas_size=1280
mag_ratio=1.5
poly=False
show_time=False
test_folder='./image_test'
refine=False
refiner_model='weights/craft_refiner_CTW1500.pth'
result_folder="./result"
test_folder="./image_test"
jsonl_path="jsonfile/"

#VietOCR config
config_ocr = Cfg.load_config_from_name('vgg_transformer')
config_ocr['weights'] = 'weights/transformerocr.pth'
config_ocr['cnn']['pretrained']= False
config_ocr['device'] = 'cuda:0'
config_ocr['predictor']['beamsearch']= False
detector = Predictor(config_ocr)



if __name__== "__main__" :
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + trained_model + ')')
    if cuda:
        net.load_state_dict(copyStateDict(torch.load(trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu')))


    if cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + refiner_model + ')')
        if cuda:
            refine_net.load_state_dict(copyStateDict(refiner_model))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(refiner_model, map_location='cpu')))

        refine_net.eval()
        poly = True


    """ For test images in a folder """
    image_list, _, _ = file_utils.get_files(test_folder)
    result_folder = './result/'
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    # t = time.time()
    # t= time.perf_counter_ns()
    # load data
    # with torch.no_grad():
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)
        # image=image/255
        bboxes, polys, score_text = test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly,canvas_size, mag_ratio)
        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

        # print("elapsed time : {}s".format((time.perf_counter_ns() - t)/(1e+9)))

    #VietOCR


    #Sử dụng Vietocr để đọc các box
    #Get box from txt file
    df=pd.DataFrame(columns=["id","texts","bboxes","width","height"])

    for filename in os.listdir(test_folder):
        full_path=os.path.join(test_folder,filename)
        bboxes=get_bbox(filename,result_folder)

        #Convert box to rectangular
        bboxes=box_convert(bboxes)

        img=cv2.imread(full_path)
        #VietOCR
        
        raw_text=Vietocr_img(img,bboxes,detector)
        height, width, channels = img.shape
        df=df.append({'id':filename,'texts':raw_text,"bboxes":bboxes,"width":width,"height":height},ignore_index=True)
        
    df=data_arrange(df)
    if not os.path.isdir(jsonl_path):
        os.mkdir(jsonl_path)
    with open(jsonl_path+'result.jsonl', 'w', encoding='utf-8') as file:
        df.to_json(file,orient='records', lines=True,force_ascii=False)
