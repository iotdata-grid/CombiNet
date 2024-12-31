from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
import glob
import pickle
import pathlib

import multiprocessing as mp
import threading, random, socket, socketserver, time, pickle

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker, build_multitracker
from torch.autograd import Variable
from keyframe.featuremap import feature
from keyframe.keyframe_choosing import is_key, get_frames
from datasets.vid_dataset import ImagenetDataset
from datasets.data_preprocessing import group_annotation_by_class

from ecci_sdk import Client
from threading import Thread

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument("--dataset", type=str, default="",
                    help="The root directory of the VOC dataset or Open Images dataset.")
parser.add_argument('--config', type=str, help='config file', default="")
parser.add_argument('--snapshot', type=str, help='model name', default="")
parser.add_argument("--label_file", type=str, default="", help="The label file path.")
parser.add_argument("--eval_dir", default="", type=str,
                    help="The directory to store evaluation results.")

args = parser.parse_args()

def write_txt(dataset, f, bbox, label, probs):
    class_names = [name.strip() for name in open(args.label_file).readlines()]
    eval_path = pathlib.Path(args.eval_dir)
    eval_path.mkdir(exist_ok=True)

    for class_index, class_name in enumerate(class_names):
        if label == class_name:
            break
    prediction_path = eval_path / f"det_test_{class_name}.txt"
    with open(prediction_path, "a") as g:
        image_id = dataset.ids[int(f)]
        g.write(str(image_id) + " " + " " + str(probs) + " " + str(bbox[0]) + " "+ str(bbox[1])+ " "+ str(bbox[2] )+ " "+ str(bbox[3] )+ "\n")

    
def main():
    # Initialize ecci sdk and connect to the broker in edge-cloud
    ecci_client = Client()
    mqtt_thread = threading.Thread(target=ecci_client.initialize)
    mqtt_thread.start()
    ecci_client.wait_for_ready()
    print('edge start --------')

    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder(branchpoint=15)
    model_branchpoint = ModelBuilder(branchpoint=8)

    # load model   
    checkpoint = torch.load(args.snapshot)
    model.load_state_dict(checkpoint)
    for param in model.parameters():
        param.requires_grad = False
    model.eval().to(device)

    #multiprocessing
    manager = mp.Manager()
    resQueue = manager.Queue()
    multiProcess = []

    # VID dataloader
    dataset = ImagenetDataset(args.dataset, is_val=True)
    true_case_stat, all_gb_boxes = group_annotation_by_class(dataset)

    for f in range(len(dataset)):
        frame, first_frame = dataset.get_image(f)
        
        if first_frame:        
            # keyframe need to be uploaded to cloud 
            print('first frame upload to cloud')
            outputs = model_branchpoint.get_feature_map(frame)
            print("image num:", f)

            # close the last multiprocessing
            for i in range(len(multiProcess)):
                multiProcess[i].join()

            # send frame to cloud
            payload = {"type":"data","contents":{"outputs":outputs}}
            print("####################",payload)
            ecci_client.publish(payload, "cloud")

            # get rect from cloud
            cloud_data = ecci_client.get_sub_data_payload_queue().get()
            print("###########recieve data from cloud",cloud_data)
            bbox= cloud_data["bbox"]
            label = cloud_data["label"]
            probs = cloud_data["probs"]

            # wirte txt
            for i in range(len(bbox)):
                write_txt(dataset, f, bbox[i], label[i], probs[i])

            # start multiprocessing
            multiProcess = []
            for i in range(len(bbox)):
                # cv2.rectangle(frame, (int(bbox[i][0]),int(bbox[i][1])),(int(bbox[i][2]),int(bbox[i][3])),(0, 255, 255), 3)
                multiProcess.append(build_multitracker(model,label[i],probs[i],resQueue))
            for i in range(len(multiProcess)):
                init_rect = [bbox[i][0],bbox[i][1],bbox[i][2]-bbox[i][0],bbox[i][3]-bbox[i][1]]
                multiProcess[i].init(frame, init_rect)
                multiProcess[i].start()
                
            key_frame = frame   
            first_frame = False

        # elif is_key(key_frame, frame):
        elif f % 10 == 0:
            # keyframe need to be uploaded to cloud ##### outputs, time ######
            print('key frame upload to cloud')
            outputs = model_branchpoint.get_feature_map(frame)
           
            # close the last multiprocessing
            for i in range(len(multiProcess)):
                multiProcess[i].join()

            # send frame to cloud
            payload = {"type":"data","contents":{"outputs":outputs}}
            print("####################",payload)
            ecci_client.publish(payload, "cloud")

            # get rect from cloud
            cloud_data = ecci_client.get_sub_data_payload_queue().get()
            print("###########recieve data from cloud",cloud_data)
            bbox= cloud_data["bbox"]
            label = cloud_data["label"]
            probs = cloud_data["probs"]
            
            # wirte txt
            for i in range(len(bbox)):
                write_txt(dataset, f, bbox[i], label[i], probs[i])

            # start multiprocessing
            multiProcess = []
            for i in range(len(bbox)):
                # cv2.rectangle(frame, (int(bbox[i][0]),int(bbox[i][1])),(int(bbox[i][2]),int(bbox[i][3])),(0, 255, 255), 3)
                multiProcess.append(build_multitracker(model,label[i],probs[i],resQueue))
            for i in range(len(multiProcess)):
                init_rect = [bbox[i][0],bbox[i][1],bbox[i][2]-bbox[i][0],bbox[i][3]-bbox[i][1]]
                multiProcess[i].init(frame, init_rect)
                multiProcess[i].start()
            
            key_frame = frame
        else:
            print('track locally')
            for i in range(len(multiProcess)):
                multiProcess[i].track(frame)

            t= time.time()
            for i in range(len(multiProcess)):
                resDict = resQueue.get()
                resDict['bbox'] = [resDict['bbox'][0],resDict['bbox'][1],resDict['bbox'][0]+resDict['bbox'][2],resDict['bbox'][1]+resDict['bbox'][3]]
                write_txt(dataset, f, resDict['bbox'], resDict['label'], resDict['probs'])

                # bbox = list(map(int, resDict['bbox']))
                # cv2.rectangle(frame, (bbox[0], bbox[1]),(bbox[0]+bbox[2], bbox[1]+bbox[3]),(0, 255, 0), 3)
            print(time.time()-t)

        # cv2.namedWindow('image_demo',0)
        # cv2.resizeWindow('image_demo', 300,400)
        # cv2.imshow('image_demo', frame)
        # cv2.waitKey(1)

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()

