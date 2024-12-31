
import torch
#from network import *
import network.mvod_basenet
from network.predictor import Predictor 
from datasets.vid_dataset import ImagenetDataset
from config import config
from utils import box_utils, measurements
from utils.misc import str2bool, Timer
import argparse
import pathlib
import numpy as np
import logging
import sys

from ecci_sdk import Client
import threading
from threading import Thread

parser = argparse.ArgumentParser(description="MVOD Evaluation on VID dataset")
parser.add_argument('--net', default="basenet",help="The network architecture, it should be of basenet")
parser.add_argument("--trained_model",default = "", type=str)
parser.add_argument("--dataset", type=str, default="",help="The root directory of the VID dataset .")
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument('--width_mult', default=1.0, type=float,help='Width Multiplifier for network')
parser.add_argument('--nms_method', default="hard")
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

if __name__ == '__main__':
    ecci_client = Client()
    mqtt_thread = threading.Thread(target=ecci_client.initialize)
    mqtt_thread.start()
    ecci_client.wait_for_ready()
    print('cloud start --------')
    timer = Timer()
    config = config
    num_classes = 31
    if args.net == 'basenet':
        pred_enc = network.mvod_basenet.MobileNetV2(num_classes=num_classes, alpha = args.width_mult)
        pred_dec = network.mvod_basenet.SSD(num_classes=num_classes, alpha = args.width_mult, is_test=True, config= config)
        net = network.mvod_basenet.MobileVOD(pred_enc, pred_dec)
    else:
        sys.exit(1)

    timer.start("Load Model")
    net.load_state_dict(
        torch.load(args.trained_model,
            map_location=lambda storage, loc: storage))
    net = net.to(device)
    print(f'It took {timer.end("Load Model")} seconds to load the model.')
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=args.nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=200,
                          sigma=0.5,
                          device=device)

    results = []
    while True :
        # print("process image", i)
        timer.start("Load Image")
        ##############################################################
        edge_data = ecci_client.get_sub_data_payload_queue().get()##########feaature map from edge#######
        image = edge_data["frame"]
        print("#############recieve feature map from edge",image)
        ##############################################################
        print("Load Image: {:4f} seconds.".format(timer.end("Load Image")))
        timer.start("Predict")
        boxes, labels, probs = predictor.predict(image)
        ##########################################################
        # print(boxes,labels,probs)                  #########result
        payload = {"type":"data","contents":{"bbox":boxes.numpy().tolist(),"label":labels.numpy().tolist(),"probs":probs.numpy().tolist()}}
        print("###########send boxes to edge",payload)
        ecci_client.publish(payload, "edge")
        ##########################################################