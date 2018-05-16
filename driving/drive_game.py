#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:06:12 2017
@author: leoara01
Examples:
python3 drive_on_game.py --ip=10.45.64.32 --model=./cnn_14.pkl

References:
https://discuss.pytorch.org/t/how-to-cast-a-tensor-to-another-type/2713/4
https://github.com/pytorch/examples/issues/134
https://github.com/vinhkhuc/PyTorch-Mini-Tutorials/blob/master/5_convolutional_net.py
"""
import argparse
import game_communication
import torch
import torchvision.transforms as transforms
import numpy as np
import scipy.misc
from model import CNNDriver
import time

# Force to see just the first GPU
# https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/
import os

class GameRecord:
    __m_id = 0
    __m_img = 0
    __m_telemetry = []

    def __init__(self, id_record, img, telemetry):
        self.__m_id = id_record
        self.__m_img = img
        self.__m_telemetry = telemetry

    def get_id(self):
        return self.__m_id

    def get_image(self):
        return self.__m_img

    def get_telemetry(self):
        return self.__m_telemetry


# Parser command arguments
# Reference:
# https://www.youtube.com/watch?v=cdblJqEUDNo
parser = argparse.ArgumentParser(description='Drive inside game')
parser.add_argument('--ip', type=str, required=False, default='10.45.64.32', help='Server IP address')
parser.add_argument('--port', type=int, required=False, default=50007, help='Server TCP/IP port')
parser.add_argument('--model', type=str, required=False, default='C:\cnn_18.pkl', help='Trained driver model')
parser.add_argument('--gpu', type=int, required=False, default=0, help='GPU number (-1) for CPU')
parser.add_argument('--top_crop', type=int, required=False, default=126, help='Top crop to avoid horizon')
parser.add_argument('--bottom_crop', type=int, required=False, default=226, help='Bottom crop to avoid front of car')
args = parser.parse_args()


def game_pilot(ip, port, model_path, gpu, crop_start=126, crop_end=226):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Set enviroment variable to set the GPU to use

    # Load model
    # https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytroch
    print("Loading model: %s" % model_path)
    cnn = CNNDriver()
    # Model file trained with gpu need to be remaped on CPU
    if device.type == 'cpu':
        cnn.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        cnn.load_state_dict(torch.load(model_path))
    cnn.eval()
    cnn = cnn.to(device)

    #transformations = transforms.Compose([
        #transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    transformations = transforms.Compose([
        transforms.ToTensor()])

    print(ip)
    print(port)

    comm = game_communication.GameTelemetry(ip, port)
    comm.connect()

    # Run until Crtl-C
    try:
        list_records = []
        degrees = 0
        while True:
            # Sleep for 50ms
            time.sleep(0.05)

            # Get telemetry and image
            telemetry = comm.get_game_data()
            cam_img = comm.get_image()

            # Skip entire record if image is invalid
            if (cam_img is None) or (telemetry is None):
                continue

            start = time.time()
            # Resize image to the format expected by the model
            cam_img_res = (scipy.misc.imresize(np.array(cam_img)[crop_start:crop_end], [66, 200]))
            #cam_img_res = cam_img_res.transpose([2, 0, 1]).astype(np.float32)
            #cam_img_res = np.expand_dims(cam_img_res, axis=0)
            #torch_tensor = torch.from_numpy(cam_img_res).unsqueeze(0)
            #torch_tensor = trfNorm(torch_tensor)
            torch_tensor = transformations(cam_img_res)
            cam_img_res = torch_tensor.unsqueeze(0)
            cam_img_res = cam_img_res.to(device)

            # Get steering angle from model
            degrees = cnn(cam_img_res)

            # Convert Variable to numpy
            degrees = float(degrees.data.cpu().numpy())
            #time.sleep(0.55)
            end = time.time()
            elapsed_seconds = float("%.2f" % (end - start))
            print('Elapsed time:', elapsed_seconds, 'angle:', degrees)


            # Send command to game here...
            commands = [degrees, 0.5]
            comm.send_command(commands)

    except KeyboardInterrupt:
        pass

    # Python main


if __name__ == "__main__":
    # Call function that implement the auto-pilot
    game_pilot(args.ip, args.port, args.model, args.gpu, args.top_crop, args.bottom_crop)