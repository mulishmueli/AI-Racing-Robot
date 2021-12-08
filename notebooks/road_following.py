import torchvision
import torch
from torch2trt import TRTModule
from jetracer.nvidia_racecar import NvidiaRacecar
from jetcam.csi_camera import CSICamera
from utils import preprocess
import numpy as np
from torch2trt import torch2trt


CATEGORIES = ['apex']
device = torch.device('cuda')
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2 * len(CATEGORIES))
model = model.cuda().eval().half()
model.load_state_dict(torch.load('road_following_model.pth'))
data = torch.zeros((1, 3, 224, 224)).cuda().half()
model_trt = torch2trt(model, [data], fp16_mode=True)
torch.save(model_trt.state_dict(), 'road_following_model_trt.pth')
model_trt = TRTModule()
model_trt.load_state_dict(torch.load('road_following_model_trt.pth'))
car = NvidiaRacecar()
camera = CSICamera(width=224, height=224, capture_fps=65)
STEERING_GAIN = 0.75
STEERING_BIAS = 0.00
car.throttle = 0.15

while True:
    image = camera.read()
    image = preprocess(image).half()
    output = model_trt(image).detach().cpu().numpy().flatten()
    x = float(output[0])
    car.steering = x * STEERING_GAIN + STEERING_BIAS
