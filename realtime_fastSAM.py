#!/usr/bin/env python3

#https://www.youtube.com/watch?v=SslzS0AsiAw
import os
from pprint import pprint
import sys

print("Curr Dir : ", os.getcwd())
print("Dir Items :")
pprint(os.listdir())
if not os.path.exists("FastSAM"):
    os.system("git clone https://github.com/CASIA-IVA-Lab/FastSAM.git")

sys.path.append("./FastSAM")
from fastsam import FastSAM, FastSAMPrompt
import torch
import numpy as np
import cv2
import time



FASTSAM_CHECKPT = "weights/FastSAM.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")

model = FastSAM(FASTSAM_CHECKPT)


cap = cv2.VideoCapture(0) 
# cap = cv2.VideoCapture("./data/test_vid.mp4") 
while (True):
    ret, frame = cap.read()
    start = time.perf_counter()

    everything_results = model(
        source = frame,
        device = DEVICE,
        retina_masks = True,
        imgsz=1024,
        conf=0.4,
        iou=0.9
    )
    print(f"Masks shape: {everything_results[0].masks.shape}")
    print(f"Boxes shape: {everything_results[0].boxes.shape}")
    # print(f"xyxy shape: {everything_results[0].boxes[0].xyxy.cpu().numpy()}")

    for box in everything_results[0].boxes:
        box=box.xyxy.numpy()[0]
        # print(box)
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]) , int(box[3])), (0,255,0), 2)
    prompt_process = FastSAMPrompt(frame, everything_results, device=DEVICE)
    ann = prompt_process.everything_prompt()

    end = time.perf_counter()
    total_time = end-start
    fps = 1/total_time

    img = prompt_process.plot_to_result( annotations=ann)

    cv2.putText(img, f"FPS: {int(fps)}", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0) ,2)
    print(f"""
        height : {img.shape[0]}
        width : {img.shape[1]} """)
    cv2.imshow('img', img)
    # cv2.imshow('frame', frame)

    # cv2.imwrite(f"./test_{time.time()}.png",img)

    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

# After the loop release the cap object 
cap.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 




