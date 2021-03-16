
#NOTE the weight and configure file
#nhap input
import cv2
from imutils.video import VideoStream
# img = cv2.imread("stop.png")
# cap = cv2.VideoCapture()
cap= VideoStream(usePiCamera=True,resolution=(1920,1280),framerate=30).start()
# cap.set(3, 640)
# cap.set(4, 480)



clasNames = []
#coco.names file chua cac object can detection
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)
#dua 2 file .pbtxt va .pb vao cung thu muc

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightPath, configPath)
#parameter default
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()

    #confidence Threshold neu < 50% thi ignore object
    classIds, confs, bbox = net.detect(img,confThreshold=float(0.5))
    #neu du doan >50% thi print classID va boundi ng box xung quanh object
    print(classIds, bbox)
    if len(classIds) != 0:
        for classIds, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img, box, color=(0, 255, 0),thickness=2)
            cv2.putText(img, classNames[classIds-1].upper(), (box[0],box[1]+30),
                        cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),thickness=2)
    cv2.imshow('Output', img)
    cv2.waitKey(1)