import numpy as np
import math
import cv2
import os
import RSAD.static.Dataset.Model.Number_plate_detection as npd
from scipy.spatial import distance as dist
from collections import OrderedDict
from RSAD.static.Dataset.Model.CentroidTracker import CentroidTracker
#import threading


class detector:
    stopline = False
    midline = False
    parkarea = False
    slx = 0
    sly = 300
    slx1 = 900
    sly1 = 300

    mlx = 450
    mly = 0
    mlx1 = 450
    mly1 = 600


    def __init__(self,video):
        print(cv2.getBuildInformation())
#        self.outputFrame = None
#        self.lock = threading.Lock()

        self.dire = {}
        self.ct = CentroidTracker()
        self.args = {'yolo':'RSAD/static/Dataset/Model/yolo_v3','confidence':0.5,'threshold':0.3}

        self.labelsPath = os.path.sep.join([self.args["yolo"], "coco.names"])
        self.LABELS = open(self.labelsPath).read().strip().split("\n")

        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),dtype="uint8")

        self.weightsPath = os.path.sep.join([self.args["yolo"], "yolov3.weights"])
        self.configPath = os.path.sep.join([self.args["yolo"], "yolov3.cfg"])

        print("[INFO] loading YOLO from disk...")
        self.net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]


        self.vs = cv2.VideoCapture(video)


        try:
            self.prop = cv2.CAP_PROP_FRAME_COUNT

            total = int(self.vs.get(self.prop))
            print("[INFO] {} total frames in video".format(total))


        except:
        	print("[INFO] could not determine # of frames in video")
        	print("[INFO] no approx. completion time can be provided")
        	total = -1
#        t = threading.Thread(target=self.main(), args=(32,))
#        t.deaemon = True
#        t.start()
    def main(self):
        (W, H) = (None, None)
        detected =[]

        while True:

            # print(self.y)
        	# read the next frame from the file
            (grabbed, frame) = self.vs.read()
            # frame = cv2.UMat(frame)
            frame = cv2.resize(frame,(900,600))

            _,temp = self.vs.read()
            temp = cv2.resize(temp,(900,600))

            if self.stopline:
                cv2.line(frame, (self.slx, self.sly), (self.slx1, self.sly1), (0, 0, 255), 2)
            if self.midline :
                cv2.line(frame, (self.mlx, self.mly), (self.mlx1, self.mly1), (0, 0, 255), 2)

            if not grabbed:
                break

        	# if the frame dimensions are empty, grab them
            if W is None or H is None:
                (H, W) = temp.shape[:2]

            self.blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
            # self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

            self.net.setInput(self.blob)


            self.layerOutputs = self.net.forward(self.ln)


            self.boxes = []
            self.confidences = []
            self.classIDs = []

            for self.output in self.layerOutputs:

                for self.detection in self.output:

                    self.scores = self.detection[5:]
                    self.classID = np.argmax(self.scores)
                    self.confidence = self.scores[self.classID]

                    if self.confidence > self.args["confidence"]:

                        self.box=self.detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = self.box.astype("int")

                        x=int(centerX - (width / 2))
                        y=int(centerY - (height / 2))

                        self.boxes.append([x, y, int(width), int(height)])
                        self.confidences.append(float(self.confidence))
                        self.classIDs.append(self.classID)

            # apply non-maxima suppression to suppress weak, overlapping
        	# bounding boxes
            self.idxs = cv2.dnn.NMSBoxes(self.boxes, self.confidences, self.args["confidence"],self.args["threshold"])
            self.rect=[]

            if len(self.idxs) > 0:
        		# loop over the indexes we are keeping
                for i in self.idxs.flatten():
        			# extract the bounding box coordinates
                    if self.LABELS[self.classIDs[i]] in ['car','truck','bus','motorbike']:
                        (x, y) = (self.boxes[i][0], self.boxes[i][1])
                        (w, h) = (self.boxes[i][2], self.boxes[i][3])
                        self.rect.append([x,y,x+w,y+h])
                        color= [int(c) for c in self.COLORS[self.classIDs[i]]]

                        cv2.putText(frame, self.LABELS[self.classIDs[i]], (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            obj,did = self.ct.update(self.rect)
    #        print(obj,did)
            for (objectID, centroid) in obj.items():
        		# draw both the ID of the object and the centroid of the
        		# object on the output frame
                text = "ID {}".format(objectID)

                cx ,cy,x,y,x1,y1=centroid
                D = self.direction(objectID, cx, cy, did)
                Cflag = False

                Dflag = False
                if self.stopline == True:
                    Cflag = self.check_crossing(centerX, y1)
                if self.midline == True:
                    Dflag = self.check_side(centerX,y1,D)

                cv2.putText(frame, text, (cx - 10, cy - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, D, (cx - 20, cy + 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)


                if Cflag or Dflag :
                    cv2.rectangle(frame, (x, y), (x1,y1), [0,0,255], 2)
                    if objectID in detected:
                        pass
                    else:
#                        try:
                        npd.detect_LP(temp[int(math.fabs(y)):int(math.fabs(y1)), int(math.fabs(x)):int(math.fabs(x1))],frame[int(math.fabs(y)):int(math.fabs(y1)), int(math.fabs(x)):int(math.fabs(x1))])
                        cv2.imwrite('RSAD/static/Dataset/Output/'+str(objectID)+".png",frame)
                        detected.append(objectID)
#                        except Exception as e:
#                            print(e)
                else:
                    cv2.rectangle(frame, (x, y), (x1,y1), [0,255,0], 2)

            (flag0, encodedImage) = cv2.imencode(".jpg", frame)
            if not flag0:
                continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


    def check_crossing(self,x,y):

        if (y > ((self.sly + self.sly1)/2) and y < ((self.sly + self.sly1)/2)+100 and x > self.slx and x < self.slx1 ):
            return True


    def check_side(self,x,y,D):
        if ( x < ((self.mlx+self.mlx1)/2)  and y > self.mly and y < self.mly1  and D == r"/\\") or (x > ((self.mlx+self.mlx1)/2)  and y > self.mly  and y < self.mly1  and D == r"\\/"):
            return True


    def direction(self,ID,CX,CY,did):
    #    print(did)
    #    print(dire)

        if did !=[]:
            for i in did:
                try:
                    del self.dire[i]
                except :
                    pass

        if self.dire == {}:
            self.dire[ID]=(CX,CY,0,0)
        else:
            ids = list(self.dire.keys())
            if ID in ids:
                X,Y,U,D = self.dire[ID]
                if CY > Y :

                    U=0
                    if D>=3:
                        self.dire[ID]=(CX,CY,U,D)
                        return r"\\/"
                    else:
                        self.dire[ID]=(CX,CY,U,D+1)
                        return "."

                elif CY<Y:

                    D=0
                    if U>=3:
                        self.dire[ID]=(CX,CY,U,D)
                        return r"/\\"
                    else:
                        self.dire[ID]=(CX,CY,U+1,D)
                        return "."

            else:
                self.dire[ID]=(CX,CY,0,0)
                return "."


if __name__ == '__main__':
    d=detector("road.mp4")
    d.main()


			# draw a bounding box rectangle and label on the frame

#            if LABELS[classIDs[i]] in ['car','truck','bus','motorbike']:
#                flag=check_crossing(x + w, y + h)
#                if flag:
#                    cv2.rectangle(frame, (x, y), (x + w, y + h), [0,0,255], 2)
#                    if(os.path.exists('rule_break.png')):
#                        pass
#                    else:
#                        try:
#                            npd.detect_LP(temp[int(math.fabs(y)):int(math.fabs(y))+h, int(math.fabs(x)):int(math.fabs(x))+w],frame[int(math.fabs(y)):int(math.fabs(y))+h, int(math.fabs(x)):int(math.fabs(x))+w])
#
#                        except Exception as e:
#                            print(e)
#                        cv2.imwrite("rule_break.png",frame)
#                else:
#                    cv2.rectangle(frame, (x, y), (x + w, y + h), [0,255,0], 2)
#
#
#            else:
#                color= [int(c) for c in COLORS[classIDs[i]]]
#                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#                text=("{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i]))
#                cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#
#
##    cv2.imshow('object detection', cv2.resize(frame, (300,200)))
#    cv2.imshow('object detection', frame)
#
#    cv2.waitKey(1)


            # check if the video writer is None
#    if writer is None:
#		# initialize our video writer
#        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#        writer = cv2.VideoWriter(args["output"], fourcc, 30,(frame.shape[1], frame.shape[0]), True)
#
#		# some information on processing single frame
#        if total > 0:
#            elap = (end - start)
#            print("[INFO] single frame took {:.4f} seconds".format(elap))
#            print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))
#
#	# write the output frame to disk
#    writer.write(frame)

# release the file pointers
#print("[INFO] cleaning up...")
#writer.release()
#vs.release()
