import cv2
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import pytesseract
import os
import time
# import imutils


def cleanPlate(plate):
    print ("CLEANING PLATE. . .")
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

#    gray = cv2.GaussianBlur(gray,(3,3),0)
	#kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
	#thresh= cv2.dilate(gray, kernel, iterations=1)
#    _, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
#    ret3,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#    cv2.imshow("thre",thresh)
#    cv2.waitKey(0)
#    thresh = gray
#    cv2.imshow("thre",thresh)
#    cv2.waitKey(0)
    contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#    cv2.drawContours(gray,contours,-1,[0,255,0],3)
#
#    cv2.imshow("contour",gray)
#    cv2.waitKey(0)
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        max_cnt = contours[max_index]
        max_cntArea = areas[max_index]
        x,y,w,h = cv2.boundingRect(max_cnt)
#        if not ratioCheck(max_cntArea,w,h):
#            return plate,None

        cleaned_final = thresh[y:y+h, x:x+w]
#        cv2.imshow("Function Test",cleaned_final)
#        cv2.waitKey(0)
        return cleaned_final,[x,y,w,h]

    else:
        return plate,None

def ratioCheck(area, width, height):
	ratio = float(width) / float(height)
	if ratio < 1:
		ratio = 1 / ratio

	aspect = 4.7272
	min = 15*aspect*15  # minimum area
	max = 125*aspect*125  # maximum area

	rmin = 3
	rmax = 6

	if (area < min or area > max) or (ratio < rmin or ratio > rmax):
		return False
	return True

def isMaxWhite(plate):
	avg = np.mean(plate)
	if(avg>=115):
		return True
	else:
 		return True









def recognize():
    img = cv2.imread('test.jpg',cv2.IMREAD_COLOR)

    img = cv2.resize(img, (620,480) )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grey scale
    cv2.imshow('image',gray)

    cv2.waitKey(0)
    gray = cv2.bilateralFilter(gray, 11, 17, 17) #Blur to reduce noise
    cv2.imshow('image',gray)

    cv2.waitKey(0)

    edged = cv2.Canny(gray, 30, 200) #Perform Edge detection
    cv2.imshow('image',edged)

    cv2.waitKey(0)

    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None

    # loop over our contours
    for c in cnts:

        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:

            screenCnt = approx
            break



    if screenCnt is None:

        detected = 0
        print ("No contour detected")
    else:
        detected = 1

    if detected == 1:
        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
    cv2.imshow('image',img)

    cv2.waitKey(0)


    # Masking the part other than the number plate
    mask = np.zeros(gray.shape,np.uint8)
    new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
    new_image = cv2.bitwise_and(img,img,mask=mask)
    cv2.imshow('image',new_image)

    cv2.waitKey(0)
    # Now crop
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx+1, topy:bottomy+1]
    cv2.imshow('image',new_image)

    cv2.waitKey(0)


    #Read the number plate
    text = pytesseract.image_to_string(Cropped, config='--psm 11')
    print("Detected Number is:",text)

    cv2.imshow('image',img)
    cv2.imshow('Cropped',Cropped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()





def detect(image,temp):
    face_cascade = cv2.CascadeClassifier('licence_plate.xml')
#    image = cv2.imread(image)
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(image)
    faces = face_cascade.detectMultiScale(image, 1.2, 5)
    print(faces)
    for (x, y, w, h) in faces:

        #img = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        img = image[y:y+h,x:x+w]
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if(isMaxWhite(img)):
                #img = cv2.bilateralFilter(img, 11, 25, 25)
                clean_plate, rect = cleanPlate(img)
                if rect:
                    x1,y1,w1,h1 = rect
                    #x,y,w,h=x1,y+y1,w1,h1
                    cv2.imshow("Cleaned Plate",clean_plate)
                    cv2.waitKey(0)
                    plate_im = Image.fromarray(clean_plate)

                    text = pytesseract.image_to_string(plate_im, config='-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz --psm 9',lang='eng')
                    print ("Detected Text : ",text)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (x,y-10)
                    fontScale = 1
                    fontColor = (0,255,0)
                    lineType = 2
                    img = cv2.rectangle(temp,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(temp,text,bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

                    cv2.imshow("Detected Plate",temp)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()



#        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#
#        gray = img
#        gray = cv2.bilateralFilter(gray, 11, 17, 17)
#
#
#
#        cv2.imshow("camera",gray)
#        cv2.waitKey(0)
#
#
#        text = pytesseract.image_to_string(gray,config='-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz --psm 9', lang='eng')
#
#        print(text)
#        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, [0,255,0], 2)
#
#
#        cv2.imshow("camera",image)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()



def detect_LP(image,temp):
    labelsPath = os.path.sep.join(['RSAD/static/Dataset/Model/yolo_v3_lic_plt', "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
    weightsPath = os.path.sep.join(['RSAD/static/Dataset/Model/yolo_v3_lic_plt', "yolov3.weights"])
    configPath = os.path.sep.join(['RSAD/static/Dataset/Model/yolo_v3_lic_plt', "yolov3.cfg"])
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    (H, W) = image.shape[:2]

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
    	# loop over each of the detections
        for detection in output:
            scores = detection[5:]

            classID = np.argmax(scores)

            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

    			# use the center (x, y)-coordinates to derive the top and
    			# and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

    			# update our list of bounding box coordinates, confidences,
    			# and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
#    print(boxes)
#    print(classIDs)
    if len(idxs) > 0:
    	# loop over the indexes we are keeping
        for i in idxs.flatten():
#            print(i)
    #        xtract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

    		# draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(temp, (x, y), (x + w, y + h), color, 2)
            #text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            #cv2.putText(temp, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
            img = image[y:y+h,x:x+w]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = img
#            gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)

            text = pytesseract.image_to_string(gray,config='-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz --psm 9', lang='eng')

            cv2.putText(temp, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, [0,255,0], 2)



def google_api():
    from google.cloud import vision
    from google.cloud.vision import types
    import io
    image_uri = '/Thunder/YOLO_object_detection/images/car7.jpg'
    client = vision.ImageAnnotatorClient.from_service_account_file('lp-detection.json')
    with io.open(image_uri,'rb') as image_file:
        content = image_file.read()
        image = vision.types.Image(content=content)
        t = client.text_detection(image=image)

if __name__ == '__main__':
#    recognize()
    detect_LP('11.jpg','11.jpg')
#    google_api()