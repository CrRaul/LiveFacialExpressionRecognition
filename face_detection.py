
import numpy as np
import cv2

emotions = ['Angry', 'Disgust', 'Fear', 'Happy','Neutral', 'Sad', 'Surprise']

def face_detect(im):

    net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
  
    
    image = im
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
    	(300, 300), (104.0, 177.0, 123.0))
    
    net.setInput(blob)
    detections = net.forward()
    
    # loop over the detections
    for i in range(0, detections.shape[2]):
    	# extract the confidence (i.e., probability) associated with the
    	# prediction
    	confidence = detections[0, 0, i, 2]
    
    	# filter out weak detections by ensuring the `confidence` is
    	# greater than the minimum confidence
    	if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")  
            faceCoordinates = box.astype("int")
            
            #text = "{:.2f}%".format(confidence * 100)
            
    return faceCoordinates
    

    
def draw_rect(image, startX, startY, endX, endY):
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
     
    """+"  Acc:"+str(acc)"""
def face_crop(image, faceCoordinates, face_shape):
    face = crop_face(image, faceCoordinates)            
    face_scaled = cv2.resize(face, face_shape)
    face_gray = cv2.cvtColor(face_scaled, cv2.COLOR_BGR2GRAY)
    return face_gray

def crop_face(img, faceCoordinates):
    return img[faceCoordinates[1]:faceCoordinates[3], faceCoordinates[0]:faceCoordinates[2]]
