import numpy as np
import cv2
import sys
from PIL import Image
import face_detection as fd
import cnn1 as cnn
from keras.preprocessing.image import img_to_array

from PIL import ImageTk
import tkinter as tki
import threading
import datetime
import imutils
import time
import cv2
import os


class OneCam:
    def __init__(self, vs):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        self.vs = vs
        self.frame = None
        self.thread = None
        self.stopEvent = None
                                                                             
		# initialize the root window and image panel
        self.root = tki.Tk()
        self.root.geometry("340x500") #You want the size of the app to be 500x500
        self.root.resizable(0, 0) 
        self.panelA = None
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy','Neutral', 'Sad', 'Surprise']
        self.model = cnn.load_model("ck.h5")
        self.model._make_predict_function()
        print("\n\nmodel loaded\n\n")
	
        self.resultLabel = []
        for i in range(0,len(emotions)):
            w = tki.Label(self.root, text=emotions[i])
            w.pack()	
            w.place(x=20, y = 300 + 25*i)
            self.resultLabel.append(tki.Label(self.root, text="."))
            self.resultLabel[i].pack()
            self.resultLabel[i].place(x=100, y = 300+25*i)

	    #  btn = tki.Button(self.root, text="Snapshot!",command=self.takeSnapshot)
            # btn.pack(side="bottom", fill="both", expand="yes", padx=10,pady=10)
                                 
	    # start a thread that constantly pools the video sensor for
	    # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()
                                                                         
		# set a callback to handle when the window is closed
        self.root.wm_title("MHP FER oneCam")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)
		
    def videoLoop(self):
        try:
			# keep looping over frames until we are instructed to stop
            i = 0 
            while not self.stopEvent.is_set():
		# grab the frame from the video stream and resize it to
		# have a maximum width of 300 pixels
                flag, self.frame = vs.read()
      
                if i==0:
                    i += 2
                    faceCoordinates=None
                else:
				
                    try:
                        faceCoordinates = fd.face_detect(self.frame)      
                        startX = faceCoordinates[0]
                        startY = faceCoordinates[1]
                        endX = faceCoordinates[2]
                        endY = faceCoordinates[3]
                        if startX is not None:
                            image = fd.draw_rect(self.frame, startX, startY, endX, endY)
                            face_img = fd.face_crop(self.frame,faceCoordinates,face_shape=(128,128)) 
                            im = img_to_array(face_img)
                            im = np.expand_dims(im,axis=0)
                            result = self.model.predict(im)
                            for i in range(0,7):
                                self.resultLabel[i].config(text=str(round(result[0][i],4)))
                    except:
                        pass
				
                self.frame = imutils.resize(self.frame, width=300)
                self.frame = cv2.flip(self.frame, 1)
		# OpenCV represents images in BGR order; however PIL
		# represents images in RGB order, so we need to swap
		# the channels, then convert to PIL and ImageTk format
                image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)
				
		
		# if the panel is not None, we need to initialize it
                if self.panelA is None:
                    self.panelA = tki.Label(image=image)
                    self.panelA.image = image
                    self.panelA.pack(side="left", padx=10, pady=10)
                    self.panelA.place(x=20, y=20)
		# otherwise, simply update the panel
                else:
                    self.panelA.configure(image=image)
                    self.panelA.image = image
 
        except RuntimeError:
            print("[INFO] caught a RuntimeError")
	
    def onClose(self):
		# set the stop event, cleanup the camera, and allow the rest of
		# the quit process to continue
        print("[INFO] closing...")
        self.root.destroy()

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] warming up camera...")
try:
	vs = cv2.VideoCapture(1)
except:
	vs = cv2.VideoCapture(0)

time.sleep(2.0) 
 

# start the app
pba = OneCam(vs)
pba.root.mainloop()
