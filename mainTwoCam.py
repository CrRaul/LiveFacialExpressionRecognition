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


class TwoCam:
    def __init__(self, vs, vs1):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        self.vs = vs
        self.vs1 = vs1
        self.frame = None
        self.frame1 = None
        self.thread = None
        self.stopEvent = None
                                                                             
		# initialize the root window and image panel
        self.root = tki.Tk()
        self.root.geometry("660x500") #You want the size of the app to be 500x500
        self.root.resizable(0, 0) 
        self.panelA = None
        self.panelB = None
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy','Neutral', 'Sad', 'Surprise']
        self.model = cnn.load_model("ck.h5")
        self.model._make_predict_function()
	
	##########################
	# create the text label
        w = tki.Label(self.root,text = "Camera 1")
        w.pack()
        w.place(x = 40, y = 270)
        
        w = tki.Label(self.root,text = "Camera 2")
        w.pack()
        w.place(x = 270, y = 270)
	
        w = tki.Label(self.root,text = "1 / 2")
        w.pack()
        w.place(x = 500, y = 270)

        self.resultLabel = []
        self.resultLabel1 = []
        self.cam12 = []
        for i in range(0,len(emotions)):
            w = tki.Label(self.root, text=emotions[i])
            w.pack()	
            w.place(x=20, y = 300 + 25*i)
            self.resultLabel.append(tki.Label(self.root, text=""))
            self.resultLabel[i].pack()
            self.resultLabel[i].place(x=100, y = 300+25*i)
            
            w = tki.Label(self.root, text=emotions[i])
            w.pack()	
            w.place(x=250, y = 300 + 25*i)
            self.resultLabel1.append(tki.Label(self.root, text=""))
            self.resultLabel1[i].pack()
            self.resultLabel1[i].place(x=330, y = 300+25*i)
	    
            w = tki.Label(self.root, text=emotions[i])
            w.pack()	
            w.place(x=480, y = 300 + 25*i)
            self.cam12.append(tki.Label(self.root, text=""))
            self.cam12[i].pack()
            self.cam12[i].place(x=560, y = 300+25*i)
         ############################

	    #  btn = tki.Button(self.root, text="Snapshot!",command=self.takeSnapshot)
            # btn.pack(side="bottom", fill="both", expand="yes", padx=10,pady=10)
                                 
	# start a thread that constantly pools the video sensor for
	# the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()
                                                                         
        # set a callback to handle when the window is closed
        self.root.wm_title("MHP FER twoCam")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)
		
    def videoLoop(self):
        try:
			# keep looping over frames until we are instructed to stop
            i = 0 
            while not self.stopEvent.is_set():
		# grab the frame from the video stream and resize it to
		# have a maximum width of 300 pixels
                flag, self.frame = vs.read()
                flag, self.frame1 = vs1.read()      

                if i==0:
                    i += 2
                    faceCoordinates = None
                    faceCoordinates1 = None
                else:
                    try:
                        faceCoordinates = fd.face_detect(self.frame)      
                        startX = faceCoordinates[0]
                        startY = faceCoordinates[1]
                        endX = faceCoordinates[2]
                        endY = faceCoordinates[3]
                         	    
                        faceCoordinates1 = fd.face_detect(self.frame1)      
                        startX1 = faceCoordinates1[0]
                        startY1 = faceCoordinates1[1]
                        endX1 = faceCoordinates1[2]
                        endY1 = faceCoordinates1[3]
                         	    
                        if startX is not None:
                            image = fd.draw_rect(self.frame, startX, startY, endX, endY)
                            face_img = fd.face_crop(self.frame,faceCoordinates,face_shape=(128,128)) 
                            im = img_to_array(face_img)
                            im = np.expand_dims(im,axis=0)
                            result = self.model.predict(im)

                            for i in range(0,7):
                                self.resultLabel[i].config(text=str(round(result[0][i],4)))
                        
                        if startX1 is not None:
                            image1 = fd.draw_rect(self.frame1, startX1, startY1, endX1, endY1)
                            face_img1 = fd.face_crop(self.frame1,faceCoordinates,face_shape=(128,128)) 
                            im1 = img_to_array(face_img1)
                            im1 = np.expand_dims(im1,axis=0)
                            result1 = self.model.predict(im1)

                            for i in range(0,7):
                               self.resultLabel1[i].config(text=str(round(result1[0][i],4)))
			
                        if startX is not None and startX1 is not None:
                            for i in range(0,7):
                               self.cam12[i].config(text=str((round(result[0][i],4)+round(result1[0][i],4))/2))
                    except:
                        pass   
		
                self.frame = imutils.resize(self.frame, width=300)
                self.frame = cv2.flip(self.frame, 1)
                
                self.frame1 = imutils.resize(self.frame1, width=300)
                self.frame1 = cv2.flip(self.frame1 , 1)
		# OpenCV represents images in BGR order; however PIL
		# represents images in RGB order, so we need to swap
		# the channels, then convert to PIL and ImageTk format
                image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)
				
                image1 = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2RGB)
                image1 = Image.fromarray(image1)
                image1 = ImageTk.PhotoImage(image1)
		
		# if the panel is not None, we need to initialize it
                if self.panelA is None:
                    self.panelA = tki.Label(image=image)
                    self.panelA.image = image
                    self.panelA.pack(side="left", padx=10, pady=10)
                    self.panelA.place(x=20, y=20)
		    
                    self.panelB = tki.Label(image=image1)
                    self.panelB.image = image1
                    self.panelB.pack(side="right", padx=10, pady=10)
                    self.panelB.place(x=340, y=20)
		# otherwise, simply update the panel
                else:
                    self.panelA.configure(image=image)
                    self.panelA.image = image
                    self.panelB.configure(image=image1)
                    self.panelB.image = image1
        except RuntimeError:
            print("[INFO] caught a RuntimeError")
	
    def onClose(self):
		# set the stop event, cleanup the camera, and allow the rest of
		# the quit process to continue
        print("[INFO] closing...")
        self.root.destroy()
        sys.exit()
# initialize the video stream and allow the camera sensor to warmup
print("[INFO] warming up camera...")
vs = cv2.VideoCapture(0)
vs1 = cv2.VideoCapture(1)
time.sleep(2.0) 
 

# start the app
pba = TwoCam(vs, vs1)
pba.root.mainloop()
