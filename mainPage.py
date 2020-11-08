from tkinter import *
from tkinter import filedialog
from PIL import ImageTk,Image  
import cv2
import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from model import UnetModel
import matplotlib.pyplot as plt

class UnetGUI():
    def __init__(self,main):    
        main.resizable(0, 0)

        self.trainVidPath = ""
        self.testVidPath = ""
        self.model = None
        self.modelPath = ""


        self.currentVidFrame = None


        self.trainingSet = []
        self.images = []
        self.masks  = []

        self.currentColor = (0,250,0)
        self.points = []


        self.Epochs = IntVar()
        self.BatchSize = IntVar()
        self.TrainTestSplit = DoubleVar()
        self.Threshold = DoubleVar()


        #self.smallph = ImageTk.PhotoImage(Image.open("256ph.jpg"))
        #self.largeph = ImageTk.PhotoImage(Image.open("512ph.jpg"))
        smallPlaceHolder = ImageTk.PhotoImage(Image.fromarray(np.zeros((256,256,3),np.uint8)+255))
        largePlaceHolder = ImageTk.PhotoImage(Image.fromarray(np.zeros((512,512,3),np.uint8)+255))


        #########################MainPanles

        self.testVid = Label(main, image = largePlaceHolder)
        self.testVid.grid(row=1,column=0,columnspan=4,rowspan=7,sticky = "nsew")

        self.ground = Label(main, image = smallPlaceHolder)
        self.ground.grid(row=1,column=4,columnspan=2,rowspan=2,sticky = "nsew")

        self.mask = Label(main, image = smallPlaceHolder)
        self.mask.grid(row=1,column=6,columnspan=2,rowspan=2,sticky = "nsew")


        #########################TestLayout

        self.vidSelectButton = Button(main, text="Open Video for Training Model", command=self.selectTrainingVideo)
        self.vidSelectButton.grid(row=0, column=4,sticky = "nsew")

        self.newTrainButton = Button(main, text="Add example",command=self.CreateTrainingItem)
        self.newTrainButton.grid(row=0, column=5,sticky = "nsew")

        self.TrainingItems = StringVar(main, value="0 Training Items")
        self.TrainingItemsCounter = Label(main, text=self.TrainingItems.get())
        self.TrainingItemsCounter.grid(row=0, column=6,columnspan=2,sticky = "nsew")


        #########################TrainLayout


        self.suggestParameters = Button(main, text="Suggest HyperParameters",command=self.SuggestHP)
        self.suggestParameters.grid(row=3, column=4, columnspan=2,sticky = "nsew")


        self.setEpoch = Label(main,text = "Epochs")
        self.setEpoch.grid(row=4,column=4, columnspan=1)

        self.getEpoch = Scale(main, from_=0, to=100, resolution=10,orient=HORIZONTAL, variable=self.Epochs)
        self.getEpoch.grid(row=4, column=5, columnspan=1)
        
        self.setBatch = Label(main,text = "Batch Size")
        self.setBatch.grid(row=5,column=4,columnspan=1)

        self.getBatch = Scale(main, from_=0, to=10,orient=HORIZONTAL,variable = self.BatchSize)
        self.getBatch.grid(row=5, column=5, columnspan=1)

        self.setTTS = Label(main,text = "Test:Train Split")
        self.setTTS.grid(row=6,column=4,columnspan=1)

        self.getTTS = Scale(main, from_=0, to=100,orient=HORIZONTAL, variable=self.TrainTestSplit)
        self.getTTS.grid(row=6, column=5, columnspan=1)

        self.setThreshold = Label(main,text = "Threshold")
        self.setThreshold.grid(row=7,column=4,columnspan=1)

        self.getThreshold = Scale(main, from_=0, to=100,orient=HORIZONTAL, variable=self.Threshold)
        self.getThreshold.grid(row=7, column=5, columnspan=1)

        self.epochNum = StringVar(main, value="Epoch 0/0")
        self.epochCounter = Label(main, text=self.epochNum.get())
        self.epochCounter.grid(row=8, column=4)
        
        self.randomFramea = Button(main, text="Train On Batch",command=self.TrainModel)
        self.randomFramea.grid(row=8, column=5)

    

        #########################ModelLayout

        self.createModelButton = Button(main, text="Open Model", command=self.selectModel)
        self.createModelButton.grid(row=0,column=0,sticky = "nsew")


        self.modelSelectButton = Button(main, text="Create New Model", command=self.createModel)
        self.modelSelectButton.grid(row=0, column=1,sticky = "nsew")


        self.modelSelectButton = Label(main, text="Current Model:")
        self.modelSelectButton.grid(row=0, column=2,sticky = "nsew")


        self.currentModel = StringVar(main, value=".../ModelPath.h")
        self.currentModelPath = Message(main,text=self.currentModel.get(),width=150)
        self.currentModelPath.grid(row=0, column=3,sticky = "w")


        #########################HyperParameterLayout


        self.modelSelectButton = Button(main, text="Open Video to Analyze", command=self.selectTestVideo)
        self.modelSelectButton.grid(row=8, column=0,sticky = "nsew")
        
        self.TestVideo = StringVar(main, value=".../VideoPath.h")
        self.TestVideoPath = Message(main,text=self.TestVideo.get(),width=150)
        self.TestVideoPath.grid(row=8,column=1)

        self.vidPrevButton = Button(main, text="Random Frame",command=self.getRandomTestFrame)
        self.vidPrevButton.grid(row=8, column=2,sticky = "nsew")

        self.vidPrevButton = Button(main, text="Test Model",command=self.TestModel)
        self.vidPrevButton.grid(row=8, column=3,sticky = "nsew")
        
        #########################HyperParameterLayout

        self.vidPrevButton = Button(main, text="Run Tracking",command=self.RunTracking)
        self.vidPrevButton.grid(row=8, column=6,columnspan=2,sticky = "nsew")


        #########################trackingLayout


        self.a = Button(main, text="Save Dataset",command=self.SaveDataset)
        self.a.grid(row=3, column=6, columnspan=2,sticky = "nsew")


        self.b = Label(main,text = "Video Height:")
        self.b.grid(row=4,column=6)

        self.c = Entry(main, width=4)
        self.c.grid(row=4, column=7)
        
        self.d = Label(main,text ="Video Width:")
        self.d.grid(row=5,column=6)

        self.e = Entry(main, width=4)
        self.e.grid(row=5, column=7)

        self.csvName = StringVar(main, value="Epoch 0/0")
        self.h = Entry(main, width=4, text=self.csvName.get())
        self.h.grid(row=6,column=6)

        self.catagoryName = StringVar(main, value="Epoch 0/0")
        self.i =Entry(main, width=4, text=self.catagoryName.get())
        self.i.grid(row=6, column=7)

        self.f = Label(main,text = "Display Tracking")
        self.f.grid(row=7,column=6)

        self.g = Checkbutton(root,)
        self.g.grid(row=7, column=7)



    

    def TrainModel(self):
    
        self.images = []
        self.masks = []

        while(len(self.trainingSet) != 0):
            self.images.append(self.trainingSet[0]/255.0)
            self.trainingSet.pop(0)
            self.masks.append(self.trainingSet[0]/255.0)
            self.trainingSet.pop(0)

        self.images = np.array(self.images)
        self.masks  = np.array(self.masks)

        x_train,x_test,y_train, y_test = train_test_split(self.images,self.masks, test_size =self.TrainTestSplit.get()/100)

        for i in range(0,self.Epochs.get()):
            history = self.model.fit(x_train,y_train, validation_data=(x_test, y_test),epochs = 1 ,batch_size = self.BatchSize.get()) 
            self.epochNum.set("Epoch " + str(i + 1) + "/" + str(self.Epochs.get()))
            self.epochCounter.config(text=self.epochNum.get())
            self.epochCounter.text = self.epochNum.get()
            self.epochCounter.update_idletasks()
            self.maskMainPanel()
        
        self.model.save(self.modelPath)

    def maskMainPanel(self):
        frame = self.currentVidFrame

        im1 = cv2.resize(frame,(256,256))
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

        temp = []
        temp.append(im1/255.0)
        ground = np.array(temp)
        im2 = self.model.predict(ground)

        ground = ((cv2.resize(ground[0],(512,512))* 255).astype(np.uint8))
        mask =((cv2.resize(im2[0],(512,512))* 255).astype(np.uint8))

        concat = cv2.addWeighted(ground, .5, mask, .5, 0.0)

        concat = Image.fromarray(concat)
        concat = ImageTk.PhotoImage(concat)

        self.testVid.config(image=concat)
        self.testVid.image = concat
        self.testVid.update_idletasks()
        self.testVid.update()

    def TestModel(self):
        i = 0
        while(i < 25):
            cap = cv2.VideoCapture(self.testVidPath)
            randomFrameNum =np.random.randint(1,cap.get(7))
            cap.set(1, randomFrameNum)
            ret, frame = cap.read()
            im1 = cv2.resize(frame,(256,256))
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

            temp = []
            temp.append(im1/255.0)
            ground = np.array(temp)
            im2 = self.model.predict(ground)

            ground = ((cv2.resize(ground[0],(512,512))* 255).astype(np.uint8))
            mask = ((cv2.resize(im2[0],(512,512))* 255).astype(np.uint8))
            mask2 = self.maskToPoint(mask,ground)
            concat = cv2.addWeighted(ground, .7, mask2, .3, 0.0)

            concat = Image.fromarray(concat)
            concat = ImageTk.PhotoImage(concat)

            self.testVid.config(image=concat)
            self.testVid.image = concat
            self.testVid.update_idletasks()
            self.testVid.update()
            i += 1

    def selectTrainingVideo(self):
        path = filedialog.askopenfilename()
        self.trainVidPath = path

    def selectModel(self):
        self.modelPath = filedialog.askopenfilename()
        self.model = keras.models.load_model(self.modelPath)
        self.currentModel.set(str(self.modelPath[-10:]))
        self.currentModelPath.config(text=self.currentModel.get())
        self.currentModelPath.text = self.currentModel.get()
        self.currentModelPath.update_idletasks()
    
    def createModel(self):

        self.modelPath = filedialog.asksaveasfilename(defaultextension=".h5")
        self.model = UnetModel().getModel()
        self.model.save(self.modelPath)
        self.model = keras.models.load_model(self.modelPath)
        self.currentModel.set(".../"+str(self.modelPath[-8:]))
        self.currentModelPath.config(text=self.currentModel.get())
        self.currentModelPath.text = self.currentModel.get()
        self.currentModelPath.update_idletasks()

    def selectTestVideo(self):
        self.testVidPath = filedialog.askopenfilename()
        self.TestVideo.set(".../"+str(self.testVidPath[-8:]))
        self.TestVideoPath.config(text=self.TestVideo.get())
        self.TestVideoPath.text = self.TestVideo.get()
        self.TestVideoPath.update_idletasks()
        self.getRandomTestFrame()

    def getRandomTestFrame(self):
        cap = cv2.VideoCapture(self.testVidPath)
        randomFrameNum =np.random.randint(1,cap.get(7))
        cap.set(1, randomFrameNum)
        ret, frame = cap.read() 
        self.currentVidFrame = frame

        frame = cv2.resize(frame,(512, 512))
        framePP = cv2.resize(frame,(256, 256))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(frame)
        self.testVid.config(image=frame)
        self.testVid.image = frame
        self.testVid.update_idletasks()

    def drawPoint(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.currentGround,(x,y),1,self.currentColor,-1)
            point =[]
            point.append(x)
            point.append(y)
            self.points.append(point)

    def CreateTrainingItem(self):
        self.points.clear()
        cap = cv2.VideoCapture(self.trainVidPath)
        randomFrameNum =np.random.randint(1,cap.get(7))
        cap.set(1, randomFrameNum)
        ret, frame= cap.read()

        self.currentGround = frame
        mask=np.zeros((frame.shape[0],frame.shape[1],3),np.uint8)

        windowName = "| <esc> : submit |"
        cv2.namedWindow(windowName)
        cv2.setMouseCallback(windowName,self.drawPoint)

        while(True):
            cv2.imshow(windowName,self.currentGround)

            if cv2.waitKey(20) & 0xFF == 27:
                cv2.destroyWindow(windowName)
                break

        holder = []
        holder.append(self.points)

        poly = np.array(holder, np.int32)
        mask = cv2.fillPoly(mask, [poly], (255, 255, 255))
        frame = cv2.resize(frame,(256, 256))
        mask = cv2.resize(mask,(256, 256))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)


        self.trainingSet.append(frame)
        self.trainingSet.append(mask)

        frame = Image.fromarray(frame)
        mask = Image.fromarray(mask)

        frame = ImageTk.PhotoImage(frame)
        mask = ImageTk.PhotoImage(mask)


        self.ground.config(image=frame)
        self.mask.config(image=mask)

        self.ground.image = frame
        self.mask.image = mask

        self.TrainingItems.set(str(int(len(self.trainingSet)/2)) + " Training Items")
        self.TrainingItemsCounter.config(text=self.TrainingItems.get())
        self.TrainingItemsCounter.text = self.TrainingItems.get()
        self.TrainingItemsCounter.update_idletasks()
        self.ground.update_idletasks()
        self.mask.update_idletasks()

    def maskToPoint(self, image, ground):

        imGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, imThresh = cv2.threshold(imGray, 10, 255, 0)
        im, contours, hierarchy = cv2.findContours(imThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        largestContour = max(contours, key = cv2.contourArea)

        M = cv2.moments(largestContour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        x = 75
        image = cv2.circle(ground, (cX, cY), 2, (100, 255, 0), -1)
        image = cv2.rectangle(ground,(cX-x, cY-x),(cX+x, cY+x),(100,255,0),5)

        return image

    def RunTracking(self):
        cap = cv2.VideoCapture(self.testVidPath)
        randomFrameNum =np.random.randint(1,cap.get(7))
        cap.set(1, randomFrameNum)
        trackingData = []
        trackingData.append(["frame", "x", "y"])
        i = 1
        numFrames = cap.get(7)
        while(i < 1000):
            ret, frame = cap.read()
            ground = cv2.resize(frame,(256,256))
            ground = cv2.cvtColor(ground, cv2.COLOR_BGR2GRAY)
            temp = []
            temp.append(ground/255.0)
            ground = np.array(temp)
            mask = self.model.predict(ground)
            mask = ((cv2.resize(mask[0],(100,100))* 255).astype(np.uint8))
            ret, imThresh = cv2.threshold(imGray, 10, 255, 0)
            im, contours, hierarchy = cv2.findContours(imThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            largestContour = max(contours, key = cv2.contourArea)
            M = cv2.moments(largestContour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            trackingData.append([i,cX, cY])
            i+=1

        with open('protagonist.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(trackingData)
        print("done")
    def SaveDataset(self):
        print("datasave")

    def SuggestHP(self):
        print("hp")





root = Tk()
root.title("SegTrack")
UnetGUI(root)
root.mainloop()
