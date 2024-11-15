import numpy as np
import cv2
from keras.models import load_model

################################################
width = 640
height = 480
################################################

cap = cv2.VideoCapture(0)
cap.set(3,width)  #3 for width
cap.set(4,height) #4 for height

new_model = load_model('/Users/user/Documents/Project/___e-zest___/Dashboard_division/model/graph_recognisor2.h5')

def preprocessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #img = cv2.equalizeHist(img)
    #normalize image ie its values are from 0-255 to 0-1
    img = img/255
    return img
#check_list = {'0':'Line chart','1':'Bar chart','2':'Scatter plot','3':'Histogram'}

webcam=1

if webcam == 1: #from webcam live
    while True:
        success , imgOriginal = cap.read()
        img = np.asarray(imgOriginal)
        img = cv2.resize(img,(32,32))
        img = preprocessing(img)
        #cv2.imshow("processed Image",cv2.resize(img,(320,320)))
        img = img.reshape(1,32,32,1)
        #predict
        classIndex = int(new_model.predict_classes(img))
        #print(classIndex)
        predictions = new_model.predict(img)
        probVal = np.amax(predictions)
        print(f'{classIndex} | {(probVal*100)}%')  #detect no with accuracy

        if probVal> 0.95:
            cv2.putText(imgOriginal,str(classIndex)+" "+str(probVal*100)+'%',(50,50),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,(0,0,255),1)
        cv2.imshow("Original Image",imgOriginal)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

elif webcam == 0: #local folder
    multiple = 0
    if multiple == 1:  #for multiple pics in folder
        img_arr=[([0]*9) for _ in range(9)]
        for i in range(9):
            for j in range(9):
                name = 'data/image/sq2/temp_' + str(i) + str(j) + '.jpeg'
                img=cv2.imread(name,1)
                #img = img[4:, :-2]
                img = cv2.resize(img, (32, 32))
                img = preprocessing(img)
                img = img.reshape(1, 32, 32, 1)
                classIndex = int(new_model.predict_classes(img))
                predictions = new_model.predict(img)
                probVal = np.amax(predictions)
                if probVal > 0.8:
                    print(f'{name} -> {classIndex} | {(probVal * 100)}%')  # detect no with accuracy
                    img_arr[i][j] = classIndex
                else:
                    print(f'{name} -> {0} | {(probVal * 100)}%')  # detect no with accuracy
                    img_arr[i][j] = 0

        print(img_arr)
    elif multiple == 0:
        name = '/Users/user/Documents/Project/___e-zest___/Dashboard_division/img/Retail_Sales_Distribution_Network_Dashboard_With_Forecasting_3 (1).jpg'
        img = cv2.imread(name, 1)
        cv2.imshow('img',img)
        #img = img[3:,:-2]
        img = cv2.resize(img, (32, 32))
        img = preprocessing(img)
        cv2.imshow('img1', img)
        img = img.reshape(1, 32, 32, 1)

        classIndex = int(new_model.predict_classes(img))
        predictions = new_model.predict(img)
        probVal = np.amax(predictions)
        cv2.waitKey(0)
        if probVal > 0:
            print(f'{classIndex} | {(probVal * 100)}%')  # detect no with accuracy
        # else:
        #     print(f'{classIndex} | {(probVal * 100)}%')  # detect no with accuracy