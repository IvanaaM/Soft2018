import cv2
import numpy as np
import matplotlib.pyplot as plt 
from scipy.ndimage import rotate
from math import hypot

import sys

from PIL import ImageChops

from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD

from PIL import Image

from keras.models import load_model

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 130, 255, cv2.THRESH_BINARY)
    return image_bin

def invert(image):
    return 255-image

def triangle_define(x1, y1, x2, y2, x3, y3, x, y):

    alpha = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))

    beta = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) /((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))

    gamma = 1 - alpha - beta

    ret = False

    if alpha > 0 and beta > 0 and gamma > 0:
        ret = True
    else:
        ret = False

    return ret

def convert_output(y_train):
    nn_outputs = []
    # i - redni broj
    # j - vrednost
    for i, j in enumerate(y_train):
        output = np.zeros(10)
        output[j] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)

def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
        
    return ready_for_ann

def matrix_to_vector(image):
    return image.flatten()

def scale_to_range(image):
    return image/255

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

def display_result(outputs, alphabet):
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result

def detect_numbers(frame_or, frame_num):

    global ukupno
    global listA
    global listB
    global cont
    global regions_array
    global regions_array_copy_test
    global a1
    global a2
    global b1
    global b2

    fram_fin = frame_or.copy()

    white = [255,255,255]

    a1 = listA[0][0]
    b1 = listA[0][1]
    a2 = listB[0][0]
    b2 = listB[0][1]

    img = cv2.GaussianBlur(fram_fin, (3, 3), 0)

    img = invert(image_bin(grayscale(img)))

    kon = img.copy()

    img, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
   
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 40 and h >= 6 and w >= 7:
             cont.append(contour)
             cv2.rectangle(fram_fin,(x,y),(x+w,y+h),(0,255,0),1)
        
             P1 = np.array([a1, b1])
             P2 = np.array([a2, b2])
             P3 = np.array([x, y])

             d=np.cross(P2-P1,P3-P1)/np.linalg.norm(P2-P1)

             if triangle_define(a1,b1,a2,b2,a2,b1,x,y) == True and d > 11 and d < 13:

                 region = kon[y:y+h+1,x:x+w+1]

                 he, we = region.shape
                 
                 if he % 2 != 0:
                     he = he - 1
                
                 if we % 2 != 0:
                     we = we - 1

                 visina = 28 - he
                 sirina = 28 - we

                 top = visina//2
                 left = sirina//2

                 region_with_borders = cv2.copyMakeBorder(region, top, top, left, left, cv2.BORDER_CONSTANT,value=white)
                 
                 novoH, novoW = region_with_borders.shape
                 
                 if novoH == 29:
                     novoH = 28

                 if novoW == 29:
                     novoW = 28

                 regionNew = region_with_borders[0:novoH, 0:novoW]
                 regions_array.append(invert(regionNew))
                 ukupno += 1

def takeFirst(elem):
    return elem[0]

def takeSecond(elem):
    return elem[1]


frame_final = 0
frame_num = 0
frame_num1= 0
frame_num2= 0
frame_num3= 0
frame_num4= 0
frame_num5= 0
frame_num6= 0
frame_num7= 0
frame_num8= 0
frame_num9= 0
a1 = 0
a2 = 0
b1 = 0
b2 = 0

listA = []
listB = []

alphabet = [0,1,2,3,4,5,6,7,8,9]   

cont = []
regions_array_copy_test = []
ukupno = 0

model = load_model('modelSa30.h5')

lin = 0

def getLines(frame_or):

    frame = cv2.GaussianBlur(frame_or, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_blue = np.array([110, 100, 20])
    up_blue = np.array([130, 255, 255])

    detect = cv2.inRange(hsv, low_blue, up_blue)
    edges = cv2.Canny(detect, 75, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 5, maxLineGap=0)

    return lines

#Video 0
regions_array = []

cap = cv2.VideoCapture('videos/video-0.avi')
cap.set(1, frame_num) 
while True:

    frame_num += 1 
    ret_val, frame_or = cap.read() 
    if ret_val:

        lines = getLines(frame_or)

        if(frame_num==1):
            listA=[]
            listB=[]
            lin = lines
            for line in lin:
                listA.append(line[0])
                listB.append(line[0])
                listA.sort(key=takeFirst)
                listB.sort(key=takeSecond)

        detect_numbers(frame_or, frame_num)
 
        key = cv2.waitKey(15)
        if key == 17:
            print("ukupno je :", ukupno)
            break
    else:
        newArray = []

        newArray.append(regions_array[0])

        for i, j in enumerate(regions_array):
                br = 0
                for m, n in enumerate(newArray):
                    if (regions_array[i] == newArray[m]).all():
                        br += 1    
      
                if (br == 0):
                    newArray.append(j)
         
        regions  = [region for region in newArray]

        inputs = prepare_for_ann(regions)

        result = model.predict(np.array(inputs, np.float32))
    
        print(display_result(result, alphabet))
        print('Suma video 0: ', sum(display_result(result, alphabet)))

        suma0 = sum(display_result(result, alphabet))

        break

cap.release()
regions_array = []

#Video 1

cap1 = cv2.VideoCapture('videos/video-1.avi')
listA=[]
listB=[]
cap1.set(1, frame_num1) 
while True:

    frame_num1 += 1 
    ret_val1, frame_or1 = cap1.read() 
    if ret_val1:

        lines = getLines(frame_or1)

        if(frame_num1==1):
            lin = lines
            for line in lin:
                listA.append(line[0])
                listB.append(line[0])
                listA.sort(key=takeFirst)
                listB.sort(key=takeSecond)

        detect_numbers(frame_or1, frame_num1)
 
        key1 = cv2.waitKey(15)
        if key1 == 17:
            print("ukupno je :", ukupno)
            break
    else:
        newArray1 = []

        newArray1.append(regions_array[0])

        for i, j in enumerate(regions_array):
                br = 0
                for m, n in enumerate(newArray1):
                    if (regions_array[i] == newArray1[m]).all():
                        br += 1    
      
                if (br == 0):
                    newArray1.append(j)

        regions  = [region for region in newArray1]

        inputs = prepare_for_ann(regions)

        result = model.predict(np.array(inputs, np.float32))
    
        print(display_result(result, alphabet))
        print('Suma video 1: ', sum(display_result(result, alphabet)))

        suma1 = sum(display_result(result, alphabet))

        break

cap1.release()

#Video 2

regions_array = []

cap2 = cv2.VideoCapture('videos/video-2.avi')
listA=[]
listB=[]
cap2.set(1, frame_num2) 
while True:

    frame_num2 += 1 
    ret_val2, frame_or2 = cap2.read() 
    if ret_val2:

        lines = getLines(frame_or2)

        if(frame_num2==1):
            lin = lines
            for line in lin:
                listA.append(line[0])
                listB.append(line[0])
                listA.sort(key=takeFirst)
                listB.sort(key=takeSecond)

        detect_numbers(frame_or2, frame_num2)
 
        key2 = cv2.waitKey(15)
        if key2 == 17:
            print("ukupno je :", ukupno)
            break
    else:
        newArray2 = []

        newArray2.append(regions_array[0])

        for i, j in enumerate(regions_array):
                br = 0
                for m, n in enumerate(newArray2):
                    if (regions_array[i] == newArray2[m]).all():
                        br += 1    
      
                if (br == 0):
                    newArray2.append(j)

        regions  = [region for region in newArray2]

        inputs = prepare_for_ann(regions)

        result = model.predict(np.array(inputs, np.float32))
    
        print(display_result(result, alphabet))
        print('Suma video 2: ', sum(display_result(result, alphabet)))

        
        suma2 = sum(display_result(result, alphabet))

        break

cap2.release()

#Video 3

regions_array = []

cap3 = cv2.VideoCapture('videos/video-3.avi')
listA=[]
listB=[]
cap3.set(1, frame_num3) 
while True:

    frame_num3 += 1 
    ret_val3, frame_or3 = cap3.read() 
    if ret_val3:

        lines = getLines(frame_or3)

        if(frame_num3==1):
            lin = lines
            for line in lin:
                listA.append(line[0])
                listB.append(line[0])
                listA.sort(key=takeFirst)
                listB.sort(key=takeSecond)

        detect_numbers(frame_or3, frame_num3)
 
        key3 = cv2.waitKey(15)
        if key3 == 17:
            print("ukupno je :", ukupno)
            break
    else:
        newArray3 = []

        newArray3.append(regions_array[0])

        for i, j in enumerate(regions_array):
                br = 0
                for m, n in enumerate(newArray3):
                    if (regions_array[i] == newArray3[m]).all():
                        br += 1    
      
                if (br == 0):
                    newArray3.append(j)

        regions  = [region for region in newArray3]

        inputs = prepare_for_ann(regions)

        result = model.predict(np.array(inputs, np.float32))
    
        print(display_result(result, alphabet))
        print('Suma video 3: ', sum(display_result(result, alphabet)))

        suma3 = sum(display_result(result, alphabet))

        break

cap3.release()

#Video 4

regions_array = []

cap4 = cv2.VideoCapture('videos/video-4.avi')
listA=[]
listB=[]
cap4.set(1, frame_num4) 
while True:

    frame_num4 += 1 
    ret_val4, frame_or4 = cap4.read() 
    if ret_val4:

        lines = getLines(frame_or4)

        if(frame_num4==1):
            lin = lines
            for line in lin:
                listA.append(line[0])
                listB.append(line[0])
                listA.sort(key=takeFirst)
                listB.sort(key=takeSecond)

        detect_numbers(frame_or4, frame_num4)
 
        key4 = cv2.waitKey(15)
        if key4 == 17:
            print("ukupno je :", ukupno)
            break
    else:
        newArray4 = []

        newArray4.append(regions_array[0])

        for i, j in enumerate(regions_array):
                br = 0
                for m, n in enumerate(newArray4):
                    if (regions_array[i] == newArray4[m]).all():
                        br += 1    
      
                if (br == 0):
                    newArray4.append(j)

        regions  = [region for region in newArray4]

        inputs = prepare_for_ann(regions)

        result = model.predict(np.array(inputs, np.float32))

        print(display_result(result, alphabet))
        print('Suma video 4: ', sum(display_result(result, alphabet)))

        suma4 = sum(display_result(result, alphabet))

        break

cap4.release()

#Video 5

regions_array = []

cap5 = cv2.VideoCapture('videos/video-5.avi')
listA=[]
listB=[]
cap5.set(1, frame_num5) 
while True:

    frame_num5 += 1 
    ret_val5, frame_or5 = cap5.read() 
    if ret_val5:

        lines = getLines(frame_or5)

        if(frame_num5==1):
            lin = lines
            for line in lin:
                listA.append(line[0])
                listB.append(line[0])
                listA.sort(key=takeFirst)
                listB.sort(key=takeSecond)

        detect_numbers(frame_or5, frame_num5)
 
        key5 = cv2.waitKey(15)
        if key5 == 17:
            print("ukupno je :", ukupno)
            break
    else:
        newArray5 = []

        newArray5.append(regions_array[0])

        for i, j in enumerate(regions_array):
                br = 0
                for m, n in enumerate(newArray5):
                    if (regions_array[i] == newArray5[m]).all():
                        br += 1    
      
                if (br == 0):
                    newArray5.append(j)

        regions  = [region for region in newArray5]

        inputs = prepare_for_ann(regions)

        result = model.predict(np.array(inputs, np.float32))
    
        print(display_result(result, alphabet))
        print('Suma video 5: ', sum(display_result(result, alphabet)))

        suma5 = sum(display_result(result, alphabet))

        break

cap5.release()

#Video 6

regions_array = []

cap6 = cv2.VideoCapture('videos/video-6.avi')
listA=[]
listB=[]
cap6.set(1, frame_num6) 
while True:

    frame_num6 += 1 
    ret_val6, frame_or6 = cap6.read() 
    if ret_val6:

        lines = getLines(frame_or6)

        if(frame_num6==1):
            lin = lines
            for line in lin:
                listA.append(line[0])
                listB.append(line[0])
                listA.sort(key=takeFirst)
                listB.sort(key=takeSecond)

        detect_numbers(frame_or6, frame_num6)
 
        key6 = cv2.waitKey(15)
        if key6 == 17:
            print("ukupno je :", ukupno)
            break
    else:
        newArray6= []

        newArray6.append(regions_array[0])

        for i, j in enumerate(regions_array):
                br = 0
                for m, n in enumerate(newArray6):
                    if (regions_array[i] == newArray6[m]).all():
                        br += 1    
      
                if (br == 0):
                    newArray6.append(j)

        regions  = [region for region in newArray6]

        inputs = prepare_for_ann(regions)

        result = model.predict(np.array(inputs, np.float32))
    
        print(display_result(result, alphabet))
        print('Suma video 6: ', sum(display_result(result, alphabet)))

        suma6 = sum(display_result(result, alphabet))

        break

cap6.release()

#Video 7

regions_array = []

cap7 = cv2.VideoCapture('videos/video-7.avi')
listA=[]
listB=[]
cap7.set(1, frame_num7) 
while True:

    frame_num7 += 1 
    ret_val7, frame_or7 = cap7.read() 
    if ret_val7:

        lines = getLines(frame_or7)

        if(frame_num7==1):
            lin = lines
            for line in lin:
                listA.append(line[0])
                listB.append(line[0])
                listA.sort(key=takeFirst)
                listB.sort(key=takeSecond)

        detect_numbers(frame_or7, frame_num7)
 
        key7 = cv2.waitKey(15)
        if key7 == 17:
            print("ukupno je :", ukupno)
            break
    else:
        newArray7= []

        newArray7.append(regions_array[0])

        for i, j in enumerate(regions_array):
                br = 0
                for m, n in enumerate(newArray7):
                    if (regions_array[i] == newArray7[m]).all():
                        br += 1    
      
                if (br == 0):
                    newArray7.append(j)

        regions  = [region for region in newArray7]

        inputs = prepare_for_ann(regions)

        result = model.predict(np.array(inputs, np.float32))
    
        print(display_result(result, alphabet))
        print('Suma video 7: ', sum(display_result(result, alphabet)))
        
        suma7 = sum(display_result(result, alphabet))

        break

cap7.release()

#Video 8

regions_array = []

cap8 = cv2.VideoCapture('videos/video-8.avi')
listA=[]
listB=[]
cap8.set(1, frame_num8) 
while True:

    frame_num8 += 1 
    ret_val8, frame_or8 = cap8.read() 
    if ret_val8:

        lines = getLines(frame_or8)

        if(frame_num8==1):
            lin = lines
            for line in lin:
                listA.append(line[0])
                listB.append(line[0])
                listA.sort(key=takeFirst)
                listB.sort(key=takeSecond)

        detect_numbers(frame_or8, frame_num8)
 
        key8 = cv2.waitKey(15)
        if key8== 17:
            print("ukupno je :", ukupno)
            break
    else:
        newArray8= []

        newArray8.append(regions_array[0])

        for i, j in enumerate(regions_array):
                br = 0
                for m, n in enumerate(newArray8):
                    if (regions_array[i] == newArray8[m]).all():
                        br += 1    
      
                if (br == 0):
                    newArray8.append(j)


        regions  = [region for region in newArray8]

        inputs = prepare_for_ann(regions)

        result = model.predict(np.array(inputs, np.float32))
    
        print(display_result(result, alphabet))
        print('Suma video 8: ', sum(display_result(result, alphabet)))
        
        suma8 = sum(display_result(result, alphabet))

        break

cap8.release()

#Video 9

regions_array = []

cap9 = cv2.VideoCapture('videos/video-9.avi')
listA=[]
listB=[]
cap9.set(1, frame_num9) 
while True:

    frame_num9 += 1 
    ret_val9, frame_or9 = cap9.read() 
    if ret_val9:

        lines = getLines(frame_or9)

        if(frame_num9==1):
            lin = lines
            for line in lin:
                listA.append(line[0])
                listB.append(line[0])
                listA.sort(key=takeFirst)
                listB.sort(key=takeSecond)

        detect_numbers(frame_or9, frame_num9)
 
        key9 = cv2.waitKey(15)
        if key9== 17:
            print("ukupno je :", ukupno)
            break
    else:
        newArray9= []

        newArray9.append(regions_array[0])

        for i, j in enumerate(regions_array):
                br = 0
                for m, n in enumerate(newArray9):
                    if (regions_array[i] == newArray9[m]).all():
                        br += 1    
      
                if (br == 0):
                    newArray9.append(j)

        regions  = [region for region in newArray9]

        inputs = prepare_for_ann(regions)

        result = model.predict(np.array(inputs, np.float32))
    
        print(display_result(result, alphabet))
        print('Suma video 9: ', sum(display_result(result, alphabet)))
        suma9 = sum(display_result(result, alphabet))

        break

cap9.release()

with open("out.txt", "r") as file:
    lines = file.readlines()


lines[0] = "RA 69/2015 Ivana Marin\n"
lines[1] = "file\t" + "sum" + "\n"
lines[2] = "video-0.avi\t" + str(suma0) + "\n"
lines[3] = "video-1.avi\t" + str(suma1) + "\n"
lines[4] = "video-2.avi\t" + str(suma2) + "\n"
lines[5] = "video-3.avi\t" + str(suma3) + "\n"
lines[6] = "video-4.avi\t" + str(suma4) + "\n"
lines[7] = "video-5.avi\t" + str(suma5) + "\n"
lines[8] = "video-6.avi\t" + str(suma6) + "\n"
lines[9] = "video-7.avi\t" + str(suma7) + "\n"
lines[10] = "video-8.avi\t" + str(suma8) + "\n"
lines[11] = "video-9.avi\t" + str(suma9) + "\n"

with open("out.txt", "w") as file:
    for line in lines:
        file.write(line)