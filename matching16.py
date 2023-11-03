# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 17:40:33 2023

@author: HOCHOA
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import atan2, cos, sin, sqrt, pi
import math

"""

        self.templateBoard 
        self.templateBase
        self.templateLDrills
        self.templateRDrills
        self.templateOneDrill
    
        self.templateBase
        self.templateBaseDrillC
        self.templateBaseDrillL 
        self.templateBaseDrillR  
        
    def readTemplates(self):
"""

class objectDetect:
  
    def __init__(self):

        # PCB templates
        self.templateBoard = cv2.imread("templates/templateBoard.png", cv2.IMREAD_GRAYSCALE)
        self.templateBase = cv2.imread("templates/templateBase.png", cv2.IMREAD_GRAYSCALE)
        self.templateLDrills = cv2.imread("templates/templateDrillsL.png", cv2.IMREAD_GRAYSCALE)
        self.templateRDrills = cv2.imread("templates/templateDrillsR.png", cv2.IMREAD_GRAYSCALE)
        self.templateOneDrill = cv2.imread("templates/templateOneDrill.png", cv2.IMREAD_GRAYSCALE)
    
        # Base templates
        self.templateBase = cv2.imread("templates/templateBase.png", cv2.IMREAD_GRAYSCALE)
        self.templateBaseDrillC = cv2.imread("templates/templateBaseDrillC.png", cv2.IMREAD_GRAYSCALE)
        # self.templateBaseDrillL = cv2.imread("templates/templateBaseDrillL.png", cv2.IMREAD_GRAYSCALE)
        # self.templateBaseDrillR = cv2.imread("templates/templateBaseDrillR.png", cv2.IMREAD_GRAYSCALE)

        # PCB variables
        self.pcbLeftCentroids = []
        self.pcbRightCentroids = []
        self.pcbCentroidsLength = 0.0
        self.x0p = 0.0
        self.y0p = 0.0
        self.x1p = 0.0
        self.y1p = 0.0
        self.xnewp = 0.0
        self.pcbOrientationDegs = 0.0
        self.pcbOrientationRads= 0.0

        # Base variables
        self.upperCentroid = []
        self.lowerLeftCentroid = []
        self.lowerRightCentroid = []
        self.x0b = 0.0
        self.y0b = 0.0
        self.x1b = 0.0
        self.y1b = 0.0
        self.xnewb = 0.0
        self.baseOrientationDegs = 0.0
        self.baseOrientationRads = 0.0


    def changeBrightnessAndContrast(self, img, brightness, contrast):
        # define the contrast and brightness value
        # Contrast control ( 0 to 127)
        # Brightness control (0-100)

        # call addWeighted function. use beta = 0 to effectively only operate on one image
        out = cv2.addWeighted( img, contrast, img, 0, brightness)

        return out
    
    ############ Centroids calculation ####################

    def inspectCentroids(self, centL, centR):
        (hL,_ )= centL.shape[:2]
        leftCentroids=[]
        rightCentroids=[]

        for i in range(hL-1):
            pt = centL[i]
            if pt[0] ==0 and pt[1]==0: 
                continue
            for n in range(hL-1):
                pt1 = np.array(centL[n+1])
                if pt1[0] > int(pt[0]) and pt1[1] > int(pt[1]):
                    centL[n+1] = [0,0]  
        centL = centL[centL[:,0].argsort()[::-1]] # Ordena por las filas de mayor a menor los izquierdos

        (hR,_ )= centR.shape[:2]
        for i in range(hR-1):
            pt = centR[i]
            if pt[0] ==0 and pt[1]==0: 
                continue
            for n in range(hR-1):
                pt1 = np.array(centR[n+1])
                if pt1[0] < int(pt[0]) and pt1[1] > int(pt[1]):
                    centR[n+1] = [0,0]
        centR=centR[centR[:,0].argsort()]  # Ordena por las filas de menor a mayor los derechos

        for i in range(hL):
            if centL[i][0] != 0 and centL[i][1] != 0:
                leftCentroids.append(centL[i])

        for i in range(hR):
            if centR[i][0] != 0 and centR[i][1] != 0:
                rightCentroids.append(centR[i])        
        Min = min( len(leftCentroids), len(rightCentroids) )

        return leftCentroids, rightCentroids, Min


    def computeCentroids(self,inputImage, template, Dist, th):
        res = cv2.matchTemplate(inputImage,template,cv2.TM_CCORR_NORMED)
        threshold = th
        loc = np.where( res >= threshold)
        centroids = self.getCleanCentroids(inputImage, template, loc, Dist)
        return centroids
    
    def getCleanCentroids(self, inputImage, template, loc, Dist=30):
        (h, w) = template.shape[:2]
        centroids=[]
        for pt in zip(*loc[::-1]):  # pt posicion esquina sup izq del cuadro  
            c = inputImage[pt[1]: pt[1]+h,pt[0]: pt[0]+w,]
            cX, cY = self.getCentroids(c, 120, 255)
            
            [CX,CY] = [pt[0]+cX, pt[1]+cY]
            
            flag = 0
            if centroids == []:
                centroids.append([CX, CY])
            else:
                for cent in centroids:
                    Distance =  np.sum( np.abs(np.array(cent) - np.array([CX, CY]) ) )
                    if Distance > Dist:
                        flag = 1 
                    else:
                        flag = 0
                        break

                if flag == 1:  
                    centroids.append([CX, CY]) 

        return centroids
    
    def getCentroids(self, inputImage, thLow=120, thHigh = 255):
        ret,thresh = cv2.threshold(inputImage,thLow, thHigh,0)
        M = cv2.moments(thresh)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        return cX, cY
    #######################################################
    ############ ROI management        ####################

    def getAugmentedRoi(self,loc, roi, im, N, M):
        (h,w) = im.shape
        (hr,wr) = roi.shape[:2]

        startRow = loc[1]-N
        endRow = loc[1]+hr+M
        startCol = loc[0]-N
        endCol = loc[0]+wr+N

        if startRow < 0:
            startRow = 0
        if endRow > h:
            endRow = h
        
        if startCol < 0:
            startCol = 0
        if endCol > w:
            endCol = w

        return startRow, endRow, startCol, endCol
    #######################################################
    ############ Template matching     ####################
    def obtieneMatch(self, inputImage, template, draw_rect):
        H, W = template.shape
        #resultL = cv2.matchTemplate(inputImage, template, cv2.TM_CCORR_NORMED)
        #resultL = cv2.matchTemplate(inputImage, template, cv2.TM_CCORR_NORMED)
        resultL = cv2.matchTemplate(inputImage,template, cv2.TM_CCOEFF)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(resultL)
        location = max_loc
        bottom_right = (location[0] + W, location[1] + H)  

        if draw_rect == True:
            cv2.rectangle(inputImage, location,bottom_right, 0, 3)

        roi = inputImage[location[1] : location[1]+H,  location[0] : location[0] + W]

        return inputImage, roi, location


    #######################################################
    ############ Orientation calculation ##################
    # Gira dos puntos x,y  alpha radianes
    def rotateCoordinates(self, x, y, x0, y0, angle):
        alpha = math.radians(angle)
        x1 =  int( x*sin(alpha) + y*cos(alpha) + x0 )
        y1 =  int( x*cos(alpha) - y*sin(alpha) + y0)
        return x1, y1


    # x0, y0, punto medio entre dos barrenos
    # x1, y1, punto a 90 grados con respecto al punto medio
    # angleDegs, angleRads, es la orientacion en grados y en radianes
    #                       de la PCB wrt el eje x de la imagen

    def determineOrientation(self, x,y, imShape, angle): 
        (h,w) = imShape
        #alpha = math.radians(angle) 

        # x0, y0 will become the middle point in the lineaa
        # x1, y1 will become the point rotated 90 degrees
        (x0, y0)=np.array(x) - np.array(y)
        x0 //=2 
        x0 += y[0]
        y0 //=2 
        y0 += y[1]
        
        x1, y1 = self.rotateCoordinates((x[1] - y0), (x[0] - x0), x0, y0, angle)

        x0new = x0 + 1500

        #Determina angle wrt to x
        y0new = h-y0
        y1new = h-y1
        
        u = np.array([x0new, y0new]) - np.array([x0, y0new])  
        v = np.array([x1, y1new])    -np.array([x0, y0new])  
        dotProd = np.dot(u,v)
        magU = np. linalg. norm(u)
        magV = np. linalg. norm(v)

        angleRads = np.arccos(dotProd/(magU*magV))
        angleDegs = angleRads * 180 / np.pi 
    
        return x0, y0, x1, y1, x0new, angleDegs, angleRads


    def computePcbOrientation(self, frame):
        imGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imp , roip, locp = self.obtieneMatch(imGray, self.templateBoard, False)  # Detecta la tarjeta
        
        # Aumenta la region de interés detectada de la tarjeta y filtra y aplica correccion de contraste
        startRowP, endRowP, startColP, endColP = self.getAugmentedRoi(locp, roip, imGray, 150, 150)  
        newRoiBoard = imGray[startRowP:endRowP, startColP:endColP]
        newRoiBoard = cv2.medianBlur(newRoiBoard, 5)
        newRoiBoard = self.changeBrightnessAndContrast(newRoiBoard, 0, 2)

        # Calculo de centroides centroidsL en los barrenos superiores izquierdos en la PCB
        # Detecta el área donde se encuentran los barrenos izquierdos 
        # limitando el área de búsqueda en un cuarto de la region 
        # superior izquierda de la tarjeta
        (hal,wal) = newRoiBoard.shape[:2]
        newRoiAreaL = newRoiBoard[0:hal//2, 0:wal//2]
        imDrillL , roidrillL, locdrillL = self.obtieneMatch(newRoiAreaL, self.templateLDrills, False)
        startRowL, endRowL, startColL, endColL = self.getAugmentedRoi(locdrillL, roidrillL, newRoiAreaL, 10, 10)
        newRoiDrillL = newRoiAreaL[startRowL:endRowL, startColL:endColL]
        newRoiDrillL = cv2.medianBlur(newRoiDrillL, 5)  

        # Obtiene los centroides de los barrenos izquierdos con distancias mayores de 30 pixeles
        centroidsL = self.computeCentroids(newRoiDrillL, self.templateOneDrill, 30, 0.8) 


        # Calculo de centroides centroidsR en los barrenos superiores derechos en la PCB
        # Detecta el área donde se encuentran los barrenos derechos  
        # limitando el área de búsqueda en un cuarto de la region 
        # superior derecha de la tarjeta
        (har,war) = newRoiBoard.shape[:2]
        newRoiAreaR = newRoiBoard[0:har//2, war//2:war]
        imDrillR , roidrillR, locdrillR = self.obtieneMatch(newRoiAreaR, self.templateRDrills, False)
        startRowR, endRowR, startColR, endColR = self.getAugmentedRoi(locdrillR, roidrillR, newRoiAreaR, 10, 10)
        newRoiDrillR =  newRoiAreaR[startRowR:endRowR, startColR:endColR]
        newRoiDrillR = cv2.medianBlur(newRoiDrillR, 5)

        # Obtiene centroides de los barrenos derechos con distancias mayores de 40 pixeles
        centroidsR = self.computeCentroids(newRoiDrillR, self.templateOneDrill, 40, 0.8)  

        if len(centroidsL) > 0 and len(centroidsR) > 0 :
            centroidsL += np.array([startColP, startRowP]) + np.array([startColL, startRowL])  # centroids wrt to imGray
            centroidsR += np.array([startColP, startRowP]) + np.array([startColR, startRowR]) + np.array([war//2,0]) # centroides wrt a imGray
        else:
            print("Pasó algo raro en calculo inicial de centroides de la PCB voy por otro frame") 
            print("Verificar: que el brillo/cont de la imagen sean correctos, demasiado brillo/cont no detecto centroides") 
            print("           que la tarjeta no tenga mas de +-10 grados de desviación wrt la imagen") 
            print("           que la tarjeta se encuentra dentro de la escena") 
            return

        # Quedarse con los centroides mas externos delos barrenos y la longitud de ellos 
        # utilizar los tres primero que son los mas estables (98.99%)
        self.pcbLeftCentroids, self.pcbRightCentroids, self.pcbCentroidsLength= self.inspectCentroids(centroidsL, centroidsR) 

        if self.pcbCentroidsLength==0: 
            print("Pasó algo raro en calculo inicial de centroides de la PCB voy por otro frame") 
            print("Verificar: que el brillo/cont de la imagen sean correctos, demasiado brillo/cont no detecto centroides") 
            print("           que la tarjeta no tenga mas de +-10 grados de desviación wrt la imagen") 
            print("           que la tarjeta se encuentra dentro de la escena")
            return
        
        # x0p, y0p punto medio d ela línea formada entre los dos primeros centroides (left - right) 
        # x1p, y1p punto a 90 grado del punto medio con respecto a la línea entre los dos primeros centroides (left - right) 
        self.x0p, self.y0p, self.x1p, self.y1p, self.xnewp, self.pcbOrientationDegs, self.pcbOrientationRads  = self.determineOrientation(self.pcbRightCentroids[0], self.pcbLeftCentroids[0], imGray.shape[:2], 90)

        return  
    def computeBaseOrientation(self, frame):
        imGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        imb , roib, locb = self.obtieneMatch(imGray, self.templateBase, False)   # Detecta la base     

        # Aumenta la region de interés detectada de la base 
        startRowB, endRowB, startColB, endColB = self.getAugmentedRoi(locb, roib, imGray, 150, 150)
        newRoiBase = imGray[startRowB:endRowB, startColB:endColB]
        (hb,wb) = newRoiBase.shape[:2]

        # Detecta los tres barenos mas externos en la base
        imDrillC , roidrillC, locdrillC = self.obtieneMatch(newRoiBase[0:int(hb//2), :], self.templateBaseDrillC, False)
        imDrillL , roidrillL, locdrillL = self.obtieneMatch(newRoiBase[int(hb//2):hb, 0:wb//2], self.templateBaseDrillC, False)
        imDrillR , roidrillR, locdrillR = self.obtieneMatch(newRoiBase[int(hb//2):hb, wb//2:wb], self.templateBaseDrillC, False)

        # Detecta los centroides de los tres barrenos detectados  
        self.upperCentroid = self.getCentroids(roidrillC, 120, 255)
        self.lowerLeftCentroid = self.getCentroids(roidrillL, 120, 255)
        self.lowerRightCentroid = self.getCentroids(roidrillR, 120, 255)
        #centroidsBC = self.getCentroids(roidrillC, 120, 255)
        #centroidsBL = self.getCentroids(roidrillL, 120, 255)
        #centroidsBR = self.getCentroids(roidrillR, 120, 255)

                #verifica centroides
        if len(self.upperCentroid) ==2 and len(self.lowerLeftCentroid) == 2 and len(self.lowerRightCentroid) == 2:
            self.upperCentroid  += np.array([startColB, startRowB])+np.array([locdrillC[0], locdrillC[1]])
            self.lowerLeftCentroid  += np.array([startColB, startRowB])+np.array([locdrillL[0], locdrillL[1]+hb//2])
            self.lowerRightCentroid += np.array([startColB, startRowB])+np.array([locdrillR[0]+wb//2, locdrillR[1]+hb//2])
        else:
            print("Pasó algo raro en calculo inicial de centroides de la BASE voy por otro frame") 
            print("Verificar: que el brillo/cont de la imagen sean correctos, demasiado brillo/cont no detecto centroides") 
            print("           que la tarjeta no tenga mas de +-10 grados de desviación wrt la imagen") 
            print("           que la tarjeta se encuentra dentro de la escena")
            return 
         
        # Determina la orientacion de la base en grados
        # x0p, y0p punto medio de la línea formada entre los dos centroides inferiores (left - right) 
        # x1p, y1p punto a 90 grado del punto medio con respecto a la línea entre los dos centroides inferiores (left - right) 

        self.x0b, self.y0b, self.x1b, self.y1b, self.xnewb, self.baseOrientationDegs, self.baseOrientationRads = self.determineOrientation(self.lowerRightCentroid, self.lowerLeftCentroid,imGray.shape[:2], 90)
    
    def drawPcbInfo(self, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX 
        orgL = np.array([50, 300])
        orgR = np.array([600, 300])
        fontScale = 2 
        cv2.putText(frame, "Centroides de los barrenos rojos en la PCB", orgL, font, fontScale,(0,0,255), 6, cv2.LINE_AA, False)
        orgL[1] = orgL[1]+100
        cv2.putText(frame, "Izquierdos", orgL, font, fontScale,(0,0,255), 6, cv2.LINE_AA, False)
        orgR[1] = orgR[1]+100
        cv2.putText(frame, "Derechos", orgR, font, fontScale,(0,0,255), 6, cv2.LINE_AA, False)

        if  self.pcbCentroidsLength ==0: return
        if  self.pcbCentroidsLength >3 :  self.pcbCentroidsLength = 3

        for i in range(self.pcbCentroidsLength):
            LabelL = str(self.pcbLeftCentroids[i])
            LabelR = str(self.pcbRightCentroids[i])
            orgL[1] = orgL[1]+100
            orgR[1] = orgR[1]+100
            cv2.putText(frame, LabelL, orgL, font, fontScale,(0,0,255), 5, cv2.LINE_AA, False) 
            cv2.putText(frame, LabelR, orgR, font, fontScale,(0,0,255), 5, cv2.LINE_AA, False) 
            cv2.circle(frame, self.pcbLeftCentroids[i], 3, (0, 0, 255), 8)
            cv2.circle(frame, self.pcbRightCentroids[i], 3, (0, 0, 255), 8)
        
        orgL[1] = orgL[1]+100
        pcbOrient= round(self.pcbOrientationDegs, 2)
        Label = "Orientacion de la tarjeta: " + str(pcbOrient) + " grados"
        cv2.putText(frame, Label, orgL, font, fontScale,(0,0,0), 5, cv2.LINE_AA, False) 
        cv2.line(frame, self.pcbLeftCentroids[0], self.pcbRightCentroids[0], (0, 0, 255), 5)
        cv2.line(frame, [self.x0p,self.y0p], [self.x1p,self.y1p], (0, 0, 255), 5)   # y axis red

        #Axes
        cv2.arrowedLine(frame, [self.x0p,self.y0p], [self.x0p+1000,self.y0p], (0, 0, 0), 3) # X Axis in black color
        cv2.arrowedLine(frame, [self.x0p,self.y0p], [self.x0p,self.y0p-1000], (0, 0, 0), 3) # Y Axis in black color
    
    def drawBaseInfo(self, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX 
        org = np.array([1800, 1800])
        fontScale = 2 
        cv2.putText(frame, "Centroides de los barrenos de la base", org, font, fontScale,(255,0,0), 6, cv2.LINE_AA, False)
        Label = str(self.upperCentroid)
        org[0] = 2200
        org[1] = org[1]+100
        cv2.putText(frame, Label, org, font, fontScale,(0,0,255), 6, cv2.LINE_AA, False)
        Label = str(self.lowerLeftCentroid)
        org[1] = org[1]+100
        cv2.putText(frame, Label, org, font, fontScale,(0,255,0), 6, cv2.LINE_AA, False)
        Label = str(self.lowerRightCentroid)
        org[1] = org[1]+100
        cv2.putText(frame, Label, org, font, fontScale,(255,0,0), 6, cv2.LINE_AA, False) 
        org[0] = 1800
        org[1] = org[1]+100
        baseOrient= round(self.baseOrientationDegs, 2)
        Label = "Orientacion de la base: " + str(baseOrient) + " grados"
        cv2.putText(frame, Label, org, font, fontScale,(0,0,0), 5, cv2.LINE_AA, False) 


        cv2.line(frame, self.lowerLeftCentroid, self.lowerRightCentroid, (255, 0, 0), 5)
        cv2.circle(frame, self.upperCentroid, 3, (0, 0, 255), 15)
        cv2.circle(frame, self.lowerLeftCentroid, 3, (0, 255, 0), 15)
        cv2.circle(frame, self.lowerRightCentroid, 3, (255, 0, 0), 15)
        cv2.line(frame, [self.x0b,self.y0b], [self.x1b,self.y1b], (255, 0, 0), 5)   # y Axis red

        #Axes
        cv2.arrowedLine(frame, [self.x0b,self.y0b], [self.x0b+1000,self.y0b], (0, 0, 0), 3) # X Axis black
        cv2.arrowedLine(frame, [self.x0b,self.y0b], [self.x0b,self.y0b-1000], (0, 0, 0), 3) # Y Axis balck


    def printImage(self, imGray):
        plt.imshow(imGray, cmap='gray')
        plt.show()

    def computeOrientations(self, frame):
        self.computePcbOrientation(frame)
        self.computeBaseOrientation(frame)
        self.drawPcbInfo(frame)
        self.drawBaseInfo(frame)




if __name__ == "__main__":
    #frame = cv2.imread("originalImage.png")
    filename = 'video/videoNoLight.avi'
    
    cap = cv2.VideoCapture(filename)
    P=objectDetect()
    ii  = 0
    while(True):

        if ii < 10:   # Inicia descartando los primeros 20 frames
            ret, frame = cap.read()
            ii+=1
            continue
        else:
            ret, frame = cap.read() 
        
        if ret == True:
            P.computeOrientations(frame)
            
            resized = cv2.resize(frame, (frame.shape[1]//4,frame.shape[0]//4), interpolation = cv2.INTER_AREA)

            #writer.write(fr)

            cv2.imshow("Frame", resized)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:  
            break


    #cv2.imwrite("base90deg.png",fr)
    cap.release()
    #writer.release()
    cv2.destroyAllWindows()  






        

