

import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import atan2, cos, sin, sqrt, pi
import math

#Brightness =0, contrast 64
def configureVideo(Num, width, height):
    #cap = cv2.VideoCapture(Num, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(Num)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_MONOCHROME, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, -3.0)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, -128.0)
    #cap.set(cv2.CAP_PROP_CONTRAST, 64.0)
    return cap

def getCentroids(inputImage, thLow=120, thHigh = 255):
    ret,thresh = cv2.threshold(inputImage,thLow, thHigh,0)
    M = cv2.moments(thresh)

    if M["m00"] != 0:
      cX = int(M["m10"] / M["m00"])
      cY = int(M["m01"] / M["m00"])
    else:
      cX, cY = 0, 0

    return cX, cY

def getCleanCentroids(inputImage, template, loc, Dist=30):

  #w, h = template.shape[::-1]
  (h, w) = template.shape[:2]
  centroids=[]

  for pt in zip(*loc[::-1]):  # pt posicion esquina sup izq del cuadro  

    c = inputImage[pt[1]: pt[1]+h,pt[0]: pt[0]+w,]

    cX, cY = getCentroids(c, 120, 255)
    
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
  

def computeCentroids(inputImage, template, Dist, th):
    res = cv2.matchTemplate(inputImage,template,cv2.TM_CCORR_NORMED)
    threshold = th
    loc = np.where( res >= threshold)

    centroids = getCleanCentroids(inputImage, template, loc, Dist)
    
    return centroids



def obtieneMatch(inputImage, template, draw_rect):
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

def getAugmenteBox(loc, roi, im, N, M):

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


def inspectCentroids(centL, centR):
  (hL,wL)= centL.shape[:2]
  leftCentroids=[]
  rightCentroids=[]

  for i in range(hL-1):
    pt = centL[i]
    
    if pt[0] ==0 and pt[1]==0: continue

    for n in range(hL-1):
      pt1 = np.array(centL[n+1])
      if pt1[0] > int(pt[0]) and pt1[1] > int(pt[1]):
        centL[n+1] = [0,0]  
  centL = centL[centL[:,0].argsort()[::-1]] # Ordena por las filas de mayor a menor los izquierdos

  (hR,wR)= centR.shape[:2]
  for i in range(hR-1):
    pt = centR[i]
    
    if pt[0] ==0 and pt[1]==0: continue

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

def rotateCoordinates(x, y, x0, y0, angle):
    alpha = math.radians(angle)
    x1 =  int( x*sin(alpha) + y*cos(alpha) + x0 )
    y1 =  int( x*cos(alpha) - y*sin(alpha) + y0)


    return x1, y1


def determinePointOnBase(x,y,pix, imShape, angle):
    

     # x0, y0 will become the middle point in the line
    xtemp = np.copy(x)
    ytemp = np.copy(y)
 
    (x0, y0)=np.array(xtemp) - np.array(ytemp)
    x0 //=2 
    x0 += ytemp[0]
    y0 //=2 
    y0 += ytemp[1]

    xtemp[0] = x0+pix#x0+500
    xtemp[1] = y0
    
    x1, y1 = rotateCoordinates((xtemp[1] - y0), (xtemp[0] - x0), x0, y0, angle)
     
    # Falta calcular el punto a pix pixeles

    return x1, y1 

def determineOrientation(x,y, imShape, angle): 
    (h,w) = imShape
    #alpha = math.radians(angle) 

     # x0, y0 will become the middle point in the line
     # x1, y1 will become the point rotated 90 degrees
    (x0, y0)=np.array(x) - np.array(y)
    x0 //=2 
    x0 += y[0]
    y0 //=2 
    y0 += y[1]
    
    x1, y1 = rotateCoordinates((x[1] - y0), (x[0] - x0), x0, y0, angle)

    x0new = x0 + 1500

    #Determine angle wrt to x
  
    y0new = h-y0
    y1new = h-y1
    
    u = np.array([x0new, y0new]) - np.array([x0, y0new])  
    v = np.array([x1, y1new]) -np.array([x0, y0new])  
    dotProd = np.dot(u,v)
    magU = np. linalg. norm(u)
    magV = np. linalg. norm(v)

    angleRads = np.arccos(dotProd/(magU*magV))
    angleDegs = angleRads * 180 / np.pi 
   
    return x0,y0,x1,y1,x0new, angleDegs

def changeBrightnessAndContrast(img, brightness, contrast):
    # define the contrast and brightness value
    # Contrast control ( 0 to 127)
    # Brightness control (0-100)

    # call addWeighted function. use beta = 0 to effectively only operate on one image
    out = cv2.addWeighted( img, contrast, img, 0, brightness)

    return out



#######################################################################################################################################
  #######################################################################################################################################
    #######################################################################################################################################
      #######################################################################################################################################


if __name__ == "__main__":

  # Inicializa la captura de video
  filename = 'video/videoNoLight.avi'
  cap = cv2.VideoCapture(filename)
  #cap=configureVideo(0, 3264, 2448)


  # Leer las plantillas de la tarjeta
  templateBoard = cv2.imread("templates/templateBoard.png", cv2.IMREAD_GRAYSCALE)
  templateBase = cv2.imread("templates/templateBase.png", cv2.IMREAD_GRAYSCALE)
  templateLDrills = cv2.imread("templates/templateDrillsL.png", cv2.IMREAD_GRAYSCALE)
  templateRDrills = cv2.imread("templates/templateDrillsR.png", cv2.IMREAD_GRAYSCALE)
  templateOneDrill = cv2.imread("templates/templateOneDrill.png", cv2.IMREAD_GRAYSCALE)

  # Leer las platillas de la base
  templateBase = cv2.imread("templates/templateBase.png", cv2.IMREAD_GRAYSCALE)
  templateBaseDrillC = cv2.imread("templates/templateBaseDrillC.png", cv2.IMREAD_GRAYSCALE)
  templateBaseDrillL = cv2.imread("templates/templateBaseDrillL.png", cv2.IMREAD_GRAYSCALE)
  templateBaseDrillR = cv2.imread("templates/templateBaseDrillR.png", cv2.IMREAD_GRAYSCALE)

  ii  = 0
  while(True):

    if ii < 10:   # Inicia descartando los primeros 20 frames
      ret, frame = cap.read()
      ii+=1
      continue
    else:
      ret, frame = cap.read() 
      fr = frame

    ii+=1

    if ret == True:
      imGray =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      

  #################################  #################################   ################################# 
  ################################# PCB PROCESSING   #################################   #################

      #imGray = changeBrightnessAndContrast(imGray, 6, 2)     
      imp , roip, locp = obtieneMatch(imGray, templateBoard, False)  # Detecta la tarjeta   

      # Aumenta la region de interés detectada de la tarjeta y fitlra y aplica correccion de contraste
      startRowP, endRowP, startColP, endColP = getAugmenteBox(locp, roip, imGray, 150, 150)
      newRoiBoard = imGray[startRowP:endRowP, startColP:endColP]
      newRoiBoard = cv2.medianBlur(newRoiBoard, 5)
      newRoiBoard = changeBrightnessAndContrast(newRoiBoard, 0, 2)

      # Detecta el área donde se encuentran los barrenos izquierdos 
      # limitando el área de búsqueda en un cuarto de la region 
      # superior izquierda de la tarjeta
      (hal,wal) = newRoiBoard.shape[:2]
      newRoiAreaL = newRoiBoard[0:hal//2, 0:wal//2]
      imDrillL , roidrillL, locdrillL = obtieneMatch(newRoiAreaL, templateLDrills, False)
      startRowL, endRowL, startColL, endColL = getAugmenteBox(locdrillL, roidrillL, newRoiAreaL, 10, 10)
      newRoiDrillL = newRoiAreaL[startRowL:endRowL, startColL:endColL]
      newRoiDrillL = cv2.medianBlur(newRoiDrillL, 5)

      # Detecta el área donde se encuentran los barrenos derechos  
      # limitando el área de búsqueda en un cuarto de la region 
      # superior derecha de la tarjeta
      (har,war) = newRoiBoard.shape[:2]
      newRoiAreaR = newRoiBoard[0:har//2, war//2:war]
      imDrillR , roidrillR, locdrillR = obtieneMatch(newRoiAreaR, templateRDrills, False)
      startRowR, endRowR, startColR, endColR = getAugmenteBox(locdrillR, roidrillR, newRoiAreaR, 10, 10)
      newRoiDrillR =  newRoiAreaR[startRowR:endRowR, startColR:endColR]
      newRoiDrillR = cv2.medianBlur(newRoiDrillR, 5)
 
      # Obtiene centroides de los barrenos izquierdos con distancias mayores de 
      # 30 pixeles
      centroidsL = computeCentroids(newRoiDrillL, templateOneDrill, 30, 0.8) 

      # Obtiene centroides de los barrenos derechos con distancias mayores de 
      # 30 pixeles
      centroidsR = computeCentroids(newRoiDrillR, templateOneDrill, 40, 0.8)  

      if len(centroidsL) > 0 and len(centroidsR) > 0 :
        centroidsL += np.array([startColP, startRowP]) + np.array([startColL, startRowL])  # centroids wrt to imGray
        centroidsR += np.array([startColP, startRowP]) + np.array([startColR, startRowR]) + np.array([war//2,0]) # centroides wrt a imGray
      else:
        print("Pasó algo raro en calculo inicial de centroides de la PCB voy por otro frame") 
        continue

     
      newCentroidsL, newCentroidsR, L= inspectCentroids(centroidsL, centroidsR) 

     

      if L==0: 
        print("No hay nuevos centroides voy por otro frame") 
        continue  

      x0p, y0p, x1p, y1p,xnewp, pcbOrientation= determineOrientation(newCentroidsR[0], newCentroidsL[0], imGray.shape[:2], 90)


      #################################  #################################   ################################# 
      ################################# PLOT INFO ON PCB FRAME ###########################   #################
      #################################  THIS SECTION CAN BE DELETED #####################   #################

      font = cv2.FONT_HERSHEY_SIMPLEX 
      orgL = np.array([50, 300])
      orgR = np.array([600, 300])
      fontScale = 2 
      cv2.putText(fr, "Centroides de los barrenos rojos en la PCB", orgL, font, fontScale,(0,0,255), 6, cv2.LINE_AA, False)
      orgL[1] = orgL[1]+100
      cv2.putText(fr, "Izquierdos", orgL, font, fontScale,(0,0,255), 6, cv2.LINE_AA, False)
      orgR[1] = orgR[1]+100
      cv2.putText(fr, "Derechos", orgR, font, fontScale,(0,0,255), 6, cv2.LINE_AA, False)

   
      if L ==0: continue
      if L >3 : L = 3

  
      for i in range(L):
        LabelL = str(newCentroidsL[i])
        LabelR = str(newCentroidsR[i])
        orgL[1] = orgL[1]+100
        orgR[1] = orgR[1]+100
        cv2.putText(fr, LabelL, orgL, font, fontScale,(0,0,255), 5, cv2.LINE_AA, False) 
        cv2.putText(fr, LabelR, orgR, font, fontScale,(0,0,255), 5, cv2.LINE_AA, False) 
        cv2.circle(fr, newCentroidsL[i], 3, (0, 0, 255), 8)
        cv2.circle(fr, newCentroidsR[i], 3, (0, 0, 255), 8)

      orgL[1] = orgL[1]+100
      pcbOrient= round(pcbOrientation, 2)
      Label = "Orientacion de la tarjeta: " + str(pcbOrient) + " grados"
      cv2.putText(fr, Label, orgL, font, fontScale,(0,0,0), 5, cv2.LINE_AA, False) 
    

      
      cv2.line(fr, centroidsL[0], centroidsR[0], (0, 0, 255), 5)

    
      cv2.line(fr, [x0p,y0p], [x1p,y1p], (0, 0, 255), 5)   # y axis red
      cv2.arrowedLine(fr, [x0p,y0p], [x0p+1000,y0p], (0, 0, 0), 3) # X Axis in black color
      cv2.arrowedLine(fr, [x0p,y0p], [x0p,y0p-1000], (0, 0, 0), 3) # Y Axis in black color

      #################################  #################################   ################################# 
      ################################# BASE PROCESSING   ################################   #################

      # Detecta la Base
      imb , roib, locb = obtieneMatch(imGray, templateBase, False)   # Detecta la base     

      # Aumenta la region de interés detectada de la base 
      startRowB, endRowB, startColB, endColB = getAugmenteBox(locb, roib, imGray, 150, 150)
      newRoiBase = imGray[startRowB:endRowB, startColB:endColB]
      (hb,wb) = newRoiBase.shape[:2]

      # Detecta los tres barenos mas externos en la base
      imDrillC , roidrillC, locdrillC = obtieneMatch(newRoiBase[0:int(hb//2), :], templateBaseDrillC, False)
      imDrillL , roidrillL, locdrillL = obtieneMatch(newRoiBase[int(hb//2):hb, 0:wb//2], templateBaseDrillC, False)
      imDrillR , roidrillR, locdrillR = obtieneMatch(newRoiBase[int(hb//2):hb, wb//2:wb], templateBaseDrillC, False)

      # Detecta los centroides de los tres barrenos detectados  
      centroidsBC = getCentroids(roidrillC, 120, 255)
      centroidsBL = getCentroids(roidrillL, 120, 255)
      centroidsBR = getCentroids(roidrillR, 120, 255)

      #verifica centroides
      if len(centroidsBC) ==2 and len(centroidsBL) == 2 and len(centroidsBR) == 2:
        centroidsBC += np.array([startColB, startRowB])+np.array([locdrillC[0], locdrillC[1]])
        centroidsBL += np.array([startColB, startRowB])+np.array([locdrillL[0], locdrillL[1]+hb//2])
        centroidsBR += np.array([startColB, startRowB])+np.array([locdrillR[0]+wb//2, locdrillR[1]+hb//2])
      else:
        print("Pasó algo raro al tratar de calcular los centroides de la base voy por otro frame") 
        continue  

      # Determina la orientacion de la base en grados
      x0b, y0b, x1b, y1b, xnewb, baseOrientation = determineOrientation(centroidsBR, centroidsBL,imGray.shape[:2], 90)


  

      #################################  #################################   ################################# 
      ################################# PLOT INFO ON BASE FRAME ##########################   #################
      #################################  THIS SECTION CAN BE DELETED #####################   #################

      cv2.line(fr, centroidsBL, centroidsBR, (255, 0, 0), 5)
      cv2.circle(fr, centroidsBC, 3, (0, 0, 255), 15)
      cv2.circle(fr, centroidsBL, 3, (0, 255, 0), 15)
      cv2.circle(fr, centroidsBR, 3, (255, 0, 0), 15)
      cv2.line(fr, [x0b,y0b], [x1b,y1b], (255, 0, 0), 5)   # y Axis red
      cv2.arrowedLine(fr, [x0b,y0b], [x0b+1000,y0b], (0, 0, 0), 3) # X Axis black
      cv2.arrowedLine(fr, [x0b,y0b], [x0b,y0b-1000], (0, 0, 0), 3) # Y Axis balck


    
      font = cv2.FONT_HERSHEY_SIMPLEX 
      org = np.array([1800, 1800])
      fontScale = 2 
      cv2.putText(fr, "Centroides de los barrenos de la base", org, font, fontScale,(255,0,0), 6, cv2.LINE_AA, False)
      Label = str(centroidsBC)
      org[0] = 2200
      org[1] = org[1]+100
      cv2.putText(fr, Label, org, font, fontScale,(0,0,255), 6, cv2.LINE_AA, False)
      Label = str(centroidsBL)
      org[1] = org[1]+100
      cv2.putText(fr, Label, org, font, fontScale,(0,255,0), 6, cv2.LINE_AA, False)
      Label = str(centroidsBR)
      org[1] = org[1]+100
      cv2.putText(fr, Label, org, font, fontScale,(255,0,0), 6, cv2.LINE_AA, False) 
      org[0] = 1800
      org[1] = org[1]+100
      baseOrient= round(baseOrientation, 2)
      Label = "Orientacion de la base: " + str(baseOrient) + " grados"
      cv2.putText(fr, Label, org, font, fontScale,(0,0,0), 5, cv2.LINE_AA, False) 
      #################################  #################################   ################################# 

      
      """ 
        Dibuja la orientación de la base sobre con respecto a la orientación de la tarjeat (línea verde)
        y dibuja un círculo a 100 pixeles con respecto a la orientaciónn de ba tarjeta (cículo nego)
      """ 
      x1c, y1c = determinePointOnBase(newCentroidsR[0], newCentroidsL[0],500, imGray.shape[:2], baseOrientation)
      cv2.line(fr, [x0p,y0p], [x1c,y1c], (255, 0, 0), 5)

      x1d, y1d = determinePointOnBase(newCentroidsR[0], newCentroidsL[0],60, imGray.shape[:2], baseOrientation)
      cv2.circle(fr, [x1d,y1d], 10, (0, 0, 0),3 )
      
      #print(f"Difencia de orientaciones: ", baseOrientation - pcbOrientation)
      #print("Centroides de los barrenos izcsquierdos de la trarjeta: ", newCentroidsL)
      #print("Centroides de los barrenos derechos de la trarjeta: ", newCentroidsR)
      #print("Centroides de los barrenos de la base: ", centroidsBC, centroidsBL, centroidsBR)




      resized = cv2.resize(fr, (frame.shape[1]//4,frame.shape[0]//4), interpolation = cv2.INTER_AREA)

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


  
"""
[cX, cY] = getCentroids(roidrillC, 120, 255)
[CX,CY] = [cX, cY] + np.array([startColB, startRowB])+np.array([locdrillC[0], locdrillC[1]])
centroids.append([CX,CY])

cX, cY = getCentroids(roidrillL, 120, 255)
[CX,CY] = [cX, cY] + np.array([startColB, startRowB])+np.array([locdrillL[0], locdrillL[1]+hb//2])
centroids.append([CX,CY])

cX, cY = getCentroids(roidrillR, 120, 255)
[CX,CY] = [cX, cY] + np.array([startColB, startRowB])+np.array([locdrillR[0]+wb//2, locdrillR[1]+hb//2])
centroids.append([CX,CY])


#Calcula la orientación de la base
x0b, y0b, x1b, y1b, xnewb, baseOrientation = determineOrientation(centroids[2], centroids[1],imGray.shape[:2], 90)
"""