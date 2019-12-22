import cv2
import numpy as np
import numpy.ma as ma
import os, glob, re
import shutil
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity

def getVertical(verticalHist,W):
    righters = []
    gotZero = False
    for y in range(2,W-1):
        if not(gotZero) and verticalHist[y-2]+verticalHist[y-1]+verticalHist[y] == 0:
            righters.append(y)
            gotZero = True
        if verticalHist[y] != 0:
            gotZero = False
    for y in range(len(righters),1):
        if righters[y]-15 < righters[y-1]:
            del righters[y-1]
    return righters

def wordSegmentation(img):
    image = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## (2) threshold
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    ## (3) minAreaRect on the nozeros
    pts = cv2.findNonZero(threshed)
    ret = cv2.minAreaRect(pts)
    (cx,cy), (w,h), ang = ret
    if not(ang > -15):
        ang += 90
    ## (4) Find rotated matrix, do rotation
    M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
    rotated = cv2.warpAffine(threshed, M, (img.shape[1], img.shape[0]))
    ## (5) find and draw the upper and lower boundary of each lines
    hist = cv2.reduce(rotated,1, cv2.REDUCE_AVG).reshape(-1)
    th = 2
    H,W = img.shape[:2]
    uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
    lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),borderValue=(255,255,255),flags=cv2.INTER_LANCZOS4)
    wordSegmentation = []
    for y in range(len(uppers)):
        if (lowers[y] - uppers[y]) > 5:
            line = rotated.copy()
            line = line[uppers[y]-4:lowers[y]+4,0::]
            H,W = line.shape[:2]
            verticalHist = cv2.reduce(np.float32(line),0, cv2.REDUCE_SUM, cv2.CV_64F).reshape(-1)
            righters = getVertical(verticalHist,W)
            wordSegmentation.append(righters)
        else:
            wordSegmentation.append([])
    rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)

    word_images = []

    for y in range(len(uppers)):
        if (lowers[y] - uppers[y]) > 5:
            for z in range(len(wordSegmentation[y])-1,0,-1):
                sup = image[uppers[y]-4:lowers[y]+4,wordSegmentation[y][z-1]:wordSegmentation[y][z],:]
                word_images.append(sup)
    
    return word_images

def convolve(image, kernel):
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
        cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")

    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
 
            k = (roi * kernel).sum()

            output[y - pad, x - pad] = k

    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    return output

def clearUpperPart(img):
    newImg = img.copy()
    for x in range((int)((newImg.shape[0]*0.39))):
        for y in range(newImg.shape[1]):
            newImg[x,y] = 255
    return newImg

def MaskRow(img,row):
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if(x != row):
                img[x,y] = 0

def thresholdClearAbove(img,threshold):
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if(img[x,y] > threshold):
                img[x,y] = 255

def thresholdClearBelow(img,threshold):
    newImg = img.copy()
    for x in range(newImg.shape[0]):
        for y in range(newImg.shape[1]):
            if(newImg[x,y] < threshold):
                newImg[x,y] = 0
    return newImg

def Binarize(img,threshold):
    newImg = img.copy()
    for x in range(newImg.shape[0]):
        for y in range(newImg.shape[1]):
            if(newImg[x,y] < threshold):
                newImg[x,y] = 0
            else:
                newImg[x,y] = 255
    return newImg

def BlackPixelsNumberAbove(img,Threshold):
    noBlackPixels = 0
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if(img[x,y] < 130):
                noBlackPixels += 1
    return (noBlackPixels > Threshold)

def calculateVerticalPixels(mainImg,maskImage,row,threshold):
    for y in range(mainImg.shape[1]):
        if(maskImage[row,y] < 240):
            continue
        blackValue = 0
        pixelsAdded = 0
        for x in range(mainImg.shape[0]):
            if(x == row):
                continue
            if(mainImg[x, y] == 0):
                pixelsAdded += 1
        if(pixelsAdded > threshold):
            maskImage[row,y] = 0

def AdjustMaskWithRange(maskImg,binImg,row):
    for y in range(maskImg.shape[1] - 1):
        index = maskImg.shape[1] - y - 2
        HasTwoAdjacentWhites = True
        whitePxlsRange = 3
        for pixelMargin in range(whitePxlsRange):
            if(binImg[row, index + pixelMargin - (int)(whitePxlsRange/2)] < 10):
                HasTwoAdjacentWhites = False
        if(HasTwoAdjacentWhites):
            maskImg[row, index] = 0
        else:
            if(maskImg[row, index] > 200 and maskImg[row,index - 1] < 10):
                maskImg[row, index] = 255
            else:
                maskImg[row, index] = 0

def AdjustMask(maskImg,binImg,row):
    for y in range(maskImg.shape[1] - 1):
        index = maskImg.shape[1] - y - 2
        if(maskImg[row, index] > 200 and maskImg[row,index - 1] < 10):
            maskImg[row, index] = 255
        else:
            maskImg[row, index] = 0

def GetBaseLine(binImg):
    leastLineBlackCount = 0
    baseline = 0
    for x in range(binImg.shape[0]):
        noBlack = 0
        for y in range(binImg.shape[1]):
            if(binImg[x,y] < 10):
                noBlack += 1
        if(noBlack >= (leastLineBlackCount*(3.58/4))):
            leastLineBlackCount = noBlack
            baseline = x
    return baseline 

def detectHoles(binImg,maskImg,row):
    for y in range(binImg.shape[1]):
        if(maskImg[row,y] < 240):
            continue
        numberOfTransitions = 0
        for x in range(binImg.shape[0] - 1):
            if(binImg[x,y] != binImg[x+1,y]):
                numberOfTransitions += 1
        if(numberOfTransitions > 2):
            maskImg[row,y] = 0

def sepLetters(binImg,maskImg,row):
    for y in range(binImg.shape[1] - 1):
        whiteLastRow = True
        whiteCurrRow = True
        for x in range(binImg.shape[0]):
            if(binImg[x,y] < 10):
                whiteLastRow = False
        for x in range(binImg.shape[0]):
            if(binImg[x,y + 1] < 10):
                whiteCurrRow = False
        if(whiteLastRow and not whiteCurrRow):
            maskImg[row,y] = 255                

def removeContinousLines(binImg,maskImg,row):
    for y in range(binImg.shape[1] - 1):
        if(maskImg[row,y] < 240):
            continue
        movingOnContinousLine = True
        i = 1
        while i < binImg.shape[1] - y - 1:
            if(binImg[row,y+i] != binImg[row,y+i-1]):
                movingOnContinousLine = False
            if(maskImg[row,y+i] > 240):
                break
            i += 1
        if(movingOnContinousLine):
            maskImg[row,y+i] = 0                 

def createLettersFromImgMask(img,maskImg,row):
    lettersList = []
    for y in range(maskImg.shape[1] - 1):
        if(maskImg[row,y] < 240):
            continue
        i = 1
        lastLetter = True
        while i < maskImg.shape[1] - y:
            if(maskImg[row,y+i] > 240):
                lastLetter = False
                if(i > 2):
                    lettersList.append(img[0:maskImg.shape[0], y+1:y+i+1])
                break
            i += 1
        if (lastLetter and y != maskImg.shape[1] - 1):
            if(BlackPixelsNumberAbove(img[0:maskImg.shape[0], y:maskImg.shape[1]],5)):
                lettersList.append(img[0:maskImg.shape[0], y:maskImg.shape[1]])
    return lettersList

def removeLastSeperatorIfRedundant(binImg,maskImg,row,blackPixelsThreshold):
    firstSep = 0
    firstSepCountered = False
    
    for y in range(maskImg.shape[1]):
        if(maskImg[row,y] > 240):
            if(firstSepCountered):
                firstSep = y - 1
                break
            else:
                firstSepCountered = True
                
    blackPixels = 0
    for x in range(binImg.shape[0]):
        for y in range(firstSep):
            if(binImg[x,y] < 10):
                blackPixels += 1
    if(blackPixels < blackPixelsThreshold):
        maskImg[row,firstSep + 1] = 0

def correctArabicWord(arabicWord):
    newWord = ""
    for x in range(len(arabicWord)):
        if(arabicWord[x] == '\u0633' and x != (len(arabicWord) - 1)):#س
            newWord += '\u0649\u0649\u0649'
        elif(arabicWord[x] == '\u0633' and x == (len(arabicWord) - 1)):
            newWord += '\u0649\u0649\u066E'
        elif(arabicWord[x] == '\u0634' and x != (len(arabicWord) - 1)):#ش
            newWord += '\u0649\u062B\u0649'
        elif(arabicWord[x] == '\u0634' and x == (len(arabicWord) - 1)):
            newWord += '\u0649\u062B\u066E'
        elif(arabicWord[x] == '\u0635' and x != (len(arabicWord) - 1)):#ص
            newWord += "صي"
        elif(arabicWord[x] == '\u0635' and x == (len(arabicWord) - 1)):
            newWord += "صٮ"
        elif(arabicWord[x] == '\u0636' and x != (len(arabicWord) - 1)):#ض
            newWord += "ضي"
        elif(arabicWord[x] == '\u0636' and x == (len(arabicWord) - 1)):
            newWord += "ضٮ"
        else:
            newWord += arabicWord[x]
    return newWord

def segment_characters(img):
    thresholdClearAbove(img,230)
    
    # construct a filter for horz line detection
    horz = np.array((
        [2, 6, 2],
        [0, -18, 0],
        [2, 6, 2]), dtype="int")
    
    imgHorz = convolve(img, horz)
    
    thresholdCleared = thresholdClearBelow(imgHorz,250)
    binImage = Binarize(img,138)
    binImageCropped = clearUpperPart(binImage)
    rowBaseLine = GetBaseLine(binImage)    
    MaskRow(thresholdCleared,rowBaseLine)
    detectHoles(binImageCropped,thresholdCleared,rowBaseLine)
    calculateVerticalPixels(binImageCropped,thresholdCleared,rowBaseLine,2)
    AdjustMaskWithRange(thresholdCleared,binImageCropped,rowBaseLine) 
    sepLetters(binImage,thresholdCleared,rowBaseLine)
    removeContinousLines(binImage,thresholdCleared,rowBaseLine)    
    AdjustMask(thresholdCleared,binImage,rowBaseLine)
    removeLastSeperatorIfRedundant(binImage,thresholdCleared,rowBaseLine,6)
    
    letters = createLettersFromImgMask(img,thresholdCleared,rowBaseLine)

    return letters
 


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

path = 'D:\\Faculty\\_Fourth year\\pattern\\Project\\_Final'

os.chdir(path)

for i, infile in enumerate(sorted(glob.glob('*.png'), key=numericalSort)):
    img = cv2.imread(str(infile))
    words_Segmented = wordSegmentation(img)

    OCR_Text = ""

    for word_Segmented in words_Segmented:
        letters_segmented = segment_characters(cv2.cvtColor(word_Segmented, cv2.COLOR_BGR2GRAY))
        
        for letter_segmented in reversed(letters_segmented):
            
            #WALID : place character segmentation here then add it instead of the R
            OCR_Text += "R"
            pass
        
        OCR_Text += " "     #Placing space after a word Don't change ya Walid


    f = open(str(infile)[:-3] + ".txt", "w")
    f.write(OCR_Text)
    f.close()
    

