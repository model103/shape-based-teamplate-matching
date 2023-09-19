# -*- coding: utf-8 -*-
"""
OpenCV实现边缘模板匹配算法

This is a temporary script file.
"""

import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

class GeoMatch:
    def __init__(self):
        self.noOfCordinates=0   # 坐标数组中元素的个数
        self.cordinates = []   # 坐标数组存储模型点
        self.modelHeight=0   # 模型高
        self.modelWidth=0   # 模型宽
        self.edgeMagnitude = []  # 梯度大小
        self.edgeDerivativeX = []  # 在X方向的梯度
        self.edgeDerivativeY = []  # 在Y方向的梯度
        self.centerOfGravity  = []  # 模板重心
        self.modelDefined=0

    def CreateGeoMatchModel(self, templateArr, maxContrast, minContrast):
        Ssize = []

        src = templateArr.copy()

        # 设置宽和高
        Ssize.append(src.shape[1])  # 宽
        Ssize.append(src.shape[0])  # 高

        self.modelHeight = src.shape[0]  # 存储模板的高
        self.modelWidth = src.shape[1]  # 存储模板的宽

        self.noOfCordinates = 0  # 初始化
        self.cordinates = [] #self.modelWidth * self.modelHeight  # 为模板图像中选定点的联合分配内存

        self.edgeMagnitude = []  # 为选定点的边缘幅度分配内存
        self.edgeDerivativeX = []  # 为选定点的边缘X导数分配内存
        self.edgeDerivativeY = []  # 为选定点的边缘Y导数分配内存

        ## 计算模板的梯度
        gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, 3)

        MaxGradient = -99999.99
        orients = []

        nmsEdges = np.zeros((Ssize[1], Ssize[0]))
        magMat = np.zeros((Ssize[1], Ssize[0]))

        for i in range(1, Ssize[1]-1):
            for j in range(1, Ssize[0]-1):
                fdx = gx[i][j]  # 读x, y的导数值
                fdy = gy[i][j]

                MagG = (float(fdx*fdx) + float(fdy * fdy))**(1/2.0)  # Magnitude = Sqrt(gx^2 +gy^2)
                direction = cv2.fastAtan2(float(fdy), float(fdx))  # Direction = invtan (Gy / Gx)
                magMat[i][j] = MagG

                if MagG > MaxGradient:
                    MaxGradient = MagG  # 获得最大梯度值进行归一化。

                # 从0,45,90,135得到最近的角
                if (direction > 0 and direction < 22.5) or (direction > 157.5 and direction < 202.5) or (direction > 337.5 and direction < 360):
                    direction = 0
                elif (direction > 22.5 and direction < 67.5) or (direction >202.5 and direction <247.5):
                    direction = 45
                elif (direction >67.5 and direction < 112.5) or (direction>247.5 and direction<292.5):
                    direction = 90
                elif (direction >112.5 and direction < 157.5) or (direction>292.5 and direction<337.5):
                    direction = 135
                else:
                    direction = 0

                orients.append(int(direction))

        count = 0 # 初始化count
        # 非最大抑制
        for i in range(1, Ssize[1]-1):
            for j in range(1, Ssize[0] - 1):
                if orients[count] == 0:
                    leftPixel = magMat[i][j- 1]
                    rightPixel = magMat[i][j+1]
                elif orients[count] == 45:
                    leftPixel = magMat[i - 1][j + 1]
                    rightPixel = magMat[i+1][j - 1]
                elif orients[count] == 90:
                    leftPixel = magMat[i - 1][j]
                    rightPixel = magMat[i+1][j]
                elif orients[count] == 135:
                    leftPixel = magMat[i - 1][j-1]
                    rightPixel = magMat[i+1][j+1]

                if (magMat[i][j] < leftPixel) or (magMat[i][j] < rightPixel):
                    nmsEdges[i][j] = 0
                else:
                    nmsEdges[i][j] = int(magMat[i][j]/MaxGradient*255)
                count = count + 1

        RSum = 0
        CSum = 0
        flag = 1
        # 做滞后阈值
        for i in range(1, Ssize[1]-1):
            for j in range(1, Ssize[0]-1):
                fdx = gx[i][j]
                fdy = gy[i][j]
                MagG = (fdx*fdx + fdy*fdy)**(1/2)   # Magnitude = Sqrt(gx^2 +gy^2)
                DirG = cv2.fastAtan2(float(fdy), float(fdx))  # Direction = tan(y/x)

                flag = 1
                if float(nmsEdges[i][j]) < maxContrast:
                    if float(nmsEdges[i][j]) < minContrast:
                        nmsEdges[i][j] = 0
                        flag = 0
                    else: # 如果8个相邻像素中的任何一个不大于maxContrast，则从边缘删除
                        if float(nmsEdges[i-1][j-1]) < maxContrast and \
                            float(nmsEdges[i-1][j]) < maxContrast and \
                            float(nmsEdges[i-1][j+1]) < maxContrast and \
                            float(nmsEdges[i][j-1]) < maxContrast and \
                            float(nmsEdges[i][j+1]) < maxContrast and \
                            float(nmsEdges[i+1][j-1]) < maxContrast and \
                            float(nmsEdges[i+1][j]) < maxContrast and \
                            float(nmsEdges[i+1][j+1]) < maxContrast:
                            nmsEdges[i][j] = 0
                            flag = 0

                # 保存选中的边缘信息
                curX = i
                curY = j
                if(flag != 0):
                    if fdx != 0 or fdy != 0:
                        RSum = RSum+curX  # 重心的行和和列和
                        CSum = CSum+curY

                        self.cordinates.append([curX, curY])
                        self.edgeDerivativeX.append(fdx)
                        self.edgeDerivativeY.append(fdy)

                        # handle divide by zero
                        if MagG != 0:
                            self.edgeMagnitude.append(1/MagG)
                        else:
                            self.edgeMagnitude.append(0)

                        self.noOfCordinates = self.noOfCordinates+1

        self.centerOfGravity.append(RSum//self.noOfCordinates)  # 重心
        self.centerOfGravity.append(CSum // self.noOfCordinates)  # 重心

        # 改变坐标以反映重心
        for m in range(0, self.noOfCordinates):
            temp = 0

            temp = self.cordinates[m][0]
            self.cordinates[m][0] = temp - self.centerOfGravity[0]
            temp = self.cordinates[m][1]
            self.cordinates[m][1] = temp - self.centerOfGravity[1]

        self.modelDefined = True
        return 1

    def FindGeoMatchModel(self, srcarr, minScore, greediness):
        Ssize = []

        Sdx = []
        Sdy = []

        resultScore = 0
        partialSum = 0
        sumOfCoords = 0
        resultPoint = []

        src = srcarr.copy()
        if not self.modelDefined:
            return 0
        Ssize.append(src.shape[1])  # 高
        Ssize.append(src.shape[0])  # 宽

        matGradMag = np.zeros((Ssize[1], Ssize[0]))

        Sdx = cv2.Sobel(src, cv2.CV_32F, 1, 0, 3)  # 找到X导数
        Sdy = cv2.Sobel(src, cv2.CV_32F, 0, 1, 3)  # 找到Y导数

        normMinScore = minScore/ self.noOfCordinates  # 预计算minScore
        normGreediness = ((1- greediness*minScore)/(1-greediness)) / self.noOfCordinates  # 预计算greediness

        for i in range(0, Ssize[1]):
            for j in range(0, Ssize[0]):
                iSx = Sdx[i][j]  # 搜索图像的X梯度
                iSy = Sdy[i][j]  # 搜索图像的Y梯度

                gradMag = ((iSx*iSx)+(iSy*iSy))**(1/2)  # Magnitude = Sqrt(dx^2 +dy^2)

                if gradMag != 0:
                    matGradMag[i][j] = 1/gradMag  # 1/Sqrt(dx^2 +dy^2)
                else:
                    matGradMag[i][j] = 0
        height = Ssize[1]
        wight = Ssize[0]
        Nof = self.noOfCordinates
        for i in range(0, height):
            for j in range(0, wight):
                partialSum = 0  # 初始化partialSum
                for m in range(0, Nof):
                    curX = i + self.cordinates[m][0]  # 模板X坐标
                    curY = j + self.cordinates[m][1]  # 模板Y坐标
                    iTx = self.edgeDerivativeX[m]  # 模板X的导数
                    iTy = self.edgeDerivativeY[m]  # 模板Y的导数

                    if curX < 0 or curY < 0 or curX > Ssize[1] - 1 or curY > Ssize[0] - 1:
                        continue

                    iSx = Sdx[curX][curY]  # 从源图像得到相应的X导数
                    iSy = Sdy[curX][curY]  # 从源图像得到相应的Y导数

                    if (iSx != 0 or iSy != 0) and (iTx != 0 or iTy != 0):
                        # //partial Sum  = Sum of(((Source X derivative* Template X drivative) + Source Y derivative * Template Y derivative)) / Edge magnitude of(Template)* edge magnitude of(Source))
                        partialSum = partialSum + ((iSx*iTx)+(iSy*iTy))*(self.edgeMagnitude[m] * matGradMag[curX][curY])

                    sumOfCoords = m+1
                    partialScore = partialSum/sumOfCoords

                    # 检查终止条件
                    # 如果部分得分小于该位置所需的得分
                    # 在那个坐标中断serching。
                    if partialScore < min((minScore - 1) + normGreediness*sumOfCoords, normMinScore*sumOfCoords):
                        break

                if partialScore > resultScore:
                    resultPoint = []
                    resultScore = partialScore  # 匹配分
                    resultPoint.append(i)  # 结果X坐标
                    resultPoint.append(j)  # 结果Y坐标

        return resultPoint, resultScore

    def DrawContours(self, source, color, lineWidth):
        for i in range(0, self.noOfCordinates):
            point = []
            point.append(self.cordinates[i][1] + self.centerOfGravity[1])
            point.append(self.cordinates[i][0] + self.centerOfGravity[0])
            point = map(int, point)
            point = tuple(point)
            cv2.line(source, point, point, color, lineWidth)

    def DrawSourceContours(self, source, COG, color, lineWidth):
        for i in range(0, self.noOfCordinates):
            point = [0, 0]
            point[1] = self.cordinates[i][0] + COG[0]
            point[0] = self.cordinates[i][1] + COG[1]
            point = map(int, point)
            point = tuple(point)
            cv2.line(source, point, point, color, lineWidth)

if __name__ == '__main__':
    GM = GeoMatch()

    lowThreshold = 10  # deafult value
    highThreashold = 100  # deafult value

    minScore = 0.4  # deafult value
    greediness = 0.8  # deafult value

    total_time = 0  # deafult value
    score = 0  # deafult value

    templateImage = cv2.imread("Template.jpg")  # 读取模板图像
    searchImage = cv2.imread("Search2.jpg")  # 读取待搜索图片

    templateImage = np.uint8(templateImage)
    searchImage = np.uint8(searchImage)

    # ------------------创建基于边缘的模板模型------------------------#
    if templateImage.shape[-1] == 3:
        grayTemplateImg = cv2.cvtColor(templateImage, cv2.COLOR_BGR2GRAY)
    else:
        grayTemplateImg = templateImage.copy()
    print("\nEdge Based Template Matching Program")
    print("--------------------------------------------------------")

    if not GM.CreateGeoMatchModel(grayTemplateImg, lowThreshold, highThreashold):
        print("ERROR: could not create model...")
        assert 0
    GM.DrawContours(templateImage, (255, 0, 0), 1)
    print("Shape model created..with Low Threshold = {} High Threshold = {}".format(lowThreshold, highThreashold))

    # ------------------找到基于边缘的模板模型------------------------#
    # 转换彩色图像为灰色图像。
    if searchImage.shape[-1] == 3:
        graySearchImg = cv2.cvtColor(searchImage, cv2.COLOR_BGR2GRAY)
    else:
        graySearchImg = searchImage.copy()
    print("Finding Shape Model..Minumum Score = {} Greediness = {}".format(minScore, greediness))
    print("--------------------------------------------------------")
    start_time1 = time.time()
    result, score = GM.FindGeoMatchModel(graySearchImg, minScore, greediness)
    print("aaa")
    finish_time1 = time.time()
    total_time = finish_time1 - start_time1

    if score > minScore:
        print("Found at [{}, {}]\nScore =  {} \nSearching Time = {} ms".format(result[0], result[1], score, total_time))
        GM.DrawSourceContours(searchImage, result, (0, 255, 0), 1)
    else:
        print("Object Not found")
    plt.figure("template Image")
    plt.imshow(templateImage)
    plt.figure("search Image")
    plt.imshow(searchImage)
    plt.show()


#太慢了