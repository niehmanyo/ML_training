from PIL import Image, ImageDraw
import numpy
import cv2 as cv
import os
import sys

IMAGE_SIZE = (100, 100)


def createDatabase(path):
    """从指定的路径path中处理number个按照顺序命名的人脸图片,成为一个矩阵"""
    imageMatrix = []
    for i in range(1, 21):
        for j in range(1, 6):
            image = Image.open(path + '\\' + str(i) + '\\' + str(j) + '.jpg')
            image = image.resize(IMAGE_SIZE)  # 缩小图片
            grayImage = image.convert('L')
            imageArray = list(grayImage.getdata())  # 转换为一个一维数组，按照行排列
            imageMatrix.append(imageArray)

    imageMatrix = numpy.array(imageMatrix)  # 转换为二维矩阵，行为图像像素值按行排列，没列为一幅图
    # print imageMatix

    return imageMatrix


def eigenfaceCore(Matrix):
    """通过得到的图像矩阵训练特征脸"""
    trainNumber, perTotal = numpy.shape(Matrix)  # 返回图像的个数，和每个图像的大小

    """按照列计算平均向量"""
    meanArray = Matrix.mean(0)  # 0按照列计算平均，1按照行计算平均

    """计算每个向量与平均向量的差"""
    diffMatrix = Matrix - meanArray

    """计算协方差矩阵C的替代L"""
    # diffMatrixTranspose = numpy.transpose(diffMatrix) #矩阵转置
    diffMatrix = numpy.mat(diffMatrix)  # 创建矩阵类型的数据
    L = diffMatrix * diffMatrix.T  # 使乘得的矩阵较小
    eigenvalues, eigenvectors = numpy.linalg.eig(L)  # 特征向量v[:,i]对应特征值w[i]

    """这里得到的特征值和特征向量并无顺序，
        下一部按照特征值大于1来提取特征向量"""
    eigenvectors = list(eigenvectors.T)  # 因为特征向量矩阵的每列是一个特征向量，
    # 所以需要转置后，变为一个list,然后通过pop方法，
    # 删除其中的一行，再逆变换转回去
    for i in range(0, trainNumber):
        if eigenvalues[i] < 1:
            eigenvectors.pop(i)

    eigenvectors = numpy.array(eigenvectors)  # 由于无法直接创建一维的矩阵，所以需要一个数组过度
    eigenvectors = numpy.mat(eigenvectors).T

    """最后计算特征脸,也就是计算出C
        这种变换减少了计算次数"""
    # print numpy.shape(diffMatrix)
    # print numpy.shape(eigenvectors)
    eigenfaces = diffMatrix.T * eigenvectors
    return eigenfaces


def recognize(testIamge, Matrix, eigenface):
    """testIamge,为进行识别的测试图片
    Matrix为所有图片构成的矩阵
    eigenface为特征脸
       返回识别出的文件的行号"""

    """按照列计算平均向量"""
    meanArray = Matrix.mean(0)  # 0按照列计算平均，1按照行计算平均

    """计算每个向量与平均向量的差"""
    diffMatrix = Matrix - meanArray

    """确定经过过滤后的图片数目"""
    perTotal, trainNumber = numpy.shape(eigenface)

    """将每个样本投影到特征空间"""
    projectedImage = eigenface.T * diffMatrix.T

    # print numpy.shape(projectedImage)
    """预处理测试图片，将其映射到特征空间上"""
    testimage = Image.open(testIamge)
    testimage = testimage.resize(IMAGE_SIZE)
    grayTestImage = testimage.convert('L')
    testImageArray = list(grayTestImage.getdata())  # 转换为一个一维数组，按照行排列
    testImageArray = numpy.array(testImageArray)

    differenceTestImage = testImageArray - meanArray
    # 转换为矩阵便于接下来的乘法操作
    differenceTestImage = numpy.array(differenceTestImage)
    differenceTestImage = numpy.mat(differenceTestImage)

    projectedTestImage = eigenface.T * differenceTestImage.T
    # print numpy.shape(projectedImage)
    # print numpy.shape(projectedTestImage)

    """按照欧式距离计算最匹配的人脸"""
    distance = []
    for i in range(0, trainNumber):
        q = projectedImage[:, i]
        temp = numpy.linalg.norm(projectedTestImage - q)  # 计算范数
        distance.append(temp)

    minDistance = min(distance)
    index = distance.index(minDistance)

    return minDistance, index + 1  # 数组index是从0开始的


if __name__ == "__main__":
    Matrix = createDatabase(
        'train_faces',
    )
    eigenface = eigenfaceCore(Matrix)
    count = 0
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv.CascadeClassifier(cascPath)
    faceCascade.load("C:/Users/shang'yin/AppData/Local/Programs/Python/Python310/Lib/site-packages/opencv/data/haarcascades/haarcascade_frontalface_default.xml")
    # 打开视频捕获设备
    video_capture = cv.VideoCapture(0)  ##可改为视频
    while True:
        if not video_capture.isOpened():
            print('Unable to load camera.')
        # 读视频帧
        ret, frame = video_capture.read()
        # 转为灰度图像
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # 调用分类器进行检测
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), )

        # 画矩形框
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 显示视频
        cv.imshow('Video', frame)
        if cv.waitKey(25) & 0xFF == ord('q'):
            img=frame[y - 10: y + h + 10, x - 10: x + w + 10]
            img_new = cv.resize(img, (100,100), interpolation=cv.INTER_CUBIC)
            cv.imwrite('train_faces/1/1.jpg',img_new)
            break

    # 关闭摄像头设备
    video_capture.release()
    # 关闭所有窗口
    cv.destroyAllWindows()
    testimage='train_faces/12/1.jpg'
    distance, index = recognize(testimage, Matrix, eigenface)
    if distance < 40000000:
        print("这是第", index // 5 + 1, "号人脸")
    else:
        print("识别人脸不符", distance)


'''
    for i in range(1, 21):
        testimage = 'D:\\Desktop\\FaceData\\' + str(i) + '\\' + '1.jpg'
        distance, index = recognize(testimage, Matrix, eigenface)
        index=index//4+1
        print(index,distance)
        if index == i and distance < 30000000: count = count + 1

    result = count / 20
    print(count, result * 100, "%")
'''
'''
    unknow1 = 11
    unknow2 = 4
    testimage = 'D:\\Desktop\\FaceData\\' + str(unknow1) + '\\' + str(
        unknow2) + '.jpg'
    distance, index = recognize(testimage, Matrix, eigenface)
    if distance < 40000000 :
        print("这是第", index//5 + 1, "号人脸")
    else:
        print("识别人脸不符", distance)
'''
