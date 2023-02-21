import numpy as np
import operator


def createDataSet():
    array = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['a', 'a', 'b', 'b']
    return array, labels


def openFile(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()  ## 获取整个文件,类型是数组
    numberOfLines = len(arrayOfLines)
    returnMat = np.zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        # The strip() method removes any leading (spaces at the beginning) and
        # trailing (spaces at the end) characters (space is the default leading character to remove)

        print(line)
        listFromLine = line.split('\t')
        print(listFromLine[0][0:3])

     #   returnMat[index,:] = listFromLine[0:3]
        index+=1
    return returnMat

mat = openFile("knn.txt")
print(mat)