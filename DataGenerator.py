import numpy as np
from PIL import Image
import os
import random
import h5py


def read_image(filename):
    img = Image.open(filename)
    arr = np.array(img)
    return arr


def resize_image(img, width=80, height=80):
    img = Image.fromarray(img)
    img = img.resize((width, height), Image.ANTIALIAS)
    return np.array(img)


def normalize(img):
    return img / 255.0


class ClassificationImageGenerator:
    def __init__(self, path, batchSize=64, batchType='random'):
        self.__path = path
        self.batchSize = batchSize
        self.batchType = batchType
        self.__classes = {}
        self.__sequentialIndex = None
        self.__lastPointer = 0
        return

    def init(self):
        metaDataPath = os.path.join(self.__path, 'metadata.txt')
        metaData = None
        if os.path.exists(metaDataPath):
            metaDataReader = open(metaDataPath, 'r')
            metaData = [line.replace('\n', '') for line in metaDataReader.readlines()]
        classIndex = [dir for dir in os.listdir(self.__path) if os.path.isdir(os.path.join(self.__path, dir))]
        for i in classIndex:
            currentClassRoot = os.path.join(self.__path, i)
            currentFilesPath = [os.path.join(currentClassRoot, file) for file in os.listdir(currentClassRoot) if
                                os.path.isfile(os.path.join(currentClassRoot, file))]
            self.__classes[i] = {
                'name': metaData[int(i)] if metaData is not None else i,
                'filesPathList': currentFilesPath,
                'classRootPath': currentClassRoot
            }
        if self.__classes.keys() == 0:
            print('Nothing To Build')
        return

    def batchGenerator(self):
        classCount = len(self.__classes.keys())
        resultX = []
        resultY = []
        if self.batchType.lower() == 'random':
            i = 0
            classIndex = self.__lastPointer % classCount
            while i < self.batchSize:
                itemIndex = random.sample(range(0, len(self.__classes[str(classIndex)]['filesPathList']), ), 1)
                resultX.append(self.__classes[str(classIndex)]['filesPathList'][itemIndex[0]])
                resultY.append(classIndex)
                classIndex = (classIndex + 1) % classCount
                i = i + 1
            self.__lastPointer = classIndex
        elif self.batchType.lower() == 'sequential':
            if self.__sequentialIndex is None:
                self.__sequentialIndex = {}
                for key in self.__classes.keys():
                    self.__sequentialIndex[key] = 0
            i = 0
            classIndex = self.__lastPointer % classCount
            while i < self.batchSize:
                itemIndex = self.__sequentialIndex[str(classIndex)]
                self.__sequentialIndex[str(classIndex)] = (self.__sequentialIndex[str(classIndex)] + 1) % \
                                                          len(self.__classes[str(classIndex)]['filesPathList'])
                resultX.append(self.__classes[str(classIndex)]['filesPathList'][itemIndex])
                resultY.append(classIndex)
                classIndex = (classIndex + 1) % classCount
                i = i + 1
            self.__lastPointer = classIndex
        return resultX, resultY

    def batchProcessor(self, pathX, Y):
        resultX = []
        for x in pathX:
            x = read_image(x)
            x = resize_image(x, 50, 50)
            x = normalize(x)
            resultX.append(x)
        return np.array(resultX), np.array(Y)


class CatzGenerator:
    def __init__(self, path, batchSize=64):
        self.__path = path
        self.batchSize = batchSize

    def init(self):
        self.trainDirectory = [os.path.join(self.__path, dir) for dir in os.listdir(self.__path) if
                               os.path.isdir(os.path.join(self.__path, dir))]

    # compulsary & run seguentially just to create a job
    def batchGenerator(self):
        itemIndex = random.sample(range(0, len(self.trainDirectory)), self.batchSize)
        result = []
        for i in itemIndex:
            result.append(self.trainDirectory[i])
        return result

    # compulsary & run in parallell among multiple Processor, Each processor execute a 1 job
    def batchProcessor(self, paths):
        X = []
        Y = []
        for path in paths:
            temp = []
            path = os.path.join(path, 'cat_')
            for i in range(5):
                temp.append(normalize(read_image(path + str(i) + '.jpg')))
            Y.append(normalize(read_image(path + 'result.jpg')))
            X.append(np.array(temp))
        return np.array(X), np.array(Y)


class CocoGenerator:
    def __init__(self, path, batchSize=64, imageSize=64):
        self.__path = path
        self.batchSize = batchSize
        self.imageSize = imageSize

    def init(self):
        trainFilePath = os.path.join(os.path.join(self.__path, 'imageLists'), 'train.txt')
        self.imagesDir = os.path.join(self.__path, 'images')
        self.annotationsDir = os.path.join(self.__path, 'annotations')
        trainFileReader = open(trainFilePath, 'r')
        self.trainFilesList = [line.replace('\n', '') for line in trainFileReader.readlines()]

    # compulsary & run seguentially just to create a job
    def batchGenerator(self):
        itemIndex = random.sample(range(0, len(self.trainFilesList)), self.batchSize)
        result = []
        for i in itemIndex:
            result.append(self.trainFilesList[i])
        return result

    # compulsary & run in parallell among multiple Processor, Each processor execute a 1 job
    def batchProcessor(self, filenames):
        X = []
        Y = []
        for filename in filenames:
            img = read_image(os.path.join(self.imagesDir, filename + '.jpg'))
            img = resize_image(img, self.imageSize, self.imageSize)
            img = normalize(img)
            if len(img.shape) == 2:
                img = np.pad(img.reshape((self.imageSize, self.imageSize, 1)), [(0, 0), (0, 0), (0, 2)],
                             mode='constant', constant_values=0)
            X.append(img)
            Y.append(h5py.File(os.path.join(self.annotationsDir, filename + '.mat'), 'r'))
        return np.array(X), np.array(Y)
