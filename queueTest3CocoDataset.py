import DataGenerator
import DataQueue

if __name__ == "__main__":
    path = "F:\coco"
    generator = DataGenerator.CocoGenerator(path, batchSize=64)
    q = DataQueue.DataQueue(generator, size=16, childCount=4)
    q.start()
    for i in range(10):
        batchX, batchY = q.getBatch()
        print(batchX.shape, batchY.shape)
        # here compute something spectacular
    q.stop()
