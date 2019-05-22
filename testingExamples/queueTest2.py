import sys
sys.path.insert(0,'../')
import DataGenerator
import DataQueue

if __name__ == "__main__":
    path = "../ExampleData/catzDataset"
    generator = DataGenerator.CatzGenerator(path, batchSize=16)
    q = DataQueue.DataQueue(generator, size=16, childCount=4)
    q.start()
    for i in range(10):
        batchX, batchY = q.getBatch()
        print(batchX.shape, batchY.shape)
        # here compute something spectacular
    q.stop()
