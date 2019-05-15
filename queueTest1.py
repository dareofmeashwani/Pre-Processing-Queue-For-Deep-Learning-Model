import DataQueue
import DataGenerator

if __name__ == "__main__":
    path = "ExampleData/flowerDataset"
    generator = DataGenerator.ClassificationImageGenerator(path, batchSize=64)
    generator.init()
    generator.batchGenerator()
    q = DataQueue.DataQueue(generator, size=16, childCount=4)
    q.start()
    for i in range(10):
        batchX, batchY = q.getBatch()
        print(batchX.shape, batchY.shape)
        # here compute something spectacular
    q.stop()
