# Pre-Processing Queue For Deep Learning Model

We have implemented a data pre processing Queue for Machine learning/Deep Learning Model which allows the us manage resources(Ram/Cpu) efficiently & exploit the multi processors/Cores environment.

we just have to write the generator class with two mandatory method & its ready to go.
we just have decide how we can divide the reading of data(text/image/audio) in sub-problems, this will go into batchGenerator method. Each sub-problem Compute individual on some Processor & it will result finally stored in Queue 

Check out the DataGeneraator.py for generator class example
Feel free to Edit