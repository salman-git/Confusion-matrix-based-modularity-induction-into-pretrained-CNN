import tensorflow as tf
import numpy as np
class BatchGenerator():
    def __init__(self, data_x, data_y, batch_size, clusters=None, clustered_samples=False):    
        self.clusters = clusters
        self.batch_size = batch_size
        self.offset = 0
        self.none_indexes = []
        self.eoc = True
        self.clustered_samples = clustered_samples
        if (clusters is not None):
            self.data_x, self.data_y = self.rearange(data_x, data_y)
        else:
            self.data_x = data_x
            self.data_y = data_y

    def __next__(self):
        if (self.offset < len(self.data_x)): #and self.eoc):
            batch_samples_x = self.data_x[self.offset:self.offset + self.batch_size]
            batch_samples_y = self.data_y[self.offset:self.offset + self.batch_size]

            if self.clustered_samples and len(self.none_indexes) > 0 and self.offset <= self.none_indexes[0] <= self.offset + self.batch_size:
                none_index = self.none_indexes.pop(0)
                self.none_indexes.append(none_index)
                batch_samples_x = batch_samples_x[0:none_index]
                batch_samples_y = batch_samples_y[0:none_index]
                self.offset += none_index - self.offset + 1
                self.eoc = False
            else:
                self.offset += self.batch_size
            return (tf.cast(batch_samples_x, tf.float32), tf.cast(batch_samples_y, tf.float32))
        # self.eoc = True
        raise StopIteration

    def __iter__(self):
        if self.offset >= len(self.data_x):
            self.offset = 0
        else:
            self.eoc = True
        return self
    
    def rearange(self, data_x, data_y):
        print('rearranging data... ')
        x = data_x
        y = data_y
        x_samples = []
        y_samples = []
        for c in self.clusters:
            for i in range(len(x)):
                if tf.argmax(y[i], 0).numpy() in c:
                    x_samples.append(x[i])
                    y_samples.append(y[i])
            x_samples.append(None)
            y_samples.append(None)
            self.none_indexes.append(len(y_samples) - 1)
        return x_samples, y_samples

    def shuffle(self):
        tf.random.shuffle(self.data_x)
        tf.random.shuffle(self.data_y)

if __name__ == "__main__":
    gn = BatchGenerator(np.array([1,2,3,4,5,6,None, 1,2,3,4,5,6]), np.array(['a','b','c','d','e', 'f', None,'a','b','c','d','e', 'f']), 2, [[2,3,6,5], [1,4]])
    for (gx, gy) in gn:
        print(gx, gy)