import enum
import os

class Model_Type(enum.Enum):
    CIFAR10 = "cifar10"
    CIFAR10_Large = "cifar10-large"
    MNIST = "MNIST"

model_type = Model_Type.MNIST

checkpoint_directory = "./ckpt"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

train_activations_sub_dir = 'training_activations'
test_activations_sub_dir = 'test_activations'
validation_activations_sub_dir = 'validation_activations'

batch_size = 32
mnist_clusters = [[[2, 1, 3, 4, 7, 9], [0, 5, 6, 8]], 
                    [[6, 0, 1, 2], [5, 3], [9, 4, 7, 8]]] #mnist clusters
                    
cifar10_clusters = [[[8, 0, 1, 9],[3, 2, 4, 5, 6, 7]],
                    [[7, 2, 3, 4, 5], [1, 9], [8, 0, 6]],
                    [[8, 0], [1, 9], [3, 2, 4, 5, 7], [6]]
                    ] #cifar 10 clusters
