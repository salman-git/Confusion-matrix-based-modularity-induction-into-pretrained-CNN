from configuration_settings import *
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import Mask_Prediction
from model2 import CNN
import os
import matplotlib.pyplot as plt
import pickle
from masks import *
from BatchGenerator import BatchGenerator
import random
import time
from collections import Counter
from zca import ZCA

def scale(x, min_val=0.0, max_val=255.0):
    # x = tf.to_float(x)
    # return tf.div(tf.subtract(x, min_val), tf.subtract(max_val, min_val))
    return x

def generate_val_ds(x_train, y_train, size=5000, classes=None):
    '''
    extracts validation x and y samples with equal number of classes
    '''
    class_size = int(size / len(classes))
    x_val, y_val = [],[]
    for c in classes:
        indeces = np.arange(len(y_train))
        indxeces = indeces[y_train==c]
        tf.random.shuffle(indeces)
        indeces = indeces[:class_size]
        x_val.extend(x_train[indeces])
        y_val.extend(y_train[indeces])
        x_train = np.delete(x_train, indeces, axis=0)
        y_train = np.delete(y_train, indeces)
    return (x_val, y_val), (x_train, y_train)

def feed_forward(model, x, y, mask=None, prune=False, activations_sub_dir=None, use_mask_pred_model=False, training=False):
    logits = model(x, y, prune=prune, mask=mask, training=training, activations_sub_dir=activations_sub_dir, use_mask_pred_model=use_mask_pred_model)
    return logits

def calculate_loss(logits, y):
    normalized_logits = tf.nn.softmax_cross_entropy_with_logits_v2(
          logits=logits, labels=y)
    return tf.reduce_mean(normalized_logits)
    # return tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=tf.argmax(y, axis=1))

def get_accuracy(logits, y_true):
    predictions = tf.argmax(logits, 1)
    equality = tf.equal(predictions, tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy

def train(model, dataset, val_ds, mask=None, prune=True, best_weights_fname=None, activations_sub_dir=None, augmentation=None):
    if model.total_time and model.conv3_time:
        model.total_time = []
        model.conv3_time = []
    '''
    model: model to train
    mask: sparcity mask
    save: to save checkpoint
    '''
    epoch_accs_vector = [] #accVec
    max_epochs = 100
    best_model = {'weights': None, 'optimizer': None}
    for epoch in range(max_epochs):
        for (batch, (images, labels)) in enumerate(dataset):
            if augmentation is not None:
                (images, labels) = augment_data(images,labels, augmentation)
            loss = None
            with tfe.GradientTape() as tape:
                logits = feed_forward(model, images, labels, mask, prune, activations_sub_dir=activations_sub_dir, training=True)
                if augmentation == "test":
                    logits = get_labels_from_augmented_data(logits)
                loss = calculate_loss(logits, labels)
            grads = tape.gradient(loss, model.trainable_variables)
            
            optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())
            if batch % 10 == 0:
                acc = get_accuracy(logits, labels)
                print("Iteration {}, loss: {:.3f}, train accuracy: {:.2f}%".format(batch, loss, acc*100))
        val_acc = test(model, val_ds, None, augmentation=None)
        print('validation acc for epoch {} : {:.2f}'.format(epoch, val_acc))
        diffs = val_acc - np.array(epoch_accs_vector)
        diffs = diffs[-3:] if len(diffs) > 3 else diffs
        print('learning rate: ', optimizer._lr)
        if len(diffs) > 2 and np.all(diffs < 0.01):
            optimizer._lr = optimizer._lr * 0.1
            if best_model['weights'] is not None:
                model.set_weights(best_model['weights'])
        else:
            best_model['optimizer'] = optimizer
            best_model['weights'] = model.get_weights()

        if optimizer._lr < 1e-6:
            model.set_weights(best_model['weights'])
            print('final val ac: ', test(model, val_ds, None, False, augmentation=None))
            if best_weights_fname is not None:
                pickle.dump(best_model, open(best_weights_fname, 'wb'))
            break

        epoch_accs_vector.append(val_acc)        
    return test(model, val_ds, None, augmentation=None)

def test(model, dataset, mask, prune=False, activations_sub_dir=None, use_mask_pred_model=False, augmentation=None):
    if model.total_time and model.conv3_time:
        model.total_time = []
        model.conv3_time = []
    '''
    model: model to inference
    mask: sparcity mask
    save: it does nothing in this function. it is just to make the train and test function work with AddSparcity function
    '''
    avg_acc = 0
    for (batch, (images, labels)) in enumerate(dataset):
        if augmentation is not None:
            (images, l) = augment_data(images, labels, augmentation)
        logits = feed_forward(model, images, labels,mask=mask, prune=prune, activations_sub_dir=activations_sub_dir, use_mask_pred_model=use_mask_pred_model)
        if augmentation == "test":
            logits = get_labels_from_augmented_data(logits)
        avg_acc += get_accuracy(logits, labels)
        # if augmentation == "test": #don't need this. get_accuracy already return average
        #     avg_acc /= 5
        if batch % 100 == 0 and batch != 0:
            print("Iteration:{}, Average test accuracy: {:.2f}%".format(batch, (avg_acc/(batch+1))*100))
    # batch *= 5
    acc = (avg_acc/(batch + 1)) * 100 #since batch starts from 0 so to avg 1 is added
    print("Final test accuracy for {:.2f}%".format(acc))
    return round(acc.numpy(), 2)

def prune(model, m = None):
    #run a random test before making a call to this function. 
    #otherwise error generates '[layer name] has no attribute 'kernel' ... something like this
    mask =  m if m is not None else pickle.load(open('conv120_mask','rb'))
    model.layers[3].kernel = tf.Variable(tf.boolean_mask(model.layers[3].kernel, mask, axis=3).numpy(), trainable=True)
    model.layers[3].bias = tf.Variable(tf.boolean_mask(model.layers[3].bias, mask).numpy(), trainable=True)

    mask = pickle.load(open('fc_mask_custom', 'rb')) #fc_mask_custom is a random mask
    model.layers[4].kernel = tf.Variable(tf.boolean_mask(model.layers[4].kernel, mask, axis=0).numpy(), trainable=True)
    # model.layers[4].build((2940,84))
    print('layers pruned')

def save_weights(model, index):
    weights = [model.layers[index].weights[0].numpy(), model.layers[index].weights[1].numpy()]
    pickle.dump(weights, open('./layerwise_weights/layer_{}'.format(index), 'wb'))
    print("layer {} weights saved".format(index))

def load_weights(index):
    return pickle.load(open('./layerwise_weights/layer_{}'.format(index), 'rb'))

def augment_data(data, labels=None,  parent_function="train"):
    random.seed(42)
    
    if parent_function == "train":    
        out = np.empty((data.shape[0], 24, 24, data.shape[-1]), dtype='float32') #float32 is the default size of data
        for i in range(len(data)):
            hStIdx = random.randint(0, 8)
            wStIdx = random.randint(0, 8)
            p = data[i, hStIdx : hStIdx + 24, wStIdx : wStIdx + 24]
            rToss = random.random()
            
            if rToss >= 0 and rToss <= 0.5:
                out[i] = p #original image
            elif rToss >= 0.5 and rToss <= 0.75:
                out[i] = np.fliplr(p) #flip lr
            elif rToss >= 0.75 and rToss <= 1.0:
                out[i] = np.flipud(p) #flip ud
        return (out, labels)
    elif parent_function == "test":
        out_test = np.empty((data.shape[0] * 10, 24, 24, data.shape[-1]), dtype="float32")
        out_labels = np.empty((labels.shape[0] * 10, labels.shape[-1]))
        out_index = 0
        for i in range(len(data)):
            out_test[out_index] = data[i, 0:24, 0:24, :] #top left
            out_test[out_index + 1] = data[i, 8:32, 0:24, :] #top right
            out_test[out_index + 2] = data[i, 8:32, 8:32, :] #bottom right
            out_test[out_index + 3] = data[i, 0:24, 8:32, :] #bottom left
            out_test[out_index + 4] = data[i, 4:28, 4:28, :] #center
            out_test[out_index + 5] = np.fliplr(data[i, 8:32, 0:24, :])
            out_test[out_index + 6] = np.fliplr(data[i, 8:32, 0:24, :])
            out_test[out_index + 7] = np.fliplr(data[i, 8:32, 0:24, :])
            out_test[out_index + 8] = np.fliplr(data[i, 8:32, 0:24, :])
            out_test[out_index + 9] = np.fliplr(data[i, 8:32, 0:24, :])
            out_labels[out_index:out_index + 10] = labels[i]
            out_index += 10
        return (out_test, out_labels)

def get_labels_from_augmented_data(logits):
    new_labels = np.zeros((logits.shape[0]//10, logits.shape[-1]))
    labels = tf.argmax(logits, 1)
    out_index = 0
    for i in range(0, len(labels) - 10, 10):
        a = Counter(labels[i:i+10])
        most_frequent = a.most_common(1)[0][0].numpy()
        new_labels[out_index, most_frequent] = 1
        out_index += 1
    return new_labels

def load_data():
    dataset = None
    if model_type == Model_Type.MNIST:
        dataset = tf.keras.datasets.mnist
    elif model_type == Model_Type.CIFAR10:
        dataset = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    
    if model_type == Model_Type.CIFAR10:
        y_test = np.reshape(y_test, (len(y_test)))
        y_train = np.reshape(y_train, (len(y_train)))
    elif model_type == Model_Type.MNIST:
        x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)

    return (x_train, y_train),(x_test, y_test)

def apply_zca(X):
    shape = X.shape
    X = tf.layers.flatten(X)
    trf = ZCA().fit(X)
    X_whitened = trf.transform(X)
    X_reconstructed = trf.inverse_transform(X_whitened)
    # assert(np.allclose(X, X_reconstructed)) # True
    return tf.reshape(X_reconstructed, shape)
    
if __name__=="__main__":

    tf.enable_eager_execution()
    tf.executing_eagerly()
    
    (x_train, y_train), (x_test, y_test) = load_data()
    
    (val_x, val_y), (x_train, y_train) = generate_val_ds(x_train.numpy(), y_train, classes=np.arange(10), size=5000)
    # train_ds = BatchGenerator(scale(x_train), tf.one_hot(y_train, 10), batch_size, mnist_clusters[0])
    # val_ds = BatchGenerator(scale(val_x), tf.one_hot(val_y, 10), batch_size, mnist_clusters[0])
    # test_ds = BatchGenerator(scale(x_test), tf.one_hot(y_test, 10), batch_size, mnist_clusters[0])

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.map(lambda x, y: (scale(x), tf.one_hot(y, 10))).shuffle(55000).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y))
    val_ds = val_ds.map(lambda x, y: (scale(x), tf.one_hot(y, 10))).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(lambda x, y: (scale(x), tf.one_hot(y, 10))).shuffle(10000).batch(batch_size)

    model = CNN(1, model_type=Model_Type.MNIST)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    mask1 = np.ones((2,16), dtype="bool")
    mask1[0, : 8] = False
    mask1[1, 8:] = False
    mask2 = np.ones((3, 120), dtype='bool')
    mask2[0,:60] = False
    mask2[1, 60:] = False
    mask2[2, 30:60] = False
    mask2[2, 60:90] = False
    # mask3 = np.ones((4, 256), dtype='bool')
    # mask3[0, :64] = False
    # mask3[1, 64:128] = False
    # mask3[2, 128:192] = False
    # mask3[3, 192:] = False

    mask = [mask1, None]
    feed_forward(model, np.ones((1, 28, 28, 1), dtype='float32'), np.ones((1, 10), dtype='float32'))
    model.set_weights(pickle.load(open('mnist_pruned_[2_3]_weights_99_83', 'rb')))
    
    s_time = time.time()
    train(model, train_ds, val_ds, mask, True)
    test(model, test_ds, mask, True, use_mask_pred_model=False)
    # e_time = time.time()
    print("completed")
    