import pickle
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.contrib.eager as tfe
import numpy as np
import time
import glob

def build_model(input_shape, input_units, hidden_layers, out_units):
    model = keras.Sequential()
    # model.add(keras.layers.Flatten(input_shape=(14,14,16,)))
    model.add(keras.layers.Conv2D(input_units, 3, strides=(2, 2), input_shape=input_shape[1:], padding='SAME', activation=keras.activations.relu))
    for layer_units in hidden_layers:
        model.add(keras.layers.Conv2D(layer_units, 3, strides=(2, 2), padding='SAME', activation=keras.activations.relu))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=out_units, activation=keras.activations.softmax))
    model.compile(loss=keras.losses.binary_crossentropy,
                optimizer='adam',
                metrics=['accuracy'])
    return model

def load_data(folder):
    activations = []
    labels = []
    print('loading data from ', folder)
    for file in glob.glob('{}/*'.format(folder)):
        #print('loading file ', file)
        x = pickle.load(open(file, 'rb'))
        activations.extend(x['pre_act_sum'].numpy())
        labels.extend(x['labels'].numpy())
    return (np.array(activations), np.array(labels))

def load_mask(mask_file, x_shape, mask_key):    
    y = []
    mask = pickle.load(open(mask_file, 'rb'))[mask_key]
    mask = mask.astype(int)
    for i in range(x_shape):
        y.append(mask)
    y = np.array(y)
    return y

def split_test_train(x, y, train_size=0.8, test_size=0.2):
    train_x = x[:int(len(x)*train_size)]
    test_x = x[int(len(x)*train_size):]
    train_y = y[:int(len(y)*train_size)]
    test_y = y[int(len(y)*train_size):]
    return (train_x, train_y), (test_x, test_y)

def clusterize_labels(y, clusters):
    '''
    y: input labels
    returns index of cluster which the label belongs to
    '''
    for i in range(len(y)):
        if y[i] in clusters[0]:
            y[i] = 0
        else:
            y[i] = 1
        # for j in range(len(clusters)):
        #     if y[i] in clusters[j]:
        #         y[i] = j
    return y
    
def normalize_data (data, NewMin, NewMax):
      
    NewRange = (NewMax - NewMin)  
    
    for j in range(len(data)):
        sample = data[j]
        for i in range(len(sample)):
            sample_min = np.min(sample)
            sample_max = np.max(sample)
            OldRange = (sample_max - sample_min)
            sample[i] = (((sample[i] - sample_min) * NewRange) / OldRange) + NewMin #(sample[i] - sample_min) / (sample_max - sample_min) #normalize to range 0-1
        data[j] = sample
    return data

def run_trail(input_shape, input_units, layer_units,out_units, x_train, y_train, x_test, y_test, x_val, y_val, verbose=0):

    model = build_model(input_shape, input_units, layer_units, out_units)
    conversion_callback = keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, mode='max', min_delta=0.001, restore_best_weights=True)
    tensorboard = keras.callbacks.TensorBoard(log_dir="log")

    history = model.fit(
        np.array(x_train), np.array(y_train),
        epochs=EPOCHS, verbose=1,
        validation_data=(x_val, y_val), callbacks=[conversion_callback])
    eval_result = model.evaluate(np.array(x_val), np.array(y_val), verbose=0)
    print("val acc: " , eval_result[1])
    eval_result = model.evaluate(np.array(x_test), np.array(y_test), verbose=2)
    metrics = model.metrics_names
    for i in range(len(metrics)):
        print('{}:{}\n'.format(metrics[i], eval_result[i]))
    return model
    
if __name__ == "__main__":
    tf.enable_eager_execution()
    mnist_clusters = [[[2, 1, 3, 4, 7, 9], [0, 5, 6, 8]], 
                    [[6, 0, 1, 2], [5, 3], [9, 4, 7, 8]]]

    clusters = mnist_clusters[1]
    
    train_activations_dir = 'Mask_Prediction_v2/3_cluster/train2'
    validation_activations_dir = 'Mask_Prediction_v2/3_cluster/val2'
    test_activations_dir = 'Mask_Prediction_v2/3_cluster/test2'

    (x_train, y_train) = load_data(train_activations_dir)
    (x_test, y_test) = load_data(test_activations_dir)
    (x_val, y_val) = load_data(validation_activations_dir)

    y_train = tf.one_hot(clusterize_labels(y_train.copy(), clusters), len(clusters))
    y_test = tf.one_hot(clusterize_labels(y_test.copy(), clusters), len(clusters))
    y_val = tf.one_hot(clusterize_labels(y_val.copy(), clusters), len(clusters))

    EPOCHS = 100
    # input_units = x_train.shape[-1]
    # input_units = np.shape(x_train)[-1]
    input_shape = np.shape(x_val)
    input_units = 8
    print("input_units, [64, 64]")
    model = run_trail(input_shape, input_units, [64, 64],len(clusters), x_train, y_train, x_test, y_test, x_val, y_val)
    model.save("2_c_cnn_1.h5")
    print("input_units, [64, 128]")
    model = run_trail(input_shape, input_units, [64, 128],len(clusters), x_train, y_train, x_test, y_test, x_val, y_val)
    model.save("2_c_cnn_2.h5")
    print("input_units, [32, 64, 64]")
    model = run_trail(input_shape, input_units, [32, 64, 64],len(clusters), x_train, y_train, x_test, y_test, x_val, y_val)
    model.save("2_c_cnn_3.h5")
    print("input_units, [16, 32, 64, 64]")
    model = run_trail(input_shape, input_units, [16, 32, 64, 64],len(clusters), x_train, y_train, x_test, y_test, x_val, y_val)
    model.save("2_c_cnn_4.h5")
    print("input_units, [64, 64, 120]")
    model = run_trail(input_shape, input_units, [64, 64, 120],len(clusters), x_train, y_train, x_test, y_test, x_val, y_val)
    model.save("2_c_cnn_5.h5")
    print("input_units, [64, 64, 64, 64]")
    model = run_trail(input_shape, input_units, [64, 64, 64, 64], len(clusters),x_train, y_train, x_test, y_test, x_val, y_val)
    model.save("2_c_cnn_6.h5")
    print("input_units, [32, 64, 64, 64, 64, 120]")
    model = run_trail(input_shape, input_units, [32, 64, 64, 64, 64, 120],len(clusters), x_train, y_train, x_test, y_test, x_val, y_val)
    model.save("2_c_cnn_7.h5")
    #to predict output for single instance, use: model.predict(np.array([single_instance,]))
    print('completed')

