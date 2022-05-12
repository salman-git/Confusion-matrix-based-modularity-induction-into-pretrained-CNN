import pickle
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.contrib.eager as tfe
import numpy as np
import time
import glob

def build_model(input_shape, hidden_layers, out_units):
    model = keras.Sequential()
    # model.add(keras.layers.Flatten(input_shape=(14,14,16,)))
    model.add(keras.layers.Dense(units=input_shape, input_shape=(input_shape,)))
    for layer_units in hidden_layers:
        model.add(keras.layers.Dense(units=layer_units, activation=keras.activations.relu))
    model.add(keras.layers.Dense(units=out_units, activation=keras.activations.softmax))
    model.compile(loss=keras.losses.binary_crossentropy,
                optimizer='adam',
                metrics=['accuracy'])
    return model

def load_data(folder):
    activations = []
    labels = []
    for file in glob.glob('{}/*'.format(folder)):
        print('loading file ', file)
        x = pickle.load(open(file, 'rb'))
        activations.extend(x['pre_act_sum'].numpy())
        labels.extend(x['labels'].numpy())
    return (activations, labels)

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
        for j in range(len(clusters)):
            if y[i] in clusters[j]:
                y[i] = j
                break
        # if y[i] in clusters[0]:
        #     y[i] = 0
        # else:
        #     y[i] = 1
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

def run_trail(input_units, layer_units, out_units, x_train, y_train, x_test, y_test, x_val, y_val, verbose=0):

    model = build_model(input_units, layer_units, out_units)
    conversion_callback = keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, mode='max', min_delta=0.0001, restore_best_weights=True)
    tensorboard = keras.callbacks.TensorBoard(log_dir="log")
    history = model.fit(
        np.array(x_train), np.array(y_train),
        epochs=EPOCHS, verbose=verbose,
        validation_data=(x_val, y_val),
        shuffle=True,
        callbacks=[conversion_callback])

    eval_result = model.evaluate(np.array(x_test), np.array(y_test), verbose=2)
    metrics = model.metrics_names
    for i in range(len(metrics)):
        print('{}:{}\n'.format(metrics[i], eval_result[i]))
    return model

def hist(data, bins):
    data = normalize_data(data, -1, 1)
    out_array = np.zeros((np.shape(data)[0], bins))
    for i in range(len(out_array)):
        out_array[i] = np.histogram(data[i], bins)[0]
    return out_array

def sum(data, segments, axis):
    data = np.array(data)
    fm_len_x = int(data.shape[axis[0]]/segments[0])
    fm_len_y = int(data.shape[axis[1]]/segments[1])
    out_array = np.ones([data.shape[0], fm_len_x, fm_len_y, data.shape[-1]])
    
    for m in range(data.shape[0]):
        print('calculating sample: {}/{}'.format(m, data.shape[0]))
        for k in range(data.shape[-1]):
            start_x = 0
            for i in range(0, fm_len_x):
                start_y = 0
                for j in range(0, fm_len_y):
                    out_array[m, i, j, k] = tf.math.reduce_sum(data[m, start_x : start_x + segments[0], start_y: start_y +  segments[1], k])
                    start_y = start_y + segments[1]
                start_x = start_x + segments[0]
    return out_array

def prepare_data(data):
    data = sum(data, [2,2], [1,2])
    # data = tf.math.reduce_sum(data, axis=[-1])
    data = normalize_data(data.copy(), -1, 1)
    data = np.reshape(data, (data.shape[0], -1))
    return data
    
if __name__ == "__main__":
    tf.enable_eager_execution()
    mnist_clusters = [[[2, 1, 3, 4, 7, 9], [0, 5, 6, 8]], 
                    [[6, 0, 1, 2], [5, 3], [9, 4, 7, 8]]] #mnist clusters
    clusters = mnist_clusters[0]
    
    train_activations_dir = 'Mask_Prediction/train_activations'
    validation_activations_dir = 'Mask_Prediction/validation_activations'
    test_activations_dir = 'Mask_Prediction/test_activations'

    (x_train, y_train) = load_data(train_activations_dir)
    (x_test, y_test) = load_data(test_activations_dir)
    (x_val, y_val) = load_data(validation_activations_dir)
    # x_train = pickle.load(open('train_activations', 'rb'))
    # x_val = pickle.load(open('validation_activations', 'rb'))
    # x_test = pickle.load(open('test_activations', 'rb'))

    x_train = prepare_data(x_train)
    x_test = prepare_data(x_test)
    x_val = prepare_data(x_val)
    pickle.dump(x_train, open('train_sum', 'wb'))
    pickle.dump(x_test, open('test_sum', 'wb'))
    pickle.dump(x_val, open('validation_sum', 'wb'))

    y_train = tf.one_hot(clusterize_labels(y_train.copy(), clusters), len(clusters))
    y_test = tf.one_hot(clusterize_labels(y_test.copy(), clusters), len(clusters))
    y_val = tf.one_hot(clusterize_labels(y_val.copy(), clusters), len(clusters))

    EPOCHS = 100
    input_units = x_val.shape[-1]
    out_units = len(clusters)
    model = run_trail(input_units, [64, 64], out_units, x_train, y_train, x_test, y_test, x_val, y_val)
    model = run_trail(input_units, [64, 128], out_units, x_train, y_train, x_test, y_test, x_val, y_val)
    model = run_trail(input_units, [32, 64, 64],  out_units,x_train, y_train, x_test, y_test, x_val, y_val)
    model = run_trail(input_units, [16, 32, 64, 64], out_units, x_train, y_train, x_test, y_test, x_val, y_val)
    model = run_trail(input_units, [64, 64, 120], out_units, x_train, y_train, x_test, y_test, x_val, y_val)
    model = run_trail(input_units, [64, 64, 64, 64], out_units, x_train, y_train, x_test, y_test, x_val, y_val)
    model = run_trail(input_units, [32, 64, 64, 64, 64, 120], out_units, x_train, y_train, x_test, y_test, x_val, y_val)
    # model.save('Mask_Prediction/best_model.h5')
    #to predict output for single instance, use: model.predict(np.array([single_instance,]))
    print('completed')


