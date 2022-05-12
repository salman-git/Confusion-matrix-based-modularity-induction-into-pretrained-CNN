from configuration_settings import Model_Type
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
from CustomConv2D2 import CustomeConv2D
from CustomDense import CustomDense
import time
import pickle
from utils import Mask_Prediction_Model
from Mask_Prediction.mask_pred_model import prepare_data
from configuration_settings import mnist_clusters, cifar10_clusters

class CNN(tf.keras.Model):
    def __init__(self, model, device='cpu:0', mask=None, model_type=Model_Type.MNIST):
        super(CNN, self).__init__()
        if model_type == Model_Type.MNIST:
            self.conv6_3 = CustomeConv2D(0, 6, 3)
            self.conv16_3 = CustomeConv2D(2, 16, 3)
            self.conv120_3 = CustomeConv2D(3, 120, 3) if model == 1 else CustomeConv2D(3, 60, 3)
            self.fc84 = CustomDense(84, activation=tf.nn.relu)
        elif model_type == Model_Type.CIFAR10:
            self.conv128_3 = CustomeConv2D(4, 128, 3)
            self.conv128_3_2 = CustomeConv2D(4, 128, 3)
            self.conv128_3_3 = CustomeConv2D(4, 128, 3)
            self.conv128_3_4 = CustomeConv2D(4, 128, 3)
            self.conv256_3 = CustomeConv2D(4, 256, 3)
            self.conv256_3_2 = CustomeConv2D(4, 256, 3)
            self.fc256 = CustomDense(256, activation=tf.nn.relu)
        elif model_type == Model_Type.CIFAR10_Large:
             # 2 × 128C3 − M P 2 − 2 × 256C3 − M P 2 − 2 × 256C3 − 1 × 512C3 − 1024F C − 1024F C − 10Sof tmax
            self.conv128_3_5 = CustomeConv2D(1, 128, 3)
            self.conv128_3_6 = CustomeConv2D(1, 128, 3)
            self.conv256_3_5 = CustomeConv2D(4, 256, 3)
            self.conv256_3_6 = CustomeConv2D(4, 256, 3)
            self.conv256_3_7 = CustomeConv2D(4, 256, 3)
            self.conv256_3_8 = CustomeConv2D(4, 256, 3)
            self.conv512_3_5 = CustomeConv2D(4, 512, 3)
            self.fc1024_5 = CustomDense(1024, activation=tf.nn.relu)
            self.fc1024_6 = CustomDense(1024, activation=tf.nn.relu)
            
             
        
        self.max_pool2d = tf.layers.MaxPooling2D((3,3), strides=2, padding="same")
        self.dropout = tf.layers.Dropout(0.5)
        self.fc10 = CustomDense(10)
        self.conv3_time = []
        self.total_time = []
        self.activations_parent_dir = './Mask_Prediction'
        self.counter = 0        
        # self.clusters = [[3, 2, 5, 7, 8, 9], [6, 0, 1, 4]] #mnist clusters
        self.c2_mask_pred_model = Mask_Prediction_Model('Mask_Prediction/2_cluster_activations/2_clusters_model.h5')
        self.c3_mask_pred_model = Mask_Prediction_Model('Mask_Prediction/3_cluster_activations/3_clusters_model.h5')
        self.model_type = model_type
        self.device = device
       
    def build(self, input_shape):
        print(input_shape)

    def call(self, x=None, y=None, training=False, mask = None, prune=False, activations_sub_dir=None,
    use_mask_pred_model=False):
        if self.model_type==Model_Type.MNIST:
            x = tf.reshape(x, (-1, 28,28,1))
            s_total_time = time.time()
            x = self.conv6_3(x)
            x = self.max_pool2d(x)
            x, pre_activations = self.conv16_3(x, pre_activations=True)

            if activations_sub_dir is not None:
                pickle.dump({'pre_act_sum': pre_activations, 'labels':tf.argmax(y, axis=1)}, open('{}/{}/{}_{}'.format(self.activations_parent_dir,activations_sub_dir, 'fm',  self.counter),'wb'))
                self.counter += 1
            if prune and mask[0] is not None:
                masked_matrix = self.pruneFM(pre_activations, y, mask[0], clusters=mnist_clusters[0], use_mask_pred_model=use_mask_pred_model)#model is trained on preactivations so mask should be predicted using preactivations
                x = tf.math.multiply(x, masked_matrix)
            x = self.max_pool2d(x)
            s_conv3_time=time.time()
            x, pre_activations = self.conv120_3(x, pre_activations=True)
            # if activations_sub_dir is not None:
            #     pickle.dump({'pre_act_sum': pre_activations, 'labels':tf.argmax(y, axis=1)}, open('{}/{}/{}_{}'.format(self.activations_parent_dir,activations_sub_dir, 'fm',  self.counter),'wb'))
            #     self.counter += 1
            if prune and mask[1] is not None:
                masked_matrix = self.pruneFM(pre_activations, y, mask[1], clusters=mnist_clusters[1], use_mask_pred_model=use_mask_pred_model)#model is trained on preactivations so mask should be predicted using preactivations
                x = tf.math.multiply(x, masked_matrix)
            e_conv3_time = time.time()
            t = e_conv3_time - s_conv3_time
            self.conv3_time.append(t)
            x = tf.layers.flatten(x)
            x = self.fc84(x)
            # if (training):
            #     x = self.dropout(x)
            e_total_time = time.time()
            self.total_time.append(e_total_time - s_total_time) 
            return self.fc10(x)
        # elif self.model_type == Model_Type.CIFAR10:
        #     # 2 × 128C3 − M P 2 − 2 × 128C3 − M P 2 − 2 × 256C3 − 256F C − 10Sof tmax
        #     x = self.conv128_3(x)
        #     # x = self.conv128_3_2(x)
        #     x, pre_activations = self.conv128_3_2(x, pre_activations=True)
        #     if activations_sub_dir[0] is not None:
        #         pickle.dump({'pre_act_sum': pre_activations, 'labels':tf.argmax(y, axis=1)}, open('{}/{}_{}'.format(activations_sub_dir, 'fm',  self.counter),'wb'))
        #         self.counter += 1
        #     if prune and mask[0] is not None:
        #         masked_matrix = self.pruneFM(pre_activations, y, mask[0], clusters=cifar10_clusters[0], use_mask_pred_model=use_mask_pred_model)#model is trained on preactivations so mask should be predicted using preactivations
        #         x = tf.math.multiply(x, masked_matrix)
        #     x = self.max_pool2d(x)
        #     x, pre_activations = self.conv128_3_3(x, pre_activations=True)
        #     if activations_sub_dir[1] is not None:
        #         pickle.dump({'pre_act_sum': pre_activations, 'labels':tf.argmax(y, axis=1)}, open('{}/{}_{}'.format(activations_sub_dir, 'fm',  self.counter),'wb'))
        #         self.counter += 1
        #     if prune and mask[1] is not None:
        #         masked_matrix = self.pruneFM(pre_activations, y, mask[1],clusters=cifar10_clusters[1], use_mask_pred_model=use_mask_pred_model)#model is trained on preactivations so mask should be predicted using preactivations
        #         x = tf.math.multiply(x, masked_matrix)
        #     x = self.conv128_3_4(x)
        #     x = self.max_pool2d(x)
        #     x, pre_activations = self.conv256_3(x, pre_activations=True)
        #     if prune and mask[2] is not None:
        #         masked_matrix = self.pruneFM(pre_activations, y, mask[2],clusters=cifar10_clusters[2], use_mask_pred_model=use_mask_pred_model)#model is trained on preactivations so mask should be predicted using preactivations
        #         x = tf.math.multiply(x, masked_matrix)
        #     # x = self.max_pool2d(x)
        #     x = self.conv256_3_2(x)
        #     x = tf.layers.flatten(x)
        #     x = self.fc256(x)
        #     # if training:
        #     #     x = self.dropout(x)
        #     return self.fc10(x)
        # elif self.model_type == Model_Type.CIFAR10_Large:
        #    # 2 × 128C3 − M P 2 − 2 × 256C3 − M P 2 − 2 × 256C3 − 1 × 512C3 − 1024F C − 1024F C − 10Sof tmax
        #     x= self.conv128_3_5(x)
        #     x = self.conv128_3_6(x)
        #     x = self.max_pool2d(x)
        #     x = self.conv256_3_5(x)
        #     x = self.conv256_3_6(x)
        #     x = self.max_pool2d(x)
        #     x = self.conv256_3_7(x)
        #     x = self.max_pool2d(x)
        #     x = self.conv256_3_8(x)
        #     x = self.max_pool2d(x)
        #     x = self.conv512_3_5(x)
        #     x = tf.layers.flatten(x)
        #     x = self.fc1024_5(x)
        #     if training:
        #         x = self.dropout(x)
        #     x = self.fc1024_6(x)
        #     if training:
        #         x = self.dropout(x)
        #     x = self.fc10(x)
        #     return x

    def pruneFM (self, input, y, mask, clusters=None, axis=0, use_mask_pred_model=False):
        '''
        use this function during training time only. the labels 'y' are required for this function to use.
        input: list of feature maps (output of entire layer)
        mask: mask to apply
        axis: it is similar to appendix in human
        FeatureMaps are pruned after all calculations. Custom layers are not required then.
        If prunning before computation is required then there is prune function in custom layer.
        '''
        input_fm = np.ones(input.shape) #input.numpy()
        indices = np.arange(mask.shape[-1])
        # _mask = mask

        if use_mask_pred_model:
            model = None
            if len(clusters) == 2:
                model = self.c2_mask_pred_model
            elif len(clusters) == 3:
                model = self.c3_mask_pred_model

            mask_indexes = model.predict_mask(prepare_data(input))
            
            for i in range(input.shape[0]):
                masked_indices = indices[mask[mask_indexes[i]]]
                input_fm[i, :, :, masked_indices] = 0                
        elif (mask is not None):
            for i in range(input.shape[0]):
                label = tf.argmax(y[i]).numpy()
                mask_indexes = self.get_mask_index(label, clusters)
                masked_indices = indices[mask[mask_indexes]]
                input_fm[i, :, :, masked_indices] = 0
                
        return input_fm

    def get_mask_index(self, label, clusters):
        for i in range(len(clusters)):
            cluster = clusters[i]
            if label in cluster:
                return i
        return -1
    # def runtimePrune(self, x, mask):
    #     input_fm = x.numpy()
    #     for i in range(len(input_fm)):
    #         indices = np.arange(mask.shape[-1])
    #         masked_indices = indices[mask[i].astype(bool)]
    #         input_fm[i,:,:,masked_indices] *= 0
    #     return input_fm
