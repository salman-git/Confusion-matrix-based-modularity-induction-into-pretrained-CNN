import pickle
import numpy as np 
import matplotlib.pyplot as plt
from  matplotlib.colors import Normalize
import tensorflow as tf

if __name__ == "__main__":
    # x = pickle.load(open('16_featuremaps_10_classes/unpruned', 'rb'))
    # x2 = pickle.load(open('16_featuremaps_10_classes/pruned', 'rb'))
    ix = 0
    plt.tight_layout()
    # gd = plt.GridSpec(10, 16)
    
    # for j in range(0, 10, 2):
    #     for i in range(16):
    #         ax = plt.subplot(gd[j, i])
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         plt.imshow(x[ix,:,:,i], cmap='gray')
    #         ax = plt.subplot(gd[j+1, i])
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         plt.imshow(x2[ix, :,:,i], cmap='gray')
    #     ix = ix + 1
    
    '''visualize weights'''

    x = pickle.load(open('unpruned_best_model_dic', 'rb'))['weights'][2]
    normalize = Normalize(vmin=0, vmax=1)
    gd = plt.GridSpec(6, 16)
    for j in range(6):
        for i in range(16):
            ax = plt.subplot(gd[j, i])
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(x[:,:,j,i], cmap='gray', norm=normalize)
            
        ix = ix + 1
    
    plt.show()
    print("END")
