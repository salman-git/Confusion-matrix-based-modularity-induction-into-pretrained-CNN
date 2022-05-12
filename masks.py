import numpy as np
import pickle
import matplotlib.pyplot as plt

def generate_trail_masks(shape = (100, 200, 16), sparcity = 0.50):
    '''
    generates 100 masks of size 200x16 for sparcity given as param and save as pickle file on disk.
    100 masks per file
    '''
    total_size = 1
    for s in shape:
        total_size *= s
    masks = np.ones(total_size, dtype=bool)
    indices = np.arange(0, shape[-1])
    masks = masks.reshape(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            np.random.shuffle(indices)
            masks[i, j, indices[:int(shape[-1] * sparcity)]] = False

    trail_masks = open('./masks/trail/t_m{}_{}.pkl'.format(shape, sparcity * 100), 'wb')
    pickle.dump(masks, trail_masks)
    trail_masks.close()
    print('trail masks generated')

def load_trail_masks (shape, sparcity):
    '''
    sparcity: 0-1
    load 100 masks from one file with the given sparcity
    '''
    input_file = open('./masks/trail/t_m{}_{}.pkl'.format(shape, sparcity * 100), 'rb')
    trail_masks = pickle.load(input_file)
    return trail_masks

def save_best_masks(sparcity, mask, acc):
    '''
    saves the best mask out of 100.
    sparcity: sparcity the mask implements
    mask: the best mask
    acc: acc of model for given mask
    all this info is saved to a single pikcle file.
    keys:['acc', 'mask', 'sparcity']
    '''
    input_file = open('./masks/best/best_mask_{}'.format(sparcity * 100), 'wb')
    pickle.dump({'acc': acc, 'sparcity':sparcity, 'mask': mask}, input_file)
    input_file.close()
    print("best mask saved acc:{} sp: {}".format(acc, sparcity))

def load_best_masks(sparcity):
    '''
    loads best mask from disk
    '''
    input_file = open('./masks/best/best_mask_{}'.format(sparcity * 100), 'rb')
    return pickle.load(input_file)

if __name__ == "__main__":
    acc = [99.66]
    sp = [0]
    generate_trail_masks(shape=(100,2, 16))
    