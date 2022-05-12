import tensorflow as tf
import numpy as np

class Mask_Prediction_Model:

    def __init__(self, path):
        self.mask_prediction_model = tf.keras.models.load_model(path)

    def predict_mask(self,activations):
        # print('predicting mask index... ')
        return tf.argmax(self.mask_prediction_model.predict(np.array(activations)), axis=1)
    
def visualize_confusion_matrix (confusion_matrix, classes):
    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt

    df_cm = pd.DataFrame(confusion_matrix, index = [i for i in classes],
                columns = [i for i in classes])
    plt.figure(figsize = (10,10))
    sn.heatmap(df_cm, annot=True, fmt='.3g')
    plt.show()

def normalize_confusion_matrix(matrix):
    return [[round(x/sum(r), 5) for x in r] for r in matrix]

def get_confusion_matrix(model, dataset, mask=None):
    preds = np.array([])
    y_true = np.array([])
    for (batch, (images, labels)) in enumerate(dataset):
        logits = model(images, False, mask)
        prediction = tf.argmax(logits, 1)
        preds = np.append(preds,prediction.numpy())
        y_true = np.append(y_true,[tf.argmax(label).numpy() for label in labels])
    # preds = np.reshape(preds, (len(preds)*len(preds[0])))
    # y_true = np.reshape(y_true, (len(y_true) * len(y_true[0])))
    return tf.confusion_matrix(labels=y_true, predictions=preds, num_classes=10).numpy()
