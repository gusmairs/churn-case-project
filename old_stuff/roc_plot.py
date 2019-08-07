import numpy as np
import matplotlib.pyplot as plt

def roc(proba, y):
    '''
    Action: Build fpr and tpr arrays and plot the ROC
        In: The probability array and target labels from a fit model
       Out: The fpr, tpr arrays and the set of threshholds used
    '''
    thresh = np.argsort(proba)
    fpr, tpr = [], []
    for t in thresh:
        predict = np.array(proba > proba[t]).astype(int)
        fpr.append(np.sum(predict * (1 - y)) / np.sum(1 - y))
        tpr.append(np.sum(predict * y) / np.sum(y))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], c='gray', linewidth=0.6)
    ax.set(xlabel='False Positive Rate (1 - Specificity)',
           ylabel='True Positive Rate (Sensitivity, Recall)',
           title='ROC Plot')

    return fpr, tpr, proba[thresh]
