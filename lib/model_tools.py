import numpy as np
import matplotlib.pyplot as plt

def roc_plot(proba, y):
    '''
    Action: Build fpr and tpr arrays and plot the ROC
        In: The probability array and target labels from a fit model
       Out: The fpr, tpr arrays and the set of threshholds used
    '''
    n = y.shape[0]
    s = min(n, 1000)
    idx = np.random.randint(n, size=s)
    ps, ys = proba[idx], y[idx]
    thresh = np.argsort(ps)
    fpr, tpr = [], []
    for t in thresh:
        predict = np.array(ps > ps[t]).astype(int)
        fpr.append(np.sum(predict * (1 - ys)) / np.sum(1 - ys))
        tpr.append(np.sum(predict * ys) / np.sum(ys))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], c='gray', linewidth=0.6)
    ax.set(xlabel='False Positive Rate (1 - Specificity)',
           ylabel='True Positive Rate (Sensitivity, Recall)',
           title='ROC Plot')

    return fpr, tpr, ps[thresh]
