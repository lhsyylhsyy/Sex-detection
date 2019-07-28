# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 10:09:04 2019

@author: Ning
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score

#%% logistic regression
model1 = LogisticRegression(random_state = 10).fit(train_x, train_y)
#%% Model 1 assessment
print("training accuracy:", model1.score(train_x, train_y))
print("testing accuracy:", model1.score(test_x, test_y))
# training accuracy: 0.9996446339729922
# testing accuracy: 0.5855130784708249

yhat_test = model1.predict(test_x)
mat = confusion_matrix(yhat_test, test_y)
print(mat)
#                 women  men  true
# women predicted [[185  93]
# men predicted    [113 106]] 

prec = precision_score(test_y, yhat_test)
rec = recall_score(test_y, yhat_test)

print('precision = {}, recall = {}'.format(prec, rec))
# precision = 0.4840182648401826, recall = 0.5326633165829145

# Basically doesn't work

#%% Visualize coefficients
import seaborn 
def plot_logistic_coefs(coef):
    max_coef = np.max(coef)
    min_coef = np.min(coef)
    cmaps = ['Reds', 'Greens', 'Blues']
    plt.figure(figsize = (20, 5))
    for i in range(3):
        plt.subplot(1,3,i+1)
        seaborn.heatmap(coef_mat[:,:,i], square = False, center = 0,
                        vmin = min_coef, vmax = max_coef, cmap = cmaps[i])

coef_mat = model1.coef_.reshape(num_px, num_px, 3)
plot_logistic_coefs(coef_mat)


#%% Logistic regression with L1 penalty
model2 = LogisticRegression(penalty = 'l1', 
                            random_state = 10).fit(train_x, train_y)

print("training accuracy:", model2.score(train_x, train_y))
print("testing accuracy:", model2.score(test_x, test_y))
# training accuracy: 0.9861407249466951
# testing accuracy: 0.6297786720321932

plot_logistic_coefs(model2.coef_.reshape(num_px, num_px, 3))