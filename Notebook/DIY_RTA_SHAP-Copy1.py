#!/usr/bin/env python
# coding: utf-8

# ## Use this notebook as a playground to try shap code 

# In[1]:


# !pip install shap==0.40.0
# !pip install scikit-learn==1.0.2 --upgrade --user


# In[14]:


import shap #(0.38.0>=)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

## import your model here, on which you are going to use shap
# from sklearn."class" import "model"

import joblib

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[3]:


# Import your train/test data here
df = pd.read_csv("../../Road Traffic Accident Severity Prediction/selected_features.csv")
df


# In[4]:


# Import your model here

model = joblib.load("../../Road Traffic Accident Severity Prediction/random_forest_final.joblib")


# In[4]:


# Import your y_test values here (if available otherwise it's optional)


# In[5]:


shap.__version__


# # SHAP

# 📌 SHAP USES JS IN THE BACKEND, `initjs()` will make the plots render in the notebook

# In[13]:


shap.initjs()


# 1: import a data sample (<500)  
# 2: import your model  
# 3: calculate shap values  
# 4: start plotting  
# 
# 
# 📌 Since SHAP is a computationally heavy library, sampling your data is advised.
# Note that sampling your data doesn't mean neglecting the other observations, it simply means focusing on explaining a small portion of predictions to begin with

# In[7]:


# sample your data before moving further

data_sample = df[:1000]


# In[8]:


shap_value_filename = 'rta_shap100.pkl'

try:
    with open(shap_value_filename, 'rb') as f:
        shap_values = joblib.load(f)
except:
    print("File not found")
    shap_values = shap.TreeExplainer(model).shap_values(data_sample[:100])
    
    # save for future references
    with open(shap_value_filename, "wb") as f:
        joblib.dump(shap_values, f)


# In[9]:


explainer = shap.TreeExplainer(model)
exp = explainer(data_sample)


# In[10]:


y_hat = model.predict(data_sample)


# In[9]:


try:
    with open('shap_interact100.pkl', 'rb') as f:
        shap_values = joblib.load(f)
except:
    print("File not found")
    interaction_values = explainer(data_sample[:100])


# In[10]:


for i, inj in enumerate(y_hat):
    print(i, ":", inj)


# ### Summary Plot
# 
# Summary plot is the total of all the SHAPLEY VALUES for a feature at every observation in the dataset
# 
# Hence it is a GLOBAL Interpretation
# 
# ![Capture.JPG](attachment:Capture.JPG)
# 
# 
# 
# 

# In[11]:


shap.summary_plot(shap_values, data_sample, plot_type="bar")


# In[50]:


fig = plt.figure(figsize=(20,30))

ax0 = fig.add_subplot(131)
ax0.title.set_text('Class 2 - Fatal ')
shap.summary_plot(shap_values[2], data_sample[:100], plot_type="bar", show=False)
ax0.set_xlabel(r'SHAP values', fontsize=11)
plt.subplots_adjust(wspace = 2)

ax1 = fig.add_subplot(132)
ax1.title.set_text('Class 1 - Serious')
shap.summary_plot(shap_values[1], data_sample[:100], plot_type="bar", show=False)
# plt.subplots_adjust(wspace = 2)
ax1.set_xlabel(r'SHAP values', fontsize=11)

ax2 = fig.add_subplot(133)
ax2.title.set_text('Class 0 - Slight')
shap.summary_plot(shap_values[0], data_sample[:100], plot_type="bar", show=False)
ax2.set_xlabel(r'SHAP values', fontsize=11)

# plt.tight_layout(pad=3) # You can also use plt.tight_layout() instead of using plt.subplots_adjust() to add space between plots
plt.show()


# ### Force Plot
# 
# Force plot gives you the interaction between feature and prediction for ONE given observation
# 
# It is a LOCAL Interpretation
# 
# ![Capture.JPG](attachment:Capture.JPG)

# In[12]:


shap.force_plot(shap.TreeExplainer(model).expected_value[0],
                shap_values[0][:], 
                data_sample[:100])


# In[18]:


i=13
print(y_hat[i])
shap.force_plot(shap.TreeExplainer(model).expected_value[0], shap_values[0][i],
                data_sample.values[i], feature_names = data_sample.columns)


# In[20]:


i=90
print(y_hat[i])
shap.force_plot(shap.TreeExplainer(model).expected_value[0], shap_values[0][i],
                data_sample.values[i], feature_names = data_sample.columns)


# In[46]:


i=89
print(y_hat[i])
shap.force_plot(shap.TreeExplainer(model).expected_value[0], shap_values[0][i],
                data_sample.values[i], feature_names = data_sample.columns)


# In[44]:


i=30
print(y_hat[i])
shap.force_plot(shap.TreeExplainer(model).expected_value[0], shap_values[0][i], data_sample.values[i], feature_names = data_sample.columns)


# ### Waterfall Plot
# 
# 
# Waterfall Plot are cascading version of the force plot, where you can clearly see each feature's shap value at a clear height.
# The feature at the top has the highest impact, and the value of impact decreases below.
# 
# 

# In[18]:


row = 90
print(y_hat[row])
shap.waterfall_plot(shap.Explanation(values=shap_values[0][row], 
                                              base_values=shap.TreeExplainer(model).expected_value[0], data=data_sample.iloc[row],  
                                         feature_names=data_sample.columns.tolist()))


# In[15]:


row = 30
print(y_hat[row])
shap.waterfall_plot(shap.Explanation(values=shap_values[0][row], 
                                              base_values=shap.TreeExplainer(model).expected_value[0], data=data_sample.iloc[row],  
                                         feature_names=data_sample.columns.tolist()))


# ### Decision Plot
# 
# 
# * The x-axis represents the model's output. In this case, the units are log odds.  
# * The plot is centered on the x-axis at explainer.expected_value. All SHAP values are relative to the model's expected value like a linear model's effects are relative to the intercept.  
# * The y-axis lists the model's features. By default, the features are ordered by descending importance. The importance is calculated over the observations plotted. _This is usually different than the importance ordering for the entire dataset._ In addition to feature importance ordering, the decision plot also supports hierarchical cluster feature ordering and user-defined feature ordering.
# * Each observation's prediction is represented by a colored line. At the top of the plot, each line strikes the x-axis at its corresponding observation's predicted value. This value determines the color of the line on a spectrum.
# * Moving from the bottom of the plot to the top, SHAP values for each feature are added to the model's base value. This shows how each feature contributes to the overall prediction.
# * At the bottom of the plot, the observations converge at explainer.expected_value.

# In[33]:


row = 30
print(y_hat[row])
shap.decision_plot(shap.TreeExplainer(model).expected_value[0], 
                   shap_values[2][row], 
                   feature_names=data_sample.columns.tolist())


# In[38]:


row = [87,93]
print(y_hat[row[0]:row[1]])
# print(shap_values[2][row[0]:row[1]])
shap.decision_plot(shap.TreeExplainer(model).expected_value[0], 
                   shap_values[2][row[0]:row[1]], 
                   feature_names=data_sample.columns.tolist())


# In[61]:


row_index = 2
shap.multioutput_decision_plot(list(exp.base_values[0]), list(shap_values),
                               row_index=row_index, 
                               feature_names=data_sample.columns.tolist(), 
                               highlight=[np.argmax(y_hat[row_index])],
                               legend_labels=[0,1,2],
                               legend_location='lower right')   


# ### Dependence Plot
# 
# * It is equivalent to a matplotlib dependence plot.
# * The X-axis shows the original value of the feature, y-axis shows the shapley value canculated for that feature.
# * You can see by a dependence plot whether the shapley value increases or decreases, with the change in feature's value.
# * the axis in the right shows value for an additional column that the original feature interacts MOST with. That is automatically generated.

# In[18]:


shap.dependence_plot('day_of_week', shap_values[2], data_sample[:100])


# In[20]:


shap.dependence_plot('vehicles_involved', shap_values[2], data_sample[:100])


# In[21]:


shap.dependence_plot('driving_experience', shap_values[2], data_sample[:100])


# ### Heatmaps
# 
# This heatmap contains much information. First, the importance of the variables is labeled on the left side. The horizontal bars on the right side rank the variables from the most important to the least important. The model variable importance represents global interpretability.
# 
# 
# <img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*i6bh5eCIp-kAYWj6WbFZTw.png" width="500" />

# In[63]:


shap.plots.heatmap(exp[0] + exp[1] + exp[2], max_display=15)


# In[41]:




