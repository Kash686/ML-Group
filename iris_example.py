
# coding: utf-8

# In[42]:

import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# In[2]:

dat=pd.read_csv('data/iris.csv')


# In[41]:

pcaDat=dat[['sepal_length', 'sepal_width', 'petal_length','petal_width']]
pcaClassVec=dat.species
C=list(set(pcaClassVec))
cl=['blue','red','green']

pca=PCA(n_components=2)

pca.fit(pcaDat)

Z=pca.transform(pcaDat)

for i in range(len(C)):
    plt.scatter(Z[np.where(pcaClassVec==C[i]),0],Z[np.where(pcaClassVec==C[i]),1],c=cl[i])

plt.legend(C)
plt.show()


# In[43]:

LDA(pcaDat)


# In[45]:

LDA.fit(pcaDat,pcaClassVec)


# In[ ]:



