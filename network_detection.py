#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

df = pd.read_csv(r"C:\Users\panda\OneDrive\Desktop\KDDTrain+WithHeader-Labeled.csv")
print(df.head())
print(df['label'].value_counts())


# In[3]:


# Step 2: Preprocess and Encode Features
# Identify categorical features automatically (excluding the target column)
cat_cols = df.select_dtypes(include=['object']).columns.drop('label', errors='ignore')
df = pd.get_dummies(df, columns=cat_cols)

# Extract target
y = df['label'].map({'normal': 0, 'anomaly': 1})
X = df.drop(columns=['label'])


# In[4]:


# Step 3: Train-Test Split


# In[5]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# In[6]:


# Step 4: Train an Anomaly Detection Model


# In[7]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


# In[8]:


# Step 5: Evaulate the Model


# In[9]:


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Network Anomaly Detection')
plt.show()


# In[10]:


# Step 6: Try it Yourself


# In[11]:


from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Feature reduction
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X)

# Train again with reduced features
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reduced, y, test_size=0.3, random_state=42
)
clf.fit(X_train_r, y_train_r)

