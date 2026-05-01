'''
Second Project 

Mohammed Shamasneh		   Nom. : 1220092  		Section: 1
Sari Murad Abdalghani	   Nom.  :1220982		Section: 4

'''


#---------------------------------------------|     Libraries    |---------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#---------------------------------------------------------------------------------------------------------------------------------------------








#-------------------------------------------|   Load and procetss he images  |-----------------------------------------------------------------

def load_images(data_path, img_size=(64,64)):   # fixed dimension = 64×64 
    X, y = [], []
    total = 0 # get total images 
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)

        for fname in os.listdir(label_path):
            fpath = os.path.join(label_path, fname)
            total += 1
            img = Image.open(fpath).convert('RGB')   # convert to RGB format
            img = img.resize(img_size)  #convert to fixed dimension = 64×64 
            X.append(np.array(img).flatten())  # Flatten the image into a 1D array then append to list
            y.append(label)  # name of classification from the name of folder

    print(f"Loaded {total} images from dataset.")
    return np.array(X), np.array(y)

X, y = load_images("dataset/", (64,64)) #calls the load_images function 

le = LabelEncoder()   # to convert the name of classification to numerical value (encoding)
y_encoded = le.fit_transform(y)

#---------------------------------------------------------------------------------------------------------------------------------------------



#--------------------| to train, 75% train and 25% test, the random state to take the same photo at every run |------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25,random_state=42, stratify=y_encoded)


#--------------------------|  calculate the mean and standard deviation for Naive Bayes |-----------------------------------------------------

def stats_features(X):
    means = X.mean(axis=1)
    stds = X.std(axis=1)
    return np.stack([means, stds], axis=1)

X_train_stats = stats_features(X_train)
X_test_stats  = stats_features(X_test)


#----------------------------------------------|   Train & Evaluate Models  |------------------------------------------------------------------

metrics_results = {}  # dictionary to store performance metrics for each model
conf_matrices = {}   # dictionary to store confusion matrices for each model


#--------------------------------|   Naive Bayes (using only mean and standard deviation )  |---------------------------------------------
nb_model = GaussianNB()
nb_model.fit(X_train_stats, y_train) # to train the model
y_pred_nb = nb_model.predict(X_test_stats) # predict on the test set

# Calculate evaluation metrics
acc = accuracy_score(y_test, y_pred_nb) # find accuracy
prec = precision_score(y_test, y_pred_nb, average='macro') # find precision
rec = recall_score(y_test, y_pred_nb, average='macro') # find recall
f1 = f1_score(y_test, y_pred_nb, average='macro') # find F1 
cm = confusion_matrix(y_test, y_pred_nb) # find confusion
metrics_results["Naive Bayes"] = [acc, prec, rec, f1] # store the value 
conf_matrices["Naive Bayes"] = cm  # store the confusion

#---------------------------------------------------------------------------------------------------------------------------------------------


#---------------------------------------|   Decision Tree (using all pixel values)  |--------------------------------------------------------

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)  # Train the full-depth tree

# Create and plot a simplified tree (max depth = 3) for visualization 
dt_model_simple = DecisionTreeClassifier(random_state=42, max_depth=3)
dt_model_simple.fit(X_train, y_train)
plt.figure(figsize=(18,8))
tree.plot_tree(dt_model_simple,filled=True,feature_names=None,    class_names=le.classes_)
plt.title("Decision Tree (max_depth=3)")
plt.show()


# Predict and evaluate the full-depth tree
y_pred_dt = dt_model.predict(X_test)  # Predict on the test set
acc = accuracy_score(y_test, y_pred_dt) # find accuracy
prec = precision_score(y_test, y_pred_dt, average='macro') # find precision
rec = recall_score(y_test, y_pred_dt, average='macro') # find recall
f1 = f1_score(y_test, y_pred_dt, average='macro') # find F1
cm = confusion_matrix(y_test, y_pred_dt) # find confusion
metrics_results["Decision Tree"] = [acc, prec, rec, f1] # store the value 
conf_matrices["Decision Tree"] = cm # store the confusion

#---------------------------------------------------------------------------------------------------------------------------------------------


#---------------------------------------------|   MLP Classifier (all pixels)  |---------------------------------------------------------------

mlp_model = MLPClassifier(hidden_layer_sizes=(32, 128), max_iter=200, random_state=42)
mlp_model.fit(X_train, y_train)    # Train the MLP model
y_pred_mlp = mlp_model.predict(X_test)  # Predict on the test set

acc = accuracy_score(y_test, y_pred_mlp) # find accuracy
prec = precision_score(y_test, y_pred_mlp, average='macro') # find precision
rec = recall_score(y_test, y_pred_mlp, average='macro') # find recall
f1 = f1_score(y_test, y_pred_mlp, average='macro') # find F1
cm = confusion_matrix(y_test, y_pred_mlp) # find confusion
metrics_results["MLP Classifier"] = [acc, prec, rec, f1] # store the value 
conf_matrices["MLP Classifier"] = cm # store the confusion

#---------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------|   Print Table Summary   |---------------------------------------------------------------

print("\n==== Models Metrics Summary ====\n")
print(f"{'Model':<25} {'Accuracy':<9} {'Precision':<10} {'Recall':<8} {'F1-score'}")
for name, vals in metrics_results.items():
    print(f"{name:<25} {vals[0]:<9.2f} {vals[1]:<10.2f} {vals[2]:<8.2f} {vals[3]:.2f}")

#---------------------------------------------------------------------------------------------------------------------------------------------


#---------------------------------------------|   Plot Confusion Matrices  |---------------------------------------------------------------

for name, cm in conf_matrices.items():
    fig, ax = plt.subplots(figsize=(4,4))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title(f"{name} - Confusion Matrix")
    fig.colorbar(cax)
    ax.set_xticklabels([''] + list(le.classes_))
    ax.set_yticklabels([''] + list(le.classes_))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha='center', va='center', color='red')
    plt.show()

    #---------------------------------------------------------------------------------------------------------------------------------------------