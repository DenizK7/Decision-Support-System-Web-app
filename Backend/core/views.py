from django.shortcuts import render
from .models import Files, MyModel
from rest_framework import viewsets
from .serializers import FilesSerializer, MyModelSerializer
from django.http import JsonResponse
from django.views import View
from rest_framework import mixins, viewsets
from rest_framework.response import Response
from rest_framework.views import APIView
from django.conf import settings
import requests
import joblib
from rest_framework.decorators import api_view
import os
import logging
import json
import numpy as np
from joblib import dump, load
import pandas as pd
from scipy.io import arff
from scipy.io.arff import loadarff
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Normalizer, StandardScaler, LabelEncoder
from sklearn.preprocessing import  StandardScaler as scalerr
from django.views.decorators.csrf import csrf_exempt
from django.core.files import File
#ortak olarak hepsinde file okuyor
#eğer 2 dosya geldiyse option 2 döndürsün
#tek dosya ise işlem bittiğinde db'e sonucu yüklesin
#output kısmı bölünmesi gerekebilir.
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

api = 'http://127.0.0.1:8000/api'
endpoint = f'{api}/files/'
logger = logging.getLogger(__name__)
filename="file"
normalizer = Normalizer()
scaler = StandardScaler()
Accuracy = 0
model = RandomForestClassifier()
def randomforest(X,y,test_size,Accuracy,filename,model):
  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train a random forest classifier on the training data
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    if (Accuracy < accuracy):
        # burada accuracye göre model yazılıca
        Accuracy = accuracy
        filename = filename.split(".")[0]
        model = rf
        dump(model, os.path.join(settings.MEDIA_ROOT, filename) + ".model")
    return Accuracy, model



def LOG_Reg(X,y,test_size,Accuracy,filename,model):
    # Split the data into training and testing sets

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train a logistic regression model
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    if (Accuracy < accuracy):
        # burada accuracye göre model yazılıca
        Accuracy = accuracy
        filename = filename.split(".")[0]
        model = clf
        dump(model, os.path.join(settings.MEDIA_ROOT, filename) + ".model")
    return Accuracy, model
def S_V_C(X, y, test_size,Accuracy,filename, model):

    # Split the data into training and testing sets

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train a logistic regression model
    clf = SVC(kernel='linear', C=1, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    if (Accuracy < accuracy):
        # burada accuracye göre model yazılıca
        Accuracy = accuracy
        filename = filename.split(".")[0]
        model = clf
        dump(model, os.path.join(settings.MEDIA_ROOT, filename) + ".model")
       
    return Accuracy ,model
def KNN(X, y,test_size,Accuracy,filename,model):

# Split the data into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    # Create a KNN classifier with k=3try for 1 3 5 7 9
    k_values = [1,3,5,7,9]


    accuracies = []

    # Train a KNN classifier for each value of k and compute its accuracy on the test set
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    # Find the best value of k based on the highest accuracy
    best_k = k_values[np.argmax(accuracies)]
    print("Best k:", best_k)
    print(max(accuracies))


    # Train a final KNN classifier using the best value of k
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)



    if (Accuracy < max(accuracies)):
     # burada accuracye göre model yazılıca
         Accuracy = max(accuracy)
         filename = filename.split(".")[0]
         model = knn
         dump(model, os.path.join(settings.MEDIA_ROOT, filename) + ".model")
        
    return Accuracy ,model
def neural(X,y,test_size,Accuracy,filename,model):

    # Split the data into training and testing sets

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train a logistic regression model
    # Train an ExtraTreesClassifier
    activations = ['logistic', 'tanh', 'relu']

    # Train a neural network classifier for each activation function and record the accuracy
    best_activation = None
    best_accuracy = 0
    for activation in activations:
        clf = MLPClassifier(hidden_layer_sizes=(10,), activation=activation, solver='adam', max_iter=1000,
                            random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_activation = activation
    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = best_accuracy
    if (Accuracy < accuracy):
        # burada accuracye göre model yazılıca
        Accuracy = accuracy
        filename = filename.split(".")[0]
        model = y_pred
        dump(model, os.path.join(settings.MEDIA_ROOT, filename) + ".model")
      
    return Accuracy, model
def adaboost(X,y,test_size,Accuracy,filename, model):
    
    # Split the data into training and testing sets

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train a logistic regression model
    # Train an ExtraTreesClassifier
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    if (Accuracy < accuracy):
        # burada accuracye göre model yazılıca
        Accuracy = accuracy
        filename = filename.split(".")[0]
        model = clf
        dump(model, os.path.join(settings.MEDIA_ROOT, filename) + ".model")
       
    return Accuracy, model
def multilayer( test_size,X,y,Accuracy,filename, model):

  
    # Split the data into training and testing sets

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train a logistic regression model
    # Train an MLP model with lbfgs solver
    clf = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    if (Accuracy < accuracy):
        # burada accuracye göre model yazılıca
        Accuracy = accuracy
        filename = filename.split(".")[0]
        model = clf
        dump(model, os.path.join(settings.MEDIA_ROOT, filename) + ".model")
       
    return Accuracy, model


def naivebayes(df, Accuracy, filename, model):
    # Convert the categorical columns to integers
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.factorize(df[col])[0]

    # Handle missing values
    df = df.fillna(df.mean())

    # Split the data into features and target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Naive Bayes classifier
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = gnb.predict(X_test)

    # Evaluate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    if Accuracy < accuracy:
        # Save the model with the best accuracy
        Accuracy = accuracy
        model = gnb
        dump(model, os.path.join(settings.MEDIA_ROOT, filename.split(".")[0]) + ".model")

    return Accuracy, model
def extratree(test_size,X,y,Accuracy,filename, model):


    # Split the data into training and testing sets

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train a logistic regression model
    # Train an ExtraTreesClassifier
    clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    if (Accuracy < accuracy):
        # burada accuracye göre model yazılıca
        Accuracy = accuracy
        filename = filename.split(".")[0]
        model = clf
        dump(model, os.path.join(settings.MEDIA_ROOT, filename) + ".model")
    return Accuracy, model
option = 0
# Create your views here.

class TrainViewSet(viewsets.ModelViewSet):
    queryset = MyModel.objects.all()
    serializer_class = MyModelSerializer
class FilesViewSet(viewsets.ModelViewSet):
    queryset = Files.objects.all()
    serializer_class = FilesSerializer
class TrainView(APIView):
    def post(self, request, format=None):
        # Handle POST request
        return Response({'message': 'Hello, world!'})
    
class ArffViewSet(viewsets.ModelViewSet):
        queryset = MyModel.objects.all()
        serializer_class = MyModelSerializer
        Accuracy = 0
        
        def list(self, request, *args, **kwargs):
            Accuracy = 0
            mymodels = MyModel.objects.all()
            serializer = MyModelSerializer(mymodels, many=True)
            if len(serializer.data) == 0:
                last_element = None
            else:
             last_element = serializer.data[-1]   # Get the last element of the list
             if last_element is not None:
                filename = os.path.join(settings.MEDIA_ROOT, 'store', 'file', last_element['arff'])
                logger.info("File path: %s", filename)  # Use logger instead of print
                data, meta = arff.loadarff(filename)
                leng = len(data[0])-1
                test_size = int(leng * 0.66)  # number of test samples
                # Convert the data to a pandas DataFrame
                df = pd.DataFrame(data)
                
                data2 = loadarff(filename)
                df2 = pd.DataFrame(data2[0])
                leng = len(data[0])-1
                test_size = int(leng * 0.66)  # number of test samples
                # Convert the class labels to a string datatype
                df[df.columns[-1]] = df[df.columns[-1]].str.decode('utf-8')
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = pd.factorize(df[col])[0]

                # Handle missing values
                df = df.fillna(df.mean())

                # Drop any rows with missing values
                df = df.dropna()

                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]
                Accuracy , model = randomforest(X,y,test_size,Accuracy,filename,model)
                Accuracy,model= extratree(test_size, X,y, Accuracy,filename,model)
                Accuracy,model=  multilayer( test_size, X, y,Accuracy,filename,model)
                Accuracy,model= adaboost(X, y, test_size,Accuracy,filename,model)
                Accuracy,model= neural(X,y, test_size,Accuracy,filename,model)
                Accuracy,model= KNN(X,y, test_size,Accuracy,filename,model)
                Accuracy,model= S_V_C(X,y, test_size,Accuracy,filename,model)
                Accuracy,model=LOG_Reg(X,y, test_size,Accuracy,filename,model)
                Accuracy,model=naivebayes(df2.copy(),Accuracy,filename,model)
                filename = filename.split(".")[0]
               

              
            return Response(serializer.data)
        
def get_metadata(request, filename):
    # Load the ARFF file
    file = os.path.join(settings.MEDIA_ROOT, 'store', 'file', filename)
    data, metadata = loadarff(file)
    df = pd.DataFrame(data)
    # Convert the metadata to a dictionary
    metadata_dict = {}
    for i, key in enumerate(metadata):
        if i < len(df.columns) - 1:
            attribute_type, attribute_options = metadata[key]
            if attribute_options is not None:
                metadata_dict[key] = {'type': attribute_type, 'options': list(attribute_options)}
            else:
                metadata_dict[key] = {'type': attribute_type}

    # Return the metadata as a JSON response
    return JsonResponse(metadata_dict)
    
@csrf_exempt
def predict(request, filename):
    dc = {}
    prediction_list = []
    file = os.path.join(settings.MEDIA_ROOT, 'store', 'file', filename)
    data, metadata = loadarff(file)
    df = pd.DataFrame(data)
    file = os.path.join(settings.MEDIA_ROOT,'store','file', filename.split(".")[0]) + ".model"
    trained_model = joblib.load(file)
    input_values = json.loads(request.POST['input_values'])
    for col in df.columns:
        if df[col].dtype == 'object':
            factorized_col, uniques = pd.factorize(df[col])
            dc[col] = {i: u.decode() for i, u in enumerate(uniques)}
            df[col] = factorized_col
    print(dc)
    
    for key in input_values:
        if metadata[key][0] == 'numeric':
            input_values[key] = float(input_values[key])
        elif metadata[key][0] == 'nominal':
            # Convert nominal value to numeric value using dc dictionary
            nominal_value = input_values[key]
            numeric_value = list(dc[key].keys())[list(dc[key].values()).index(nominal_value)]
            input_values[key] = numeric_value
    
    metadata_list = list(metadata)
      
    


    # Make a prediction

    input_values = np.array(list(input_values.values())).reshape(1, -1)
    prediction = trained_model.predict(input_values)
    prediction_list = prediction.tolist()
    # Check if the last attribute is nominal
    last_column_name = df.columns[-1]
    if last_column_name in dc:
        # Retrieve all values for the last column into a list
        nominal_values = list(dc[last_column_name].values())
        prediction_list[0] = nominal_values[prediction_list[0]]
    response_dict = {'result': prediction_list, 'last_column_name': last_column_name}
    print(response_dict)
    return JsonResponse(response_dict)



@csrf_exempt
def arffa(request, filename):
                Accuracy = 0
                model = RandomForestClassifier()

                filename = os.path.join(settings.MEDIA_ROOT, 'store', 'file', filename+".arff")
                logger.info("File path: %s", filename)  # Use logger instead of print
                data, meta = arff.loadarff(filename)
                leng = len(data[0])-1
                test_size = int(leng * 0.66)  # number of test samples
                # Convert the data to a pandas DataFrame
                df = pd.DataFrame(data)
               # Accuracy, model= randomforest(df.copy(),Accuracy,filename)
                data2 = loadarff(filename)
                df2 = pd.DataFrame(data2[0])
                leng = len(data[0])-1
                test_size = int(leng * 0.66)  # number of test samples
                # Convert the class labels to a string datatype
                df[df.columns[-1]] = df[df.columns[-1]].str.decode('utf-8')
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = pd.factorize(df[col])[0]

                # Handle missing values
                df = df.fillna(df.mean())

                # Drop any rows with missing values
                df = df.dropna()
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]
                Accuracy,model= extratree(test_size, X,y, Accuracy,filename,model)
                Accuracy,model=  multilayer( test_size, X, y,Accuracy,filename,model)
                Accuracy,model= adaboost(X, y, test_size,Accuracy,filename,model)
                Accuracy,model= neural(X,y, test_size,Accuracy,filename,model)
                Accuracy,model= KNN(X,y, test_size,Accuracy,filename,model)
                Accuracy,model= S_V_C(X,y, test_size,Accuracy,filename,model)
                Accuracy,model=LOG_Reg(X,y, test_size,Accuracy,filename,model)
                Accuracy,model=naivebayes(df2.copy(),Accuracy,filename,model)
                filename = filename.split(".")[0]
                dump(model, filename + ".model")

              
                return JsonResponse(Accuracy , safe=False)