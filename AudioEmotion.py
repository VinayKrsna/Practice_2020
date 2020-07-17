import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from collections import Counter
import matplotlib.pyplot as plt
import scipy.io.wavfile
import sys


# DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        # Читаем файл
        X = sound_file.read(dtype="float32")
        # Находим частоту дискретизации входного аудио
        sample_rate = sound_file.samplerate
        if chroma:
            #Коротковременное преобразование фурье, используется
            # для определения частоты синусоидального и фазового содержания
            # локальных сечений сигнала , как он меняется с течением времени.
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            # Вычисляем среднее арифметическое MFCC,
            # что я вляется для меня входным признаком
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            #Соединяем массивы по горизонтали
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))

    if mel:
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
    return result

#DataFlair - Emotions in the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}
#DataFlair - Emotions to observe
observed_emotions_4 = ['neutral', 'calm', 'happy', 'sad']

observed_emotions_8 = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']


# DataFlair - Load the data and extract features for each sound file
def load_data(test_size=0.2, pattern="", emotions_list = []):
    x, y = [], []
    for file in glob.glob(pattern):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in emotions_list:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

# DataFlair - Load the data and extract features for each sound file
def load_data_test_rus(pattern=""):
    x, y = [], []
    for file in glob.glob(pattern):
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
    return np.array(x), len(glob.glob(pattern))

print("----------------------------------------------------------------------------------------------")
print("---------------------------TRAINING AND TESTING 4 EMOTIONS------------------------------------")
print("----------------------------------------------------------------------------------------------")

#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=load_data(test_size=0.3,
                                        pattern="C:\\Uni\\Practice\\ravdess_data\\Actor_*\\*.wav",
                                        emotions_list=observed_emotions_4)

# DataFlair - Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))

# DataFlair - Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')

# DataFlair - Initialize the Multi Layer Perceptron Classifier
model = MLPClassifier(alpha=0.01,
                      batch_size=256,
                      epsilon=1e-08,
                      hidden_layer_sizes=(300,),
                      learning_rate='adaptive',
                      max_iter=500)
#
# DataFlair - Train the model
model.fit(x_train, y_train)

# DataFlair - Predict for the test set
y_pred = model.predict(x_test)
print(f'\n y_pred = {dict(Counter(y_pred))} \n')

# DataFlair - Calculate the accuracy of our model
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
# DataFlair - Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy * 100))

print("----------------------------------------------------------------------------------------------")
print("-------------------------------RADIO TEST 4 EMOTIONS------------------------------------------")
print("----------------------------------------------------------------------------------------------")

x_test_rus, rus_count = load_data_test_rus(pattern="C:\\Uni\\Practice\\youtube\\Radio_*\\*\\*.wav")
y_pred = dict(Counter(model.predict(x_test_rus)))
print(f'\n y_pred = {y_pred} \n')
for key in y_pred:
    print(f'{key} : {(y_pred[key]/rus_count)*100} %')

print("----------------------------------------------------------------------------------------------")
print("---------------------------------MALE TEST 4 EMOTIONS------------------------------------------")
print("----------------------------------------------------------------------------------------------")

#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=load_data(test_size=0.3,
                                        pattern="C:\\Uni\\Practice\\racdess_data_male_female\\Male\\Actor_*\\*.wav",
                                        emotions_list=observed_emotions_4)

# DataFlair - Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))

# DataFlair - Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')

# DataFlair - Initialize the Multi Layer Perceptron Classifier
model = MLPClassifier(alpha=0.01, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive',
                      max_iter=500)
#
# DataFlair - Train the model
model.fit(x_train, y_train)

# DataFlair - Predict for the test set
y_pred = model.predict(x_test)
print(f'\n y_pred = {Counter(y_pred)} \n')

# DataFlair - Calculate the accuracy of our model
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
# DataFlair - Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy * 100))

print("----------------------------------------------------------------------------------------------")
print("-------------------------------FEMALE TEST 4 EMOTIONS-----------------------------------------")
print("----------------------------------------------------------------------------------------------")
#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=load_data(test_size=0.3,
                                        pattern="C:\\Uni\\Practice\\racdess_data_male_female\\Female\\Actor_*\\*.wav",
                                        emotions_list=observed_emotions_4)

# DataFlair - Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))

# DataFlair - Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')

# DataFlair - Initialize the Multi Layer Perceptron Classifier
model = MLPClassifier(alpha=0.01, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive',
                      max_iter=500)
#
# DataFlair - Train the model
model.fit(x_train, y_train)

# DataFlair - Predict for the test set
y_pred = model.predict(x_test)
print(f'\n y_pred = {Counter(y_pred)} \n')

# DataFlair - Calculate the accuracy of our model
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
# DataFlair - Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy * 100))

print("----------------------------------------------------------------------------------------------")
print("---------------------------TRAINING AND TESTING 8 EMOTIONS------------------------------------")
print("----------------------------------------------------------------------------------------------")

# DataFlair - Split the dataset
x_train, x_test, y_train, y_test = load_data(test_size=0.3,
                                             pattern="C:\\Uni\\Practice\\ravdess_data\\Actor_*\\*.wav",
                                             emotions_list=observed_emotions_8)

# DataFlair - Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))

# DataFlair - Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')

# DataFlair - Initialize the Multi Layer Perceptron Classifier
model = MLPClassifier(alpha=0.01,
                      batch_size=256,
                      epsilon=1e-08,
                      hidden_layer_sizes=(300,),
                      learning_rate='adaptive',
                      max_iter=500)
#
# DataFlair - Train the model
model.fit(x_train, y_train)

# DataFlair - Predict for the test set
y_pred = model.predict(x_test)
print(f'\n y_pred = {dict(Counter(y_pred))} \n')

# DataFlair - Calculate the accuracy of our model
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
# DataFlair - Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy * 100))

print("----------------------------------------------------------------------------------------------")
print("-------------------------------RADIO TEST 8 EMOTIONS------------------------------------------")
print("----------------------------------------------------------------------------------------------")

x_test_rus, rus_count = load_data_test_rus(pattern="C:\\Uni\\Practice\\youtube\\Radio_*\\*\\*.wav")
y_pred = dict(Counter(model.predict(x_test_rus)))
print(f'\n y_pred = {y_pred} \n')
for key in y_pred:
    print(f'{key} : {(y_pred[key] / rus_count)*100} %')

print("----------------------------------------------------------------------------------------------")
print("---------------------------------MALE TEST 8 EMOTIONS------------------------------------------")
print("----------------------------------------------------------------------------------------------")

# DataFlair - Split the dataset
x_train, x_test, y_train, y_test = load_data(test_size=0.3,
                                             pattern="C:\\Uni\\Practice\\racdess_data_male_female\\Male\\Actor_*\\*.wav",
                                             emotions_list=observed_emotions_8)

# DataFlair - Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))

# DataFlair - Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')

# DataFlair - Initialize the Multi Layer Perceptron Classifier
model = MLPClassifier(alpha=0.01, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive',
                      max_iter=500)
#
# DataFlair - Train the model
model.fit(x_train, y_train)

# DataFlair - Predict for the test set
y_pred = model.predict(x_test)
print(f'\n y_pred = {Counter(y_pred)} \n')

# DataFlair - Calculate the accuracy of our model
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
# DataFlair - Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy * 100))

print("----------------------------------------------------------------------------------------------")
print("-------------------------------FEMALE TEST 8 EMOTIONS-----------------------------------------")
print("----------------------------------------------------------------------------------------------")
# DataFlair - Split the dataset
x_train, x_test, y_train, y_test = load_data(test_size=0.3,
                                             pattern="C:\\Uni\\Practice\\racdess_data_male_female\\Female\\Actor_*\\*.wav",
                                             emotions_list=observed_emotions_8)

# DataFlair - Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))

# DataFlair - Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')

# DataFlair - Initialize the Multi Layer Perceptron Classifier
model = MLPClassifier(alpha=0.01, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive',
                      max_iter=500)
#
# DataFlair - Train the model
model.fit(x_train, y_train)

# DataFlair - Predict for the test set
y_pred = model.predict(x_test)
print(f'\n y_pred = {Counter(y_pred)} \n')

# DataFlair - Calculate the accuracy of our model
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
# DataFlair - Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy * 100))
