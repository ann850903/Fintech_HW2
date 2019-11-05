import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, CSVLogger
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import GridSearchCV

lr = 0.01
batch_size = 32
epochs = 60
num_classes = 2
validation_split = 0.2

training_filename = 'Data.csv'
hidden_test_set_filename = 'test_no_Class.csv'

def load_data(file):
    df = pd.read_csv(file)
    if file == training_filename:
        x_train = np.array(df.iloc[:,0:30].copy())
        y_train = np.array(df.iloc[:,30].copy())
        return x_train,y_train
    else:
        x_hidden = np.array(df.iloc[:,0:30].copy())
        return x_hidden

def train_test_split(training_set, validation_split_ratio): 
    training_set_num = int(np.size(training_set, 0)*(1.0-validation_split_ratio))
    training_set_new = training_set[0:training_set_num]
    validation_set = training_set[training_set_num:]
    return training_set_new, validation_set

def get_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(30,)))
    model.add(Dropout(0.1))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))
    optimizer = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
    return model

def plot_loss_accuracy(file):
    df = pd.read_csv(file)
    loss = np.array(df.loss)
    val_loss = np.array(df.val_loss)
    accuracy = np.array(df.accuracy)
    val_accuracy = np.array(df.val_accuracy)
    
    plt.figure()
    loss, = plt.plot(loss)
    val_loss, = plt.plot(val_loss)
    plt.legend([loss, val_loss], ['loss', 'val_loss'], loc='upper right')
    
    plt.figure()
    accuracy, = plt.plot(accuracy)
    val_accuracy, = plt.plot(val_accuracy)
    plt.legend([accuracy, val_accuracy], ['accuracy', 'val_accuracy'], loc='lower right')
    plt.show()
    
    

### load training file
x_train, y_train = load_data(training_filename)
y_train = to_categorical(y_train, num_classes)
x_train, x_test = train_test_split(x_train, validation_split)
y_train, y_test = train_test_split(y_train, validation_split)

### grid search
'''
model = KerasClassifier(build_fn=get_model, epochs=epochs, batch_size=batch_size, verbose=0)
param_grid = dict()
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(x_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
'''

if not os.path.exists('model_weight.h5'): 
    ### build model
    model = get_model()
    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
    csv_logger = CSVLogger('training.log')
    record = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[earlyStopping, csv_logger], 
                        validation_data=(x_test, y_test))
    ### plot loss and accuracy
    plt.figure()
    plt.plot(record.history['loss'],label='loss')
    plt.plot(record.history['val_loss'],label='val_loss')
    plt.legend(loc='upper right')
    
    plt.figure()
    plt.plot(record.history['accuracy'],label='acc')
    plt.plot(record.history['val_accuracy'],label='val_acc')
    plt.legend(loc='lower right')
    plt.show()
    model.save('model_weight.h5')
else:
    model = load_model('model_weight.h5')
    plot_loss_accuracy('training.log')

### evaluate loss and accuracy
score = model.evaluate(x_train, y_train)
print('Train loss:', score[0])
print('Train accuracy:', score[1])

score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



