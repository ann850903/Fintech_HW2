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
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from inspect import signature

input_node_num = 30
lr = 0.01
batch_size = 32
epochs = 60
num_classes = 2
validation_split = 0.2

training_filename = 'Data.csv'
hidden_test_set_filename = 'test_no_Class.csv'
output_filename = 'r07943097_answer.txt'

def load_data(file):
    df = pd.read_csv(file)
    if file == training_filename:
        x_train = np.array(df.iloc[:,0:input_node_num].copy())
        y_train = np.array(df.iloc[:,input_node_num].copy())
        return x_train,y_train
    else:
        x_hidden = np.array(df.iloc[:,0:input_node_num].copy())
        return x_hidden

def train_test_split(training_set, validation_split_ratio): 
    training_set_num = int(np.size(training_set, 0)*(1.0-validation_split_ratio))
    training_set_new = training_set[0:training_set_num]
    validation_set = training_set[training_set_num:]
    return training_set_new, validation_set

def get_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(input_node_num,)))
    model.add(Dropout(0.1))
    model.add(Dense(1024, activation='relu'))
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

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    '''
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True',
           xlabel='Predict')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def plot_precision_recall_curve(y_true, y_pred): 
    average_precision = metrics.average_precision_score(y_true, y_pred)
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
    print('AUPRC:', metrics.auc(precision, recall))
    
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('test Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))

def plot_ROC(y_true, y_pred): 
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    print('AUROC:', metrics.auc(fpr, tpr))
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

if __name__ == "__main__":
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
        earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2)
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
    
    ### load hidden test set file
    x_hidden = load_data(hidden_test_set_filename)
    
    ### predict hidden test set
    y_pred_hidden = np.argmax(model.predict(x_hidden), axis=1)
    
    ### output my answer
    with open(output_filename, 'w') as fo: 
        for predict in y_pred_hidden:
            fo.write(str(predict) + "\n")
    
    ### plot_confusion_matrix and calculate precision, recall, f1_score of model
    #### training set
    y_true = np.argmax(y_train, axis=1)
    y_pred = np.argmax(model.predict(x_train), axis=1)
    
    print('Model:')
    plot_confusion_matrix(y_true, y_pred, classes=np.arange(num_classes),
                          title='train_confusion_matrix')
    
    train_precision = metrics.precision_score(y_true, y_pred)
    print('train_precision:', train_precision)
    train_recall = metrics.recall_score(y_true, y_pred)
    print('train_recall:', train_recall)
    train_f1_score = metrics.f1_score(y_true, y_pred)
    print('train_f1_score:', train_f1_score)
    
    #### validation set
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(x_test), axis=1)
    plot_confusion_matrix(y_true, y_pred, classes=np.arange(num_classes),
                          title='validation_confusion_matrix')
    
    test_precision = metrics.precision_score(y_true, y_pred)
    print('test_precision:', test_precision)
    test_recall = metrics.recall_score(y_true, y_pred)
    print('test_recall:', test_recall)
    test_f1_score = metrics.f1_score(y_true, y_pred)
    print('test_f1_score:', test_f1_score)
    
    plot_ROC(y_true, y_pred)
    plot_precision_recall_curve(y_true, y_pred)
    
    #### average score
    print('average_precision:', (train_precision+test_precision)/2)
    print('average_recall:', (train_recall+test_recall)/2)
    print('average_f1_score:', (train_f1_score+test_f1_score)/2)
    plt.show()
    
    ### decision tree
    clf = DecisionTreeClassifier()
    #### training set
    y_true = np.argmax(y_train, axis=1)
    clf = clf.fit(x_train, y_true)
    y_pred = clf.predict(x_train)
    
    print('Decision Tree:')
    train_accuracy = metrics.accuracy_score(y_true, y_pred)
    print('train_accuracy:', train_accuracy)
    train_precision = metrics.precision_score(y_true, y_pred)
    print('train_precision:', train_precision)
    train_recall = metrics.recall_score(y_true, y_pred)
    print('train_recall:', train_recall)
    train_f1_score = metrics.f1_score(y_true, y_pred)
    print('train_f1_score:', train_f1_score)
    
    #### validation set
    y_true = np.argmax(y_test, axis=1)
    y_pred = clf.predict(x_test)
    
    test_accuracy = metrics.accuracy_score(y_true, y_pred)
    print('test_accuracy:', test_accuracy)
    test_precision = metrics.precision_score(y_true, y_pred)
    print('test_precision:', test_precision)
    test_recall = metrics.recall_score(y_true, y_pred)
    print('test_recall:', test_recall)
    test_f1_score = metrics.f1_score(y_true, y_pred)
    print('test_f1_score:', test_f1_score)
    
    #### average score
    print('average_accuracy:', (train_accuracy+test_accuracy)/2)
    print('average_precision:', (train_precision+test_precision)/2)
    print('average_recall:', (train_recall+test_recall)/2)
    print('average_f1_score:', (train_f1_score+test_f1_score)/2)
    
    ### random forest
    clf = RandomForestClassifier(n_estimators=1000,
                                  random_state=0)
    #### training set
    y_true = np.argmax(y_train, axis=1)
    clf = clf.fit(x_train, y_true)
    y_pred = clf.predict(x_train)
    
    print('Random Forest:')
    train_accuracy = metrics.accuracy_score(y_true, y_pred)
    print('train_accuracy:', train_accuracy)
    train_precision = metrics.precision_score(y_true, y_pred)
    print('train_precision:', train_precision)
    train_recall = metrics.recall_score(y_true, y_pred)
    print('train_recall:', train_recall)
    train_f1_score = metrics.f1_score(y_true, y_pred)
    print('train_f1_score:', train_f1_score)
    
    #### validation set
    y_true = np.argmax(y_test, axis=1)
    y_pred = clf.predict(x_test)
    
    test_accuracy = metrics.accuracy_score(y_true, y_pred)
    print('test_accuracy:', test_accuracy)
    test_precision = metrics.precision_score(y_true, y_pred)
    print('test_precision:', test_precision)
    test_recall = metrics.recall_score(y_true, y_pred)
    print('test_recall:', test_recall)
    test_f1_score = metrics.f1_score(y_true, y_pred)
    print('test_f1_score:', test_f1_score)
    
    #### average score
    print('average_accuracy:', (train_accuracy+test_accuracy)/2)
    print('average_precision:', (train_precision+test_precision)/2)
    print('average_recall:', (train_recall+test_recall)/2)
    print('average_f1_score:', (train_f1_score+test_f1_score)/2)
