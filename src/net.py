from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
# Tutorial Link:https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

def run_model(X,y):
    # Step 2) Define Keras Model
    model = Sequential()
    model.add(Dense(15, input_dim=10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='relu'))

    # Step 3) Compile Keras Model --> for loss function minus logarithm of maximum likelihood estimator
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Step 4) Fit Keras Model
    model.fit(X, y, epochs=100, verbose=0)

    # Step 5) Evaluate Keras Model
    loss, accuracy = model.evaluate(X,y)
    print('Loss: ', loss)
    print('Accuracy: %.2f' % (accuracy * 100))
    # Step 6) Make class prediction with the model
    predictions = model.predict(X)

    print(len(predictions))
    # Step 7) Summarize the first 5 cases
    for i in range(50):
        print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))

    #




if __name__ == '__main__':

    # Step 1) Load data
    dataset = loadtxt('master_state_data - Copy_stripped.csv', delimiter=',')

    X = dataset[:, 0:10]

    y_negative = dataset[:, 10]
    y_posative = dataset[:, 11]
    y_death = dataset[:, 12]
    y_totaltestresults = dataset[:, 14]

    run_model(X, y_negative)

    print('done')


