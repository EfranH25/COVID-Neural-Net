from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
# Tutorial Link:https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

if __name__ == '__main__':

    # Step 1) Load data
    dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

    X = dataset[:, 0:8]
    y = dataset[:, 8]

    # Step 2) Define Keras Model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Step 3) Compile Keras Model --> for loss function minus logarithm of maximum likelihood estimator
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Step 4) Fit Keras Model
    model.fit(X, y, epochs=1000, batch_size=20)

    # Step 5) Evaluate Keras Model
    loss, accuracy = model.evaluate(X,y)
    print('Loss: ', loss)
    print('Accuracy: %.2f' % (accuracy * 100))
    # Step 6) Make class prediction with the model
    predictions = model.predict(X)

    # Step 7) Summarize the first 5 cases
    for i in range(5):
        print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))

    #
    print('done')