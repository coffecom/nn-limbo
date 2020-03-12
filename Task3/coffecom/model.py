import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization, softmax
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        _, _, input_channels = input_shape
        self.cv1 = ConvolutionalLayer(input_channels, conv1_channels, 3, 1)
        self.a1 = ReLULayer()
        self.pool1 = MaxPoolingLayer(4,4)

        self.cv2 = ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1)
        self.a2 = ReLULayer()
        self.pool2 = MaxPoolingLayer(4,4)

        self.flat = Flattener()
        self.fc = FullyConnectedLayer(4*conv2_channels, n_output_classes)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        for param in self.params():
            self.params()[param].grad = np.zeros_like(self.params()[param].value)

        out = self.pool1.forward(self.a1.forward(self.cv1.forward(X)))
        out = self.pool2.forward(self.a2.forward(self.cv2.forward(out)))
        out = self.fc.forward(self.flat.forward(out))

        loss, grad = softmax_with_cross_entropy(out, y)

        grad = self.flat.backward(self.fc.backward(grad))
        grad = self.cv2.backward(self.a2.backward(self.pool2.backward(grad)))
        grad = self.cv1.backward(self.a1.backward(self.pool1.backward(grad)))

        return loss

    def predict(self, X):
        out = self.pool1.forward(self.a1.forward(self.cv1.forward(X)))
        out = self.pool2.forward(self.a2.forward(self.cv2.forward(out)))
        out = self.fc.forward(self.flat.forward(out))

        return np.argmax(softmax(out), axis=1)

    def params(self):

        result = {'cv1.W': self.cv1.W, 'cv1.B': self.cv2.B,
                'cv2.W': self.cv2.W, 'cv2.B': self.cv2.B,
                'fc.W': self.fc.W, 'fc.B': self.fc.B}

        return result
