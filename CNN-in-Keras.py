from tensorflow.keras.callbacks import History
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.engine import training
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation, Average, Input, Flatten, Dense, Concatenate
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.python.framework.ops import Tensor
from typing import Tuple, List
import glob
import numpy as np
import os
import pandas as pd

def load_data() -> Tuple [np.ndarray, np.ndarray, 
                          np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255.
    x_test = x_test / 255.
    
    df = pd.DataFrame(list(zip(x_train, y_train)), columns =['Image', 'label']) 
    val = df.sample(frac=0.01)
    x_train = np.array([ i for i in list(val['Image'])])
    y_train = np.array([ [i[0]] for i in list(val['label'])])
    
    
    df2 = pd.DataFrame(list(zip(x_test, y_test)), columns =['Image', 'label']) 
    val2 = df2.sample(frac=0.01)
    x_test = np.array([ i for i in list(val2['Image'])])
    y_test = np.array([ [i[0]] for i in list(val2['label'])])
    
    
    y_train = to_categorical(y_train, num_classes=10)
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = load_data()

print('x_train shape: {} | y_train shape: {}\nx_test shape : {} | y_test shape : {}'.format(x_train.shape, y_train.shape,x_test.shape, y_test.shape))

input_shape = x_train[0,:,:,:].shape
model_input = Input(shape=input_shape)

def conv_pool_cnn(model_input: Tensor) -> training.Model:
    
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding = 'same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (1, 1), activation='relu')(x)
    x = Conv2D(10, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)
    
    model = Model(model_input, x, name='conv_pool_cnn')
    
    return model

conv_pool_cnn_model = conv_pool_cnn(model_input)

NUM_EPOCHS = 1
def compile_and_train(model: training.Model, num_epochs: int) -> Tuple [History, str]: 
    
    accuracies = []
    losses = []
    model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['acc']) 
    filepath = 'weights/' + model.name + '.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True,
                                                 save_best_only=True, mode='auto', save_freq = 1, period = 1)
    tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=5)
    history = model.fit(x=x_train, y=y_train, batch_size=32, 
                     epochs=num_epochs, verbose=1, callbacks=[checkpoint, tensor_board], validation_split=0.2)
    weight_files = glob.glob(os.path.join(os.getcwd(), 'weights/*'))
    weight_file = max(weight_files, key=os.path.getctime) # most recent file
    return history, weight_file


conv_history, conv_pool_cnn_weight_file = compile_and_train(conv_pool_cnn_model, NUM_EPOCHS)

def evaluate_error(model: training.Model) -> np.float64:
    pred = model.predict(x_test, batch_size = 32)
    pred = np.argmax(pred, axis=1)
    pred = np.expand_dims(pred, axis=1) # make same shape as y_test
    error = np.sum(np.not_equal(pred, y_test)) / y_test.shape[0]   
 
    return error
pool_error = evaluate_error(conv_pool_cnn_model)

def nin_cnn(model_input: Tensor) -> training.Model:
    
    x = Conv2D(64, (3,3), activation = 'relu', kernel_initializer='he_uniform', padding ='same')(model_input)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(128, (3,3), activation = 'relu', kernel_initializer='he_uniform', padding ='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(256, (3,3), activation = 'relu', kernel_initializer='he_uniform', padding ='same')(x)
    x = Conv2D(256, (3,3), activation = 'relu', kernel_initializer='he_uniform', padding ='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(512, (3,3), activation = 'relu', kernel_initializer='he_uniform', padding ='same')(x)
    x = Conv2D(512, (3,3), activation = 'relu', kernel_initializer='he_uniform', padding ='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(512, (3,3), activation = 'relu', kernel_initializer='he_uniform', padding ='same')(x)
    x = Conv2D(512, (3,3), activation = 'relu', kernel_initializer='he_uniform', padding ='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.2)(x)
    
    x = Flatten()(x)
    x = Dense(128, activation = 'relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.2)(x)
    x = Dense(10, activation='softmax')(x)
        
    model = Model(model_input, x, name='nin_cnn')
    
    return model
nin_cnn_model = nin_cnn(model_input)

nin_history, nin_cnn_weight_file = compile_and_train(nin_cnn_model, NUM_EPOCHS)

nin_error = evaluate_error(nin_cnn_model)




def all_cnn(model_input: Tensor) -> training.Model:
    
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding = 'same')(model_input)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding = 'same', strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same', strides = 2)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding = 'same')(x)
    x = Conv2D(192, (1, 1), activation='relu')(x)
    x = Conv2D(10, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)
        
    model = Model(model_input, x, name='all_cnn')
    
    return model
all_cnn_model = all_cnn(model_input)
all_history, all_cnn_weight_file = compile_and_train(all_cnn_model, NUM_EPOCHS)

all_error = evaluate_error(all_cnn_model)

CONV_POOL_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'conv_pool_cnn.hdf5')
ALL_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'all_cnn.hdf5')
NIN_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'weights', 'nin_cnn.hdf5')

conv_pool_cnn_model = conv_pool_cnn(model_input)
all_cnn_model = all_cnn(model_input)
nin_cnn_model = nin_cnn(model_input)

conv_pool_cnn_model.load_weights(CONV_POOL_CNN_WEIGHT_FILE)
all_cnn_model.load_weights(ALL_CNN_WEIGHT_FILE)
nin_cnn_model.load_weights(NIN_CNN_WEIGHT_FILE)

models = [conv_pool_cnn_model, all_cnn_model, nin_cnn_model]

def average(models: List [training.Model], model_input: Tensor) -> training.Model:
    
    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)
    
    model = Model(model_input, y, name='ensemble')
    
    return model

average_ensemble = average(models,model_input)
average_error = evaluate_error(average_ensemble)

def concatenated(models: List [training.Model], model_input: Tensor) -> training.Model:
    
    outputs = [model.outputs[0] for model in models]
    y = Concatenate()(outputs)
    
    model = Model(model_input, y, name='ensemble')
    
    return model
concatenate_ensemble = concatenated(models,model_input)
concatenate_error = evaluate_error(concatenate_ensemble)

