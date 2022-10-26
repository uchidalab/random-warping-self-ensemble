from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from imp import new_module

import os
import re
# from cv2 import FileNode_NAMED
import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow.python.platform import flags
import numpy as np
import keras
from keras import backend
from keras.layers import Layer
import pandas as pd

from tensorflow.python.ops import array_ops
from tensorflow.python.keras import backend as K
from cleverhans_copy.attacks import FastGradientMethod
from cleverhans_copy.attacks import BasicIterativeMethod
from cleverhans_copy.utils import AccuracyReport
from cleverhans_copy.utils_keras import KerasModelWrapper
from cleverhans_copy.utils_tf import model_eval
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from cleverhans_tutorials import augmentation
from collections import defaultdict

tf.compat.v1.disable_eager_execution()
FLAGS = flags.FLAGS

BATCH_SIZE = 256

class Warp_Layer(Layer):
### default  sigma = 0.5  knot = 4
    def __init__(self,sigma,knot):
        super().__init__()
        self.sigma = sigma
        self.knot = knot
        self.batch_size = BATCH_SIZE

    def call(self,inputs):
        # print('now warp_layer->call')
        return time_warp_tf(inputs,self.sigma,self.knot)

    

def time_warp(x, sigma=0.5, knot=4,batch_size=BATCH_SIZE):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(batch_size, knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    
    ret = np.zeros_like(x)
    # ret = tf.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
            scale = (x.shape[1]-1)/time_warp[-1]
            ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:,dim]).T
    return ret

# @tf.function(input_signature=[tf.TensorSpec(None, tf.float64)])
def time_warp_tf(input,sigma,knot):
    # print('now time')
    ret = tf.numpy_function(time_warp,[input,sigma,knot],tf.float32)
    return ret

def create_warp_model(model,sigma,knot,batch):

    backbone = model
    sigma = sigma
    knot = knot
    batch = batch
    mapping = defaultdict() 

    for i, layer in enumerate(backbone.layers):
        if i == 0: 
            inpt = layer.input  # backboneモデルの下端のテンソル(Input)
            x = layer.input
            out_name = layer.output.name
            mapping[layer.output.name] = x  # モデルの上方でこのレイヤと繋がっている場合はこのテンソルを持ってきて入力する
            continue

        # 元モデルのレイヤーに入力されるテンソルに対応した、改変後モデルにおけるテンソルを持ってくる
        if type(layer.input) is list: # layer.inputは複数入力のときだけlistになっている
            input_tensors = list(map(lambda t: mapping[t.name], layer.input))
        else:
            input_tensors = mapping[layer.input.name]



        out_name = layer.output.name
        # ここで差し替え
        if isinstance(layer, tf.keras.layers.Flatten):
            warp_layer = Warp_Layer(sigma,knot)
            # convert input_tensor to numpy
            # tensor_np = input_tensors.eval(session=tf.compat.v1.Session())
            x = warp_layer(input_tensors)
            # convert numpy to tensor
            # x = tf.convert_to_tensor(x,dtype=tf.float64)
            x = layer(x)
        else:
            # 差し替えの必要がないレイヤーは再利用
            x = layer(input_tensors)
        mapping[out_name] = x
    return tf.keras.Model(inpt, x)

def SE(model,input,num):
    out_list = []
    for _ in range(num):

        out = model.predict(input)
        out_list.append(out)

    return(sum(out_list))

def report_match(y_pred,y_true):
    y_pred_argmax = K.argmax(y_pred)
    y_true_argmax = K.argmax(y_true)


    match = K.equal(y_pred_argmax,y_true_argmax)
    
    with tf.compat.v1.Session():
        match =  match.eval()
    return np.sum(match)

def main(attack_method = 'fgsm'):
    file_path = 'home/yamashita/time_series/model/'
    model = keras.models.load_model(file_path+'best_model.hdf5')
    warp_model = create_warp_model(model,sigma,knot,batch)
    warp_model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy'])

    for i in range(0,len(X),batch_size):
        curr_X = X[i:i+batch_size]
        curr_Y = Y[i:i+batch_size]

        # Define input TF placeholder
        x = tf.compat.v1.placeholder(tf.float32, shape=(None, img_rows, nchannels))
        y = tf.compat.v1.placeholder(tf.float32, shape=(None, nb_classes))

        print("Defined TensorFlow model graph.")

        wrap = KerasModelWrapper(model)

        if attack_method == 'fgsm':  
            # Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
            fgsm = FastGradientMethod(wrap, sess=sess)
            fgsm_params = {'eps': eps }
            adv_x = fgsm.generate(x, **fgsm_params)
        elif attack_method == 'bim':
         
            BasicIterativeMethod
            bim = BasicIterativeMethod(wrap,sess=sess)
            bim_params = {'eps':eps, 'eps_iter':0.05, 'nb_iter':10}
            adv_x = bim.generate(x,**bim_params)
        else:
            print('Either bim or fgsm are acceptable as attack methods')
            
        adv_x = tf.stop_gradient(adv_x)

        adv = adv_x.eval({x: curr_X}, session=sess)


        preds_rwse_adv = SE(w_model,adv,num)
 
        rwse_ata = report_match(preds_rwse_adv,curr_Y)
  
        rwse_adv += rwse_ata

        print(rwse_adv)


