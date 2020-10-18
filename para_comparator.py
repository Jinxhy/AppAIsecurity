"""
Written by Yujin Huang(Jinx)
Started 24/07/2020 10:17 pm
Last Editted 

Description of the purpose of the code
"""

import os
import tensorflow as tf
import re
from tqdm import tqdm
from fuzzywuzzy import fuzz
import numpy as np
import itertools


class Model:
    def __init__(self, name, layers):
        self.name = name
        self.layers = layers


class Layer:
    def __init__(self, weight_bias):
        self.detail = weight_bias


def load_db():
    db = []

    for model_name in os.listdir("DL_models/v2/"):

        try:
            # Load TFLite model and allocate tensors.
            interpreter = tf.lite.Interpreter("DL_models/v2/" + model_name)
            interpreter.allocate_tensors()
        except:
            print(model_name, "loading error")
            continue

        # Get tensor details.
        details = interpreter.get_tensor_details()

        # init pre-trained model
        layers = []
        model = Model(model_name, layers)

        # MobileNets,
        for detail in details:
            layer_details = interpreter.get_tensor(detail['index'])

            if "weights" in detail['name']:
                weight = Layer(layer_details)
                layers.append(weight.detail)

            if "bias" in detail['name']:
                bias = Layer(layer_details)
                layers.append(bias.detail)

        model.layers = layers
        db.append(model)

    return db


def load_target_models():
    targets = []

    for model_name in os.listdir("DL_models/v2para_check/"):
        # Load TFLite model and allocate tensors.
        try:
            interpreter = tf.lite.Interpreter("DL_models/v2para_check/" + model_name)
            interpreter.allocate_tensors()
        except:
            print(model_name, "loading error")
            continue

        # Get tensor details.
        details = interpreter.get_tensor_details()

        # init pre-trained model
        layers = []
        model = Model(model_name, layers)

        # MobileNets,
        for detail in details:
            layer_details = interpreter.get_tensor(detail['index'])

            if "weights" in detail['name']:
                weight = Layer(layer_details)
                layers.append(weight.detail)

            if "bias" in detail['name']:
                bias = Layer(layer_details)
                layers.append(bias.detail)

        model.layers = layers
        targets.append(model)

    return targets


def main():
    tf_hub_db = load_db()
    targets = load_target_models()
    # target_model = load_single_target_model("DL_models/TFLite/mobilenet.letgo.v1_1.0_224_quant.v7.tflite")
    # target_model_layers = target_model.layers

    # compute the similarity
    for target in targets:
        print(target.name)
        for model in tf_hub_db:
            results = []
            for x, y in zip(target.layers, model.layers):
                try:
                    results.append(np.allclose(x, y, equal_nan=True))
                except:
                    print("shape incomparable")


            max_continue = [sum(1 for _ in group) for key, group in itertools.groupby(results) if key]
            if max_continue:
                similarity = max_continue[0] / len(results)
                print(target.name, model.name, "{:.2%}".format(similarity))


1

if __name__ == "__main__":
    main()
