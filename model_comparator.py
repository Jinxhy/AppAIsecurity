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


class Model:
    def __init__(self, name, layers):
        self.name = name
        self.layers = layers


class Layer:
    def __init__(self, name, index, shape, dtype):
        self.detail = name + "," + index + "," + shape + "," + dtype


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
            shape = re.sub(' +', ',', str(detail['shape']))[:1] + re.sub(' +', ',', str(detail['shape']))[2:]
            layer = Layer(detail['name'], str(detail['index']), shape,
                          re.sub(' +', ',', str(detail['dtype'])))

            layers.append(layer.detail)

        model.layers = layers
        db.append(model)

    return db


def load_single_target_model(model_path):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()

    # Get tensor details.
    details = interpreter.get_tensor_details()

    layers = []
    target_model = Model(model_path.rsplit("/", 1)[-1], layers)

    for detail in details:
        shape = re.sub(' +', ',', str(detail['shape']))[:1] + re.sub(' +', ',', str(detail['shape']))[2:]
        layer = Layer(detail['name'], str(detail['index']), shape,
                      re.sub(' +', ',', str(detail['dtype'])))

        layers.append(layer.detail)

        target_model.layers = layers

    return target_model


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
            shape = re.sub(' +', ',', str(detail['shape']))[:1] + re.sub(' +', ',', str(detail['shape']))[2:]
            layer = Layer(detail['name'], str(detail['index']), shape,
                          re.sub(' +', ',', str(detail['dtype'])))

            layers.append(layer.detail)

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
        results = {}

        for model in tqdm(tf_hub_db):
            results[model.name] = fuzz.ratio(",".join(target.layers), ",".join(model.layers))

        result = {k: v for k, v in sorted(results.items(), key=lambda x: x[1], reverse=True)}

        if result[list(result.keys())[0]] > 0:
            if result[list(result.keys())[0]] > 60:
                print(target.name)
                print(list(result.keys())[0], ":", result[list(result.keys())[0]], "%")
                print(list(result.keys())[1], ":", result[list(result.keys())[1]], "%")
                print(list(result.keys())[2], ":", result[list(result.keys())[2]], "%")
                print(list(result.keys())[3], ":", result[list(result.keys())[3]], "%")
                print()


if __name__ == "__main__":
    main()
