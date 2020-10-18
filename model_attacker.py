import numpy as np
import tensorflow as tf
import cv2
import pathlib
import sys
import shutil
import os

# display all numpy array
np.set_printoptions(threshold=sys.maxsize)

# Load TFLite model and allocate tensors.
# TF_Hub/mobilenet_v2_1.0_224_quantized_1_metadata_1.tflite
# DL_models/transfer/mobile_ica_8bit_v2.tflite
# interpreter = tf.lite.Interpreter(model_path="TF_Hub/mobilenet_v2_1.0_224_1_metadata_1.tflite")
interpreter = tf.lite.Interpreter(model_path='DL_models/transfer/converted_model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def copy_raw(file_path, index, label):
    foolbox_dir = './venv/Lib/site-packages/foolbox/data/'
    shutil.copy2(file_path, foolbox_dir + 'imagenet_0' + str(index) + '_' + str(label) + '.png')


def inference(image_path, copy):
    output_labels = []
    index = 0

    for file in pathlib.Path(image_path).iterdir():
        # read and resize the image
        img = cv2.imread(r"{}".format(file.resolve()))
        new_img = cv2.resize(img, (224, 224))
        # if the input requires float32
        new_img = new_img.astype(np.float32) / 255
        # print(type(new_img))
        # print(new_img)

        # input_details[0]['index'] = the index which accepts the input
        interpreter.set_tensor(input_details[0]['index'], [new_img])

        # run the inference
        interpreter.invoke()

        # output_details[0]['index'] = the index which provides the input
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # output prediction
        max_pro_index = list(np.where(output_data[0] == np.amax(output_data[0])))

        output_labels.append(max_pro_index[0][0])

        # print(file, max_pro_index[0][0])

        # copy to foolbox directory as raw image
        if copy:
            copy_raw(file, index, (max_pro_index[0][0] - 1))
            index += 1

    return output_labels


def success_rate(raw, adv):
    ori_labels = raw
    adv_labels = adv

    sum = len(ori_labels)
    no_match = 0
    for l1, l2 in zip(ori_labels, adv_labels):
        if l1 != l2:
            no_match += 1
    print('attack success rate:', no_match / sum)


def main():
    # attacks = 'adv_examples/mobile_ica_8bit_v2/attacks'
    attacks = 'adv_examples/converted_model/resnet'
    ori_labels = inference('adv_examples/converted_model/raw', False)
    # print(ori_labels)

    for attack_name in os.listdir(attacks):
        attack_dir = os.path.join(attacks, attack_name)
        print('\n', attack_name)

        for i in range(1):
            adv_labels = inference(attack_dir + '/' + str(i), False)
            success_rate(ori_labels, adv_labels)


if __name__ == "__main__":
    main()
