"""
Written by Yujin Huang(Jinx)

"""

import tensorflow as tf
import foolbox as fb
import numpy as np
import matplotlib.pyplot as plt
import eagerpy as ep
from PIL import Image
import os


def save_advs(model_name, attack_name, advs_list):
    # Rescale to 0-255 and convert to uint8, then save adversarial examples
    for i, advs in enumerate(advs_list):
        for index, adv in enumerate(advs):
            adv_format = (255.0 / advs[index].numpy().max() * (advs[index].numpy() - advs[index].numpy().min())).astype(
                np.uint8)
            adv_img = Image.fromarray(adv_format)
            path = 'adv_examples/' + model_name + '/resnet/' + attack_name + '/' + str(i)
            if not os.path.exists(path):
                os.makedirs(path)
            adv_img.save(path + '/adv' + str(index) + '.png')


def generate_advs(tflite_model):

    # obtain a pre-trained TFLite model: MobileNetV1, MobileNetV2, ResNet50
    url = "https://github.com/jonasrauber/foolbox-tensorflow-keras-applications"
    fmodel = fb.zoo.get_model(url, name="MobileNetV1")

    # obtain the original input images of a particular TFLite model
    images, labels = fb.utils.samples(fmodel, index=0, dataset='imagenet', batchsize=10)


    # init attacks
    fgsm = fb.attacks.FGSM()
    df = fb.attacks.L2DeepFoolAttack()
    ddn = fb.attacks.DDNAttack(steps=300)
    inversion = fb.attacks.InversionAttack(distance=fb.distances.LpDistance(float('inf')))
    nf = fb.attacks.NewtonFoolAttack()
    linfpgd = fb.attacks.LinfPGD()
    l2pgd = fb.attacks.L2PGD()
    linfbi = fb.attacks.LinfBasicIterativeAttack()
    l2bi = fb.attacks.L2BasicIterativeAttack()
    sapn = fb.attacks.SaltAndPepperNoiseAttack()

    # set attack parameters
    ba_epsilons = np.linspace(2.5, 2.5, num=1)
    fgsm_epsilons = np.linspace(0.02, 0.02, num=1)
    df_epsilons = np.linspace(1.4, 1.4, num=1)
    ddn_epsilons = np.linspace(0.5, 0.5, num=1)
    inv_epsilons = np.linspace(10, 10, num=1)
    nf_epsilons = np.linspace(12, 12, num=1)
    linfpgd_epsilons = np.linspace(0.05, 0.05, num=1)
    l2pgd_epsilons = np.linspace(12, 12, num=1)
    linfbi_epsilons = np.linspace(0.05, 0.05, num=1)
    l2bi_epsilons = np.linspace(1, 1, num=1)
    sapn_epsilons = np.linspace(80, 80, num=1)

    # run attack
    raw, fgsm_advs_list, success = fgsm(fmodel, images, labels, epsilons=fgsm_epsilons)
    save_advs(tflite_model, 'FGSM', fgsm_advs_list)

    raw, df_advs_list, success = df(fmodel, images, labels, epsilons=df_epsilons)
    save_advs(tflite_model, 'DeepFool', df_advs_list)

    raw, ddn_advs_list, success = ddn(fmodel, images, labels, epsilons=ddn_epsilons)
    save_advs(tflite_model, 'DDN', ddn_advs_list)

    raw, inv_advs_list, success = inversion(fmodel, images, labels, epsilons=inv_epsilons)
    save_advs(tflite_model, 'InversionAttack', inv_advs_list)

    raw, nf_advs_list, success = nf(fmodel, images, labels, epsilons=nf_epsilons)
    save_advs(tflite_model, 'NewtonFoolAttack', nf_advs_list)

    raw, linfpgd_advs_list, success = linfpgd(fmodel, images, labels, epsilons=linfpgd_epsilons)
    save_advs(tflite_model, 'LinfPGD', linfpgd_advs_list)

    raw, l2pgd_advs_list, success = l2pgd(fmodel, images, labels, epsilons=l2pgd_epsilons)
    save_advs(tflite_model, 'L2PGD', l2pgd_advs_list)

    raw, linfbi_advs_list, success = linfbi(fmodel, images, labels, epsilons=linfbi_epsilons)
    save_advs(tflite_model, 'LinfBasicIterativeAttack', linfbi_advs_list)

    raw, l2bi_advs_list, success = l2bi(fmodel, images, labels, epsilons=l2bi_epsilons)
    save_advs(tflite_model, 'L2BasicIterativeAttack', l2bi_advs_list)

    raw, sapn_advs_list, success = sapn(fmodel, images, labels, epsilons=sapn_epsilons)
    save_advs(tflite_model, 'SaltAndPepperNoiseAttack', sapn_advs_list)


def main():
    # sotre the generated adversarial examples for the target TFLite model
    # pass the model name
    generate_advs('converted_model')


if __name__ == "__main__":
    main()
