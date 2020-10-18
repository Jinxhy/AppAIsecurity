"""
Written by Yujin Huang(Jinx)
Started 25/07/2020 8:12 pm
Last Editted 

Description of the purpose of the code
"""

import tensorflow as tf
import foolbox as fb
import numpy as np
import matplotlib.pyplot as plt
import eagerpy as ep
from PIL import Image
import os


def save_advs(model_name, attack_name, advs_list):
    # Rescale to 0-255 and convert to uint8, then save adversarial images
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
    # get a model: MobileNetV2,
    # model = tf.keras.applications.Xception(weights="imagenet")
    url = "https://github.com/jonasrauber/foolbox-tensorflow-keras-applications"
    fmodel = fb.zoo.get_model(url, name="ResNet50")

    # specify the correct bounds and preprocessing: MobileNetV2,
    # preprocessing = dict()
    # bounds = (-1, 1) # TensorFlow MobileNetV2,ResNet50V2
    # fmodel = fb.TensorFlowModel(model, bounds=bounds, preprocessing=preprocessing)
    #
    # # transform bounds
    # fmodel = fmodel.transform_bounds((0, 1))
    # assert fmodel.bounds == (0, 1)

    # get some test images
    images, labels = fb.utils.samples(fmodel, index=0, dataset='imagenet', batchsize=10)

    # check the accuracy of your model to make sure you specified the correct preprocessing
    print("Accuracy(before attack):", fb.utils.accuracy(fmodel, images, labels))
    print("Image:", type(images), images.shape)
    print("Label:", type(labels), labels)

    # generate attack:
    # ba = fb.attacks.BoundaryAttack()
    # l2cw = fb.attacks.L2CarliniWagnerAttack()
    fgsm = fb.attacks.FGSM()
    df = fb.attacks.L2DeepFoolAttack()
    ddn = fb.attacks.DDNAttack(steps=300)
    # va = fb.attacks.VirtualAdversarialAttack(steps=1)
    inversion = fb.attacks.InversionAttack(distance=fb.distances.LpDistance(float('inf')))
    nf = fb.attacks.NewtonFoolAttack()
    #
    linfpgd = fb.attacks.LinfPGD()
    l2pgd = fb.attacks.L2PGD()
    #
    linfbi = fb.attacks.LinfBasicIterativeAttack()
    l2bi = fb.attacks.L2BasicIterativeAttack()
    #
    # linfaun = fb.attacks.LinfAdditiveUniformNoiseAttack()
    # l2aun = fb.attacks.L2AdditiveUniformNoiseAttack()
    #
    sapn = fb.attacks.SaltAndPepperNoiseAttack()
    # cr = fb.attacks.L2ContrastReductionAttack()
    # agn = fb.attacks.L2AdditiveGaussianNoiseAttack()

    # use the misclassification criterion explicitly
    # images = ep.astensor(images)
    # labels = ep.astensor(labels)
    # criterion = fb.criteria.Misclassification(labels)
    ba_epsilons = np.linspace(2.5, 2.5, num=1)
    # l2cw_epsilons= np.linspace(2.5, 2.5, num=1)
    fgsm_epsilons = np.linspace(0.02, 0.02, num=1)
    df_epsilons = np.linspace(1.4, 1.4, num=1)
    ddn_epsilons = np.linspace(0.5, 0.5, num=1)
    # va_epsilons = np.linspace(0, 50, num=5)
    inv_epsilons = np.linspace(10, 10, num=1)
    nf_epsilons = np.linspace(12, 12, num=1)
    #
    linfpgd_epsilons = np.linspace(0.05, 0.05, num=1)
    l2pgd_epsilons = np.linspace(12, 12, num=1)
    #
    linfbi_epsilons = np.linspace(0.05, 0.05, num=1)
    l2bi_epsilons = np.linspace(1, 1, num=1)
    #
    # linfaun_epsilons = np.linspace(0.15, 0.15, num=1)
    # l2aun_epsilons = np.linspace(0, 50, num=5)
    #
    sapn_epsilons = np.linspace(80, 80, num=1)
    # cr_epsilons = np.linspace(0, 128, num=5)
    # agn_epsilons = np.linspace(40, 40, num=1)

    # run attack
    # raw, ba_advs_list, success = ba(fmodel, images, labels, epsilons=ba_epsilons)
    # save_advs(tflite_model, 'BoundaryAttack', ba_advs_list)

    # raw, l2cw_advs_list, success = l2cw(fmodel, images, labels, epsilons=l2cw_epsilons)
    # save_advs(tflite_model, 'L2CarliniWagnerAttack', l2cw_advs_list)

    raw, fgsm_advs_list, success = fgsm(fmodel, images, labels, epsilons=fgsm_epsilons)
    save_advs(tflite_model, 'FGSM', fgsm_advs_list)

    raw, df_advs_list, success = df(fmodel, images, labels, epsilons=df_epsilons)
    save_advs(tflite_model, 'DeepFool', df_advs_list)

    raw, ddn_advs_list, success = ddn(fmodel, images, labels, epsilons=ddn_epsilons)
    save_advs(tflite_model, 'DDN', ddn_advs_list)

    # raw, va_advs_list, success = va(fmodel, images, labels, epsilons=va_epsilons)
    # save_advs(tflite_model, 'VirtualAdversarialAttack', va_advs_list)
    #
    raw, inv_advs_list, success = inversion(fmodel, images, labels, epsilons=inv_epsilons)
    save_advs(tflite_model, 'InversionAttack', inv_advs_list)
    #
    raw, nf_advs_list, success = nf(fmodel, images, labels, epsilons=nf_epsilons)
    save_advs(tflite_model, 'NewtonFoolAttack', nf_advs_list)
    #
    raw, linfpgd_advs_list, success = linfpgd(fmodel, images, labels, epsilons=linfpgd_epsilons)
    save_advs(tflite_model, 'LinfPGD', linfpgd_advs_list)
    raw, l2pgd_advs_list, success = l2pgd(fmodel, images, labels, epsilons=l2pgd_epsilons)
    save_advs(tflite_model, 'L2PGD', l2pgd_advs_list)
    #
    raw, linfbi_advs_list, success = linfbi(fmodel, images, labels, epsilons=linfbi_epsilons)
    save_advs(tflite_model, 'LinfBasicIterativeAttack', linfbi_advs_list)
    raw, l2bi_advs_list, success = l2bi(fmodel, images, labels, epsilons=l2bi_epsilons)
    save_advs(tflite_model, 'L2BasicIterativeAttack', l2bi_advs_list)
    #
    # raw, linfaun_advs_list, success = linfaun(fmodel, images, labels, epsilons=linfaun_epsilons)
    # save_advs(tflite_model, 'LinfAdditiveUniformNoiseAttack', linfaun_advs_list)
    # raw, l2aun_advs_list, success = l2aun(fmodel, images, labels, epsilons=l2aun_epsilons)
    # save_advs(tflite_model, 'L2AdditiveUniformNoiseAttack', l2aun_advs_list)
    #
    raw, sapn_advs_list, success = sapn(fmodel, images, labels, epsilons=sapn_epsilons)
    save_advs(tflite_model, 'SaltAndPepperNoiseAttack', sapn_advs_list)
    #
    # raw, cr_advs_list, success = cr(fmodel, images, labels, epsilons=cr_epsilons)
    # save_advs(tflite_model, 'L2ContrastReductionAttack', cr_advs_list)
    #
    # raw, agn_advs_list, success = agn(fmodel, images, labels, epsilons=agn_epsilons)
    # save_advs(tflite_model, 'L2AdditiveGaussianNoiseAttack', agn_advs_list)


def main():
    # pass the target tflite model name
    generate_advs('converted_model')


if __name__ == "__main__":
    main()

# print("Success rate:", success.float32().mean().item())
# robust_accuracy = 1 - success.float32().mean(axis=-1)
# plt.plot(epsilons, robust_accuracy.numpy())
# plt.show()

# visualize adversarial examples and perturbations
# plt.imshow(images[0])
# plt.show()


# plt.imshow(advs[0])
# plt.show()
# adv_format = (255.0 / advs[0].numpy().max() * (advs[0].numpy() - advs[0].numpy().min())).astype(np.uint8)
# adv_img = Image.fromarray(adv_format)
# adv_img.save('adv_examples/MobileNetV2/adversarial.png')
# fb.plot.images(images)
# plt.show()
# fb.plot.images(advs[14])
# plt.show()
