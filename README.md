# Robustness of on-device Models: AdversarialAttack to Deep Learning Models on Android Apps
Deep learning has shown its power in many applications, including object detection in images, natural-language understanding, speech recognition. To make it more accessible to
end-users, __many deep learning models are embedded in mobile apps__. 

Compared to offloading deep learning from smartphones to the cloud, performing machine learning on-device can help improve latency, connectivity, and power consumption. However, most deep learning models within Android apps __can be easily obtained via mature reverse engineering__, and the model exposure may invite __adversarial attacks__. 

In this study, we propose __a simple but effective approach to hack deep learning models with adversarial attacks by identifying their highly similar pre-trained models from TensorFlow Hub__. All 10 real-world Android apps in the experiment are successfully attacked by our approach. Apart from the feasibility of the model attack, we also carry out an empirical study to investigate the characteristic of deep learning models of hundreds of Android apps from Google Play. The results show that many of them are similar to each other and widely use fine-tuning techniques to pre-trained models on the Internet.

## Details
To demonstrate our task, we first show some common mobile and edge use cases achieved via on-device model inference, as shown in Fig 1.

![user_cases](figures/use_cases.png)*Fig. 1. Optimized on-device deep learning models for common mobile and edge use cases from https://www.tensorflow.org/lite/models*

Unlike the central guardians of the cloud server, on-device models may be more vulnerable inside users’ phones. For instance, most model files can be obtained by decompiling Android apps without any obfuscation or encryption. Such model files may be exposed to malicious attacks like adversarial attack. Considering the fact that many mobile apps with deep learning models are used for important tasks such as finance, social or even life-critical tasks like medical, driving-assistant, attacking the models inside those apps will be a disaster for users.

## EMPIRICAL STUDY OF THE SECURITY OF ON-DEVICE MODELS
In this work, we design a simple but effective way to adapt existing adversarial attacks to hack the deep learning models in real-world mobile apps. Apart from a pipeline of attacking the deep learning models, we also carry out an empirical study in the usage of deep learning models within thousands of real-world Android apps. We present those results by answering three research questions:
- How similar are TFLite models used in mobile apps?
- How widely pre-trained TFLite models are adopted?
- How robust are fine-tuned TFLite models against adversarial attacks?

### Dataset
For the preparation of our study, we crawled 62,822 mobile apps across various categories (e.g., Photograph, Social, Shopping) related to the image domain from Google Play.

### RQ1: HOW SIMILAR ARE TFLITE MODELS USED IN MOBILE APPS?

![user_cases](figures/model_relation.png)*Fig. 2. Relations between TFLite models*
