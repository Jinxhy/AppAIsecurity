# Robustness of on-device Models: AdversarialAttack to Deep Learning Models on Android Apps
Deep learning has shown its power in many applications, including object detection in images, natural-language understanding, speech recognition. To make it more accessible to
end-users, __many deep learning models are embedded in mobile apps__. 

Compared to offloading deep learning from smartphones to the cloud, performing machine learning on-device can help improve latency, connectivity, and power consumption. However, most deep learning models within Android apps __can be easily obtained via mature reverse engineering__, and the model exposure may invite __adversarial attacks__. 

In this study, we propose __a simple but effective approach to hack deep learning models with adversarial attacks by identifying their highly similar pre-trained models from TensorFlow Hub__. All 10 real-world Android apps in the experiment are successfully attacked by our approach. Apart from the feasibility of the model attack, we also carry out an empirical study to investigate the characteristic of deep learning models of hundreds of Android apps from Google Play. The results show that many of them are similar to each other and widely use fine-tuning techniques to pre-trained models on the Internet.

## Details
To demonstrate our task, we first show some common mobile and edge use cases achieved via on-device model inference.

![img_classification](figures/use_cases.png)
