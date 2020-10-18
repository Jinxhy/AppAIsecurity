import os
import shutil
from tqdm import tqdm

original = './decomposed_apks'

# './DL_apps/TFLite'
# './DL_apps/TF'
target_tflite = './temp/TFLite'
target_tf = './DL_apps/TF'
target_pytorch = './DL_apps/PyTorch'

# .tflite
# .pb
# .pt
end_with_tflite = ['.tflite', '.lite']
end_with_pb = ['.pb', '.pbtxt']
end_with_pt = ['.pt', '.pkl']

# total 28+9 TFLite(.tflite) apps
# total 36+5 TF(.pb) apps
dl_app_count = 0

for cat_name in os.listdir(original):
    cat_dir = os.path.join(original, cat_name)

    if os.path.isdir(cat_dir):
        for apk_name in tqdm(os.listdir(cat_dir)):
            apk_dir = os.path.join(cat_dir, apk_name)
            is_dl_app = False

            # search .tflite/.pb/.pt model
            stop_looping = False
            for root, dirs, files in os.walk(apk_dir, topdown=True):
                for file in files:
                    if file.endswith(end_with_tflite[0]) or file.endswith(end_with_tflite[1]):
                        # extract DL-based app
                        copy_path = os.path.join(target_tflite, cat_name, apk_name)
                        print('copying ' + apk_name + ' now...(tflite)')
                        shutil.copytree(apk_dir, copy_path)

                        is_dl_app = True
                        stop_looping = True
                        break

                    # if file.endswith(end_with_pb[0]) or file.endswith(end_with_pb[1]):
                    #     # extract DL-based app
                    #     copy_path = os.path.join(target_tf, cat_name, apk_name)
                    #     print('copying ' + apk_name + ' now...(tf)')
                    #     shutil.copytree(apk_dir, copy_path)
                    #
                    #     is_dl_app = True
                    #     stop_looping = True
                    #     break
                    #
                    # if file.endswith(end_with_pt[0]) or file.endswith(end_with_pt[1]):
                    #     # extract DL-based app
                    #     copy_path = os.path.join(target_pytorch, cat_name, apk_name)
                    #     print('copying ' + apk_name + ' now...(pytorch)')
                    #     shutil.copytree(apk_dir, copy_path)
                    #
                    #     is_dl_app = True
                    #     stop_looping = True
                    #     break

                if stop_looping:
                    break

            if is_dl_app:
                dl_app_count += 1

print('Extract ' + str(dl_app_count) + ' DL-based apps.')
