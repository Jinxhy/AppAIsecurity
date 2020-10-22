import os
import shutil
from tqdm import tqdm

original = './decomposed_apks'
target_tflite = './DL_apps/TFLite'

# .tflite
end_with_tflite = ['.tflite', '.lite']

dl_app_count = 0

for cat_name in os.listdir(original):
    cat_dir = os.path.join(original, cat_name)

    if os.path.isdir(cat_dir):
        for apk_name in tqdm(os.listdir(cat_dir)):
            apk_dir = os.path.join(cat_dir, apk_name)
            is_dl_app = False

            # search .tflite or .lite model file
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

                if stop_looping:
                    break

            if is_dl_app:
                dl_app_count += 1

print('Extract ' + str(dl_app_count) + ' TFLite DL-based apps.')
