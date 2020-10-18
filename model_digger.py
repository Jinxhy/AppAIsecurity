import os
import shutil
from tqdm import tqdm

original = './temp/TFLite'

target = './DL_models/temp'

end_with = '.tflite'

# total 307(164) .tflite models
model_count = 267

for cat_name in os.listdir(original):
    cat_dir = os.path.join(original, cat_name)

    if os.path.isdir(cat_dir):
        for apk_name in os.listdir(cat_dir):
            apk_dir = os.path.join(cat_dir, apk_name)

            # extract model
            for root, dirs, files in os.walk(apk_dir, topdown=True):
                for file in files:
                    if file.endswith(end_with):
                        # extract DL-based app
                        model_path = os.path.join(root, file)
                        copy_file = str(model_count) + '_' + file
                        copy_path = os.path.join(target, copy_file)
                        print('copying ' + file + ' now...')
                        shutil.copy2(model_path, copy_path)
                        model_count += 1

print('\nExtract ' + str(model_count) + end_with + ' DL models.')
