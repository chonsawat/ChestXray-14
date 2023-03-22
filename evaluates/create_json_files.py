import re
import json
import tensorflow as tf
import os


def create_key_value_from_path(path):
    key = re.findall("fold_\d", path)[0]
    return key, path

if __name__ == "__main__":
    SAVED_MODEL_PATHS = "/home/jovyan/ChestXray-14/results/models"
    
    MODELS_PATHS = {
        "DenseNet121": {
            "None": { create_key_value_from_path(path)[0]:create_key_value_from_path(path)[1] for path in sorted(tf.io.gfile.glob(f"{SAVED_MODEL_PATHS}/DenseNet121_None_*.h5"))},
            "Imagenet": { create_key_value_from_path(path)[0]:create_key_value_from_path(path)[1] for path in sorted(tf.io.gfile.glob(f"{SAVED_MODEL_PATHS}/DenseNet121_imagenet_*.h5"))}
        },
        "EfficientNetB0": {
            "None": { create_key_value_from_path(path)[0]:create_key_value_from_path(path)[1] for path in sorted(tf.io.gfile.glob(f"{SAVED_MODEL_PATHS}/EfficientNetB0_None_*.h5"))},
            "Imagenet": { create_key_value_from_path(path)[0]:create_key_value_from_path(path)[1] for path in sorted(tf.io.gfile.glob(f"{SAVED_MODEL_PATHS}/EfficientNetB0_imagenet_*.h5"))}
        },
        "Resnet50": {
            "None": { create_key_value_from_path(path)[0]:create_key_value_from_path(path)[1] for path in sorted(tf.io.gfile.glob(f"{SAVED_MODEL_PATHS}/Resnet50_None_*.h5"))},
            "Imagenet": { create_key_value_from_path(path)[0]:create_key_value_from_path(path)[1] for path in sorted(tf.io.gfile.glob(f"{SAVED_MODEL_PATHS}/Resnet50_imagenet_*.h5"))}
        },
    }
    
    with open("model_path_config.json", "w") as out_file:
        json.dump(MODELS_PATHS, out_file, indent=True)
        
    CURRENT_PATH = os.path.abspath("")
    print(f"{CURRENT_PATH}/model_path_config.json : CREATED!")