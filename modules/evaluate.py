import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

if __name__ == '__main__': 
    from dataset import LABELS
else: 
    from modules.dataset import LABELS

ROOT_PATH = "/home/jovyan/ChestXray-14"

class Evaluate:
    def __init__(self, model_path):
        self.y_true = None
        self.y_preds = None
        self.model_path = model_path
        self.model = self.get_model(model_path)
        self.best_thresholds = None
        self.thresholds_200 = None
    
    def get_model(self, path):
        return tf.keras.models.load_model(path)
    
    def get_y_true(self, data):
        y_true=[]
        for X,y in data:
            for label in y:
                y_true.append(label)
        y_true = tf.Variable(y_true)
        self.y_true = y_true
        return y_true

    def get_confusion_metrics(self, y_true, y_preds):
        m = tf.keras.metrics.AUC(multi_label=True)
        m.update_state(y_true, y_preds)

        thresholds = m.thresholds
        variables = m.variables
        TP = variables[0]
        TN = variables[1]
        FP = variables[2]
        FN = variables[3]
        return thresholds, TP, TN, FP, FN

    def model_predict(self, test_dataset):
        return self.model.predict(test_dataset)

    def get_f1_scores_200_thresholds(self, test_dataset):
        self.y_true = self.get_y_true(test_dataset)
        self.y_preds = self.model_predict(test_dataset)
        
        confusion_metrics = self.get_confusion_metrics(self.y_true, self.y_preds)
        thresholds, TP, TN, FP, FN = confusion_metrics
        self.thresholds_200 = thresholds
        f1_class_dict = dict()
        for i in range(len(thresholds)):
            tp, tn, fp, fn = TP[i], TN[i], FP[i], FN[i]
            for label_index in range(15):
                f1_score = 2*tp[label_index] / (2*tp[label_index] + fp[label_index] + fn[label_index])
                try:
                    f1_class_dict[LABELS[label_index]].append(f1_score)
                except KeyError:
                    f1_class_dict[LABELS[label_index]] = [f1_score]
        print(LABELS)
        return f1_class_dict
    
    def get_f1_scores(self, test_dataset):
        self.y_true = self.get_y_true(test_dataset)
        self.y_preds = self.model_predict(test_dataset)
        metric = tfa.metrics.MultiLabelConfusionMatrix(num_classes=15)
        metric.update_state(self.y_true,
                            np.greater_equal(self.y_preds, self.best_thresholds).astype('int8'))
        result = metric.result()
        
        f1_class_dict = dict()
        for idx, confusion in enumerate(result):
            label = LABELS[idx]
            TP, TN, FP, FN = (confusion[1, 1],
                              confusion[0, 0],
                              confusion[0, 1],
                              confusion[1, 0])
            f1_score = 2*TP / (2*TP + FP + FN)
            f1_class_dict[label] = [f1_score.numpy()]
        return f1_class_dict
    
    def get_precision_scores(self, test_dataset, new_calculate=True):
        if new_calculate is True:
            self.y_true = self.get_y_true(test_dataset)
            self.y_preds = self.model_predict(test_dataset)
        metric = tfa.metrics.MultiLabelConfusionMatrix(num_classes=15)
        metric.update_state(self.y_true,
                            np.greater_equal(self.y_preds, self.best_thresholds).astype('int8'))
        result = metric.result()
        
        precision_class_dict = dict()
        for idx, confusion in enumerate(result):
            label = LABELS[idx]
            TP, TN, FP, FN = (confusion[1, 1],
                              confusion[0, 0],
                              confusion[0, 1],
                              confusion[1, 0])
            precision = TP / (TP + FP)
            precision_class_dict[label] = [precision.numpy()]
        return precision_class_dict
    
    def get_recall_scores(self, test_dataset, new_calculate=True):
        if new_calculate is True:
            self.y_true = self.get_y_true(test_dataset)
            self.y_preds = self.model_predict(test_dataset)
        metric = tfa.metrics.MultiLabelConfusionMatrix(num_classes=15)
        metric.update_state(self.y_true,
                            np.greater_equal(self.y_preds, self.best_thresholds).astype('int8'))
        result = metric.result()
        
        recall_class_dict = dict()
        for idx, confusion in enumerate(result):
            label = LABELS[idx]
            TP, TN, FP, FN = (confusion[1, 1],
                              confusion[0, 0],
                              confusion[0, 1],
                              confusion[1, 0])
            recall = TP / (TP + FN)
            recall_class_dict[label] = [recall.numpy()]
        return recall_class_dict
    
    def get_best_threshold(self,
                           test_dataset=None,
                           save_best_thresholds=f"{ROOT_PATH}/results/paper/table3_1/best_thresholds.csv",
                           save_200_thresholds=f"{ROOT_PATH}/results/paper/table3_1/f1_per_thresholds.csv"):
        if test_dataset is None:
            fold_num = int(self.model_path.split(".")[0][-1])
            test_dataset = datasets[fold_num-1]
        
        f1_scores_dict = self.get_f1_scores_200_thresholds(test_dataset)
        best_thresholds_dict = {"thresholds": [], "f1_most": [], "label": []}
        for key, value in f1_scores_dict.items():
            f1_arg_max = np.argmax(value)
            best_thresholds_dict["f1_most"].append(value[f1_arg_max].numpy())
            best_thresholds_dict["label"].append(key)
            best_thresholds_dict["thresholds"].append(self.thresholds_200[f1_arg_max])
        
        df = pd.DataFrame(best_thresholds_dict)
        df = df.set_index("label")
        df.to_csv(save_best_thresholds, index=True)
        print(f"{save_best_thresholds} was success!")
        # print(df)
        
        df_200_thresholds = pd.DataFrame(f1_scores_dict)
        df_200_thresholds.to_csv(save_200_thresholds, index=True)
        print(f"{save_200_thresholds} was success!")
        self.best_thresholds = df.copy()["thresholds"].values

    def __enter__(self):
        print("Doing ...!")
        return self

    def __exit__(self, *arg):
        self.y_true = None
        self.y_preds = None
        print("Done!")