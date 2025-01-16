import numpy as np
import torch.nn as nn
import torch
from sklearn.metrics import roc_curve
from sklearn.metrics import auc, f1_score
from sklearn.metrics import matthews_corrcoef

class Metrics():
    def __init__(self, logger, type="train") -> None:
        '''
        type: "train" | "test" | "val"
        '''
        self.prob_list = []
        self.label_list = []
        self.num_correct = 0
        self.loss = 0
        self.type = type
        self.logger = logger
        self.num_classes = 2
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def reset(self):
        self.prob_list = []
        self.label_list = []
        self.num_correct = 0
        self.loss = 0
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))


    def update(self, label, logits, loss):
        '''
        loss: Tensor(size=[1])
        '''

        self.loss += loss.item()
        prob = nn.Sigmoid()(logits)
        self.num_correct += torch.eq(prob>0.5, label).sum().item()
        self.prob_list.extend(prob.tolist())
        self.label_list.extend(label.tolist())


    def get_results(self):
        '''
        ACC AUC LOSS F1 KAPPA MCC
        '''
        loss = self.loss / len(self.label_list)
        acc = self.num_correct / len(self.label_list)
        preds = [prob > 0.5 for prob in self.prob_list]
        hist = self._fast_hist(preds, self.label_list)
        specificity = (hist[0][0]) / (self.label_list.count(0))
        self.f1_score = f1_score(self.label_list, preds, average='weighted')
        p_e = (preds.count(0) * self.label_list.count(0) + preds.count(1) * self.label_list.count(1)) / (len(self.label_list) ** 2)
        kappa = (acc - p_e) / (1 - p_e)
        fpr_keras_1, tpr_keras_1, thresholds_keras_1 = roc_curve(self.label_list, self.prob_list)
        auc_keras_1 = auc(fpr_keras_1, tpr_keras_1)
        
        # Calculate MCC
        mcc = matthews_corrcoef(self.label_list, preds)
        
        return {
            "loss": loss,
            "acc": acc,
            "auc": auc_keras_1,
            "f1": self.f1_score,
            "kappa": kappa,
            "specificity": specificity,
            "mcc": mcc,
            "avg": (self.f1_score + kappa + specificity + mcc) / 4
        }
    
    def print_res(self, res, epoch):
        '''type: train/val/test'''

        print("%s\t"%self.type, "Loss: {:.6f}\t Acc: {:.6f}\tKAPPA: {:.6f}\tF1: {:.6f}\tmcc: {:.6f}\tspecificity: {:.6f}\tAVG: {:.6f}".format(res["loss"], res["acc"], res["kappa"], res["f1"],res["mcc"], res["specificity"], res["avg"]))
        self.logger.logger.info("{}\tLoss: {:.6f}\t Acc: {:.6f}\tKAPPA: {:.6f}\tF1: {:.6f}\tmcc: {:.6f}\tspecificity: {:.6f}\tAVG: {:.6f}".format(self.type, res["loss"], res["acc"], res["kappa"], res["f1"], res["mcc"], res["specificity"], res["avg"]))
    
    def _fast_hist(self, preds, label):
        preds = np.array(preds)
        label = np.array(label)
        return np.bincount(self.num_classes * label.astype(int) + preds, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
    