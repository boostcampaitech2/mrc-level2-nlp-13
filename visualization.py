##################
# import library #
##################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, recall_score, auc

#################
# set variables #
#################

VIZ_SAVE_PATH = "./viz_results"

#####################
# class & functions #
#####################

class VizSet:
    def __init__(self, real, pred):
        self.real = real
        self.pred = pred
        self.check_output_path()

    def check_output_path(self, output_path = VIZ_SAVE_PATH):
        # make output dir
        if not os.path.exists(output_path):
            os.mkdir(output_path)

    def plot_confusion_matrix(
            self,
            real : np.array = [], 
            pred : np.array = [], 
            save : bool = True, 
            file_name : str = "cm.jpg",
            **key_args
        )->None:
        """
            Plot or save confusion matrix image

            Parameters:
                real : np.array, ground trues of label if None 
                pred : np.array, model predicted results
                save : bool, defalut = True, save or plot options : {save, linux, window, notebook}
                file_name : str, save image name containg extentions and path, default = "./viz_results/cm.jpg"
                title : str, save image's title ex)"Epoch 1 result CM"
        """
        # check input data
        if len(real) < 1:
            real = self.real
        if len(pred) < 1:
            pred = self.pred

        # init prams
        num_classes = len(np.unique(real))
        modified_figure = 1

        if num_classes < 20:
            modified_figure = 1
        elif 20 < num_classes < 30:
            modified_figure = 1.5
        elif 30 < num_classes < 50:
            modified_figure = 2
        else:
            modified_figure = 3

        width, height  = 8, 8

        # calculate confusion matrix
        cm = confusion_matrix(real, pred)
        
        # set figure
        fig = plt.figure(
            figsize = ( width * modified_figure, height * modified_figure)
        )
        
        # plot confusion matrix
        sns.heatmap(
            data=cm,
            annot=True,
            cmap='Blues',
            fmt='d',
            cbar=False,
        )

        # set axis annotation
        args_dict = {
            "title" : "Confusion Matrix",
            "xlabel" : "Real",
            "ylabel" : "Pred"
        }
        for key in args_dict.keys():
            if key in key_args:
                args_dict[key] = key_args[key]
            
        plt.title(args_dict['title'])
        plt.xlabel(args_dict['xlabel'])
        plt.ylabel(args_dict['ylabel'])

        # save or plot result
        if save:
            save_path = os.path.join(VIZ_SAVE_PATH, file_name)
            fig.savefig(save_path)
        else:
            plt.show()


    def plot_metrics(
        self,
        real : np.array = [],
        pred : np.array = [],
        save = True,
        title = "Metrics",
        file_name = "metrics.jpg",
        **metrics,
    )->None:
        """
            Plot or save selected metrics' bar chart image

            Parameters:
                Base values : Array, Essential values for calculate metrics
                ------------------------------------------------------
                    real : np.array, ground trues value list
                    pred : np.array, model predicted value list
                    
                key values : bool, Optional plotting to chart
                ------------------------------------------------------
                    Available options:
                        accuracy, f1, precision, recall, EM, AUC, BLEU
                
                accuracy  : bool, calculate accuracy
                f1        : bool, calculate f1-micro score
                precision : bool, calculate precision
                recall    : bool, calculate recall
                EM        : bool, calculate Exact Match
                AUC       : bool, calculate auc score

            Example:
            >>> vizset = VizSet(real, pred)
            >>> vizset.plot_metrics(accuracy=True, EM=True, f1=True)
        """
        # check input data
        if len(real) < 1:
            real = self.real
        if len(pred) < 1:
            pred = self.pred

        # set calculations
        cal_metrics = {
            "accuracy" : accuracy_score,
            "f1" :       f1_score,
            "recall" :   recall_score,
            "AUC" :      auc,
            "EM" :       EM_score,
        }

        # calculate metrics
        result = {}
        for key in metrics.keys():
            if key == "f1":
                args = {"average":"micro"}
                result[key] = cal_metrics[key](real, pred, **args)
                continue
            
            result[key] = cal_metrics[key](real, pred)
            

        # plot metrics
        width, height = 4, 6
        
        fig = plt.figure(
            figsize=(width, height)
        )
        plt.title(title)
        metric_df = pd.DataFrame({key:[val] for key,val in result.items()})
        sns.barplot(data=metric_df)

        # save or plot result
        if save:
            save_path = os.path.join(VIZ_SAVE_PATH, file_name)
            fig.savefig(save_path)
        else:
            plt.show()
        
        return result

def EM_score(
    real:np.array, 
    pred:np.array
)->float:
    counter = 0
    for r, p in zip(real, pred):
        if r == p:
            counter+=1
    return counter / len(real)


def print_confusion_matrix(
    real:np.array, 
    pred:np.array, 
    label:dict=None
)->None:
    """
        Print confusion matrix for console

        Parameters:
            real : np.array, real value list
            pred : np.array, predicted value list by model
            label : dict, default = None, print which idx is label

        Example:
        >>> print_confusion_matrix(np.array([1,2,3,4]),np.array([1,2,3,4]))
        0|  1  0  0  0
        1|  0  1  0  0
        2|  0  0  1  0
        3|  0  0  0  1
          ------------
            0  1  2  3
        
    """
    # print label's state
    if label:
        for key, val in label.items():
            print(f"{val} : {key}")

    # calculate confusionmatrix
    cm = confusion_matrix(real, pred)

    # print title
    print("confusion matrix")

    #print grid
    print("   "+"-"*5*cm.shape[0])
    for idx in range(cm.shape[0]):
        print("%2d|"%idx,end="")
        for jdx in range(cm.shape[1]):
            print("%5d" % cm[idx, jdx], end="")
        print()

    print("   ",end="")
    print("-"*5*cm.shape[0], end="")
    print()
    print("   ",end="")
    for idx in range(cm.shape[0]):
        print("%5d"%idx, end="")
    print()





