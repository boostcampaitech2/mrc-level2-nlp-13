import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

VIZ_SAVE_PATH = "./viz_results"

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
            Plot or save confusion matrix
            Parameters:
                real : np.array, ground trues of label if None 
                pred : np.array, model predicted results
                save : bool, defalut = True, save or plot options : {save, linux, window, notebook}
                file_name : str, save image name containg extentions

                other parameter is available for the same function, "seaborn heatmap"
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
            **key_args,
        )

        # set axis annotation
        plt.title("Confusion Matrix")
        plt.xlabel("Real")
        plt.ylabel("Pred")

        # save or plot result
        if save:
            save_path = os.path.join(VIZ_SAVE_PATH, file_name)
            fig.savefig(save_path)
        else:
            plt.show()

