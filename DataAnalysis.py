import matplotlib.pyplot as plt
import pandas as pd

from DataLoading import DataLoading

FEATURES_SRC = "data/train_features.csv"
LABELS_SRC = "data/train_labels.csv"
VISUALS = "data/visuals/"


class DataAnalysis:
    """ Visualisation and analysis of the training data """

    def __init__(self):
        dl = DataLoading()
        dl.load_data(FEATURES_SRC, LABELS_SRC, False)
        self.features, self.labels = dl.get_train_dataset()

    def exploratory_analysis(self):
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(" * Features dimension:\n{}".format(self.features.shape))
            print("\n * Example features:\n{}".format(self.features.head(1)))
            print("\n * Example labels:\n{}".format(self.labels.head(1)))
            print("\n * Features summary:")
            self.features.info()
            print("\n * Numeric features info:\n{}".format(self.features.describe()))

        # features visuals
        plot_elems = self.features.iloc[:, 1:]
        ax_count = int()
        axs = list()
        fig_idx = int()

        for feature in plot_elems.columns.to_list():
            if not ax_count:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
                axs = [ax1, ax2, ax3, ax4]

            if len(pd.unique(plot_elems[feature])) <= 10:
                # if not many different values, consider as str for better display
                elements = sorted([(str(k), v) for k, v in plot_elems[feature].value_counts().to_dict().items()],
                                  key=lambda x: x[0])
            else:
                elements = sorted([(k, v) for k, v in plot_elems[feature].value_counts().to_dict().items()],
                                  key=lambda x: x[0])

            axs[ax_count].bar([e[0] for e in elements], [e[1] for e in elements], width=0.3)
            axs[ax_count].set_title(feature)
            ax_count += 1

            if not ax_count % 4:
                # plt.savefig(VISUALS + "features_visual_{}.pdf".format(fig_idx))
                plt.show()
                ax_count *= 0
                fig_idx += 1

        if ax_count % 4:
            # plt.savefig(VISUALS + "features_visual_{}.pdf".format(fig_idx))
            plt.show()

        # labels visuals
        fig, ax = plt.subplots(1, 1)
        elements = sorted([(str(k), v) for k, v in self.labels[self.labels.columns.to_list()[0]].value_counts().to_dict().items()],
                          key=lambda x: x[0])
        ax.bar([e[0] for e in elements], [e[1] for e in elements], width=0.3, color=["tab:blue"])
        ax.set_title("Damage grade")
        # plt.savefig(VISUALS + "labels_visual.pdf")
        plt.show()
