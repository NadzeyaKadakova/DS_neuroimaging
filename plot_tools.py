import numpy as np
import matplotlib.pyplot as plt


#create boxplot + scatter figure
#arguments: ax:                figure axis
#           list_group_data:   list of containers holding each group's data
#           list_group_names:  list of group names
#           pallet: color pallet for scatter plot, e.g. ['r', 'g', 'b', 'y']
def boxplot_scatter(ax, list_group_data, list_group_names, pallet=None):

    #create the boxplot - pretty straightforward
    ax.boxplot(list_group_data, labels=list_group_names)

    #size of jitter (noise) to add to x-axis for scatter plot
    JITTER_SIZE = 0.04

    if pallet is not None:
        assert(len(list_group_data) == len(pallet))

    scatter_x_vals = []
    for i, data_i in enumerate(list_group_data):
        y_i = list_group_data[i]
        x_i = np.random.normal(i + 1, JITTER_SIZE, len(y_i))

        if pallet is not None:
            ax.scatter(x_i, y_i, alpha=0.4, s=3, color=pallet[i])
        else:
            ax.scatter(x_i, y_i, alpha=0.4, s=3)