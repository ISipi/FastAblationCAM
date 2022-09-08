import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib import cm
from typing import List
import logging as log
import os


class Figures:
    plt.axis('off')

    def __init__(self, num_classes: int, num_test_images: int, lst_of_heatmaps: List):
        self.num_classes = num_classes
        self.num_test_images = num_test_images
        self.lst_of_figures = lst_of_heatmaps

        # initialise figure
        self._fig = None
        self._grid = None
        self._img_counter = None

    def setup_figure(self, fig_title: str, fig_width: float = 7, fig_height: float = 8.75, dpi: int = 600):
        self._fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi, frameon=False)
        columns = self.num_test_images
        rows = self.num_classes
        self._grid = self._fig.add_gridspec(ncols=columns, nrows=rows, wspace=0.05, hspace=0.1)


        self._fig.suptitle(fig_title, fontsize=14, x=0.55, y=0.98)

        # get the min and max values from all heatmaps so that they can be used as min and max values in legend
        min_vals = [np.min(i['heatmap']) for i in self.lst_of_figures]
        min_val = np.min(min_vals)
        max_vals = [np.max(i['heatmap']) for i in self.lst_of_figures]
        max_val = np.max(max_vals)
        log.debug((min_val, max_val))

        m = cm.ScalarMappable(cmap='bwr')
        m.set_array(np.array([min_val, max_val]))
        cbaxes = self._fig.add_axes([0.92, 0.25, 0.01, 0.5])
        cbar = self._fig.colorbar(m, ticks=[min_val, max_val], cax=cbaxes, drawedges=False)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label("Neuron importance (in pixel values)", fontsize=10, rotation=270)
        #cbar.set_alpha(1)
        self.img_counter = 1

    def run_image_loop(self):
        for x, row in enumerate(self.lst_of_figures):
            row_text = self.lst_of_figures[x]['final_label']
            inner_grid = self._grid[x].subgridspec(ncols=1, nrows=1, wspace=0.2, hspace=0.0)
            row_labels = self._fig.add_subplot(self._grid[x], frameon=False)
            row_labels.set_yticks([])
            row_labels.set_xticks([])
            row_labels.spines['right'].set_visible(True)
            row_labels.spines['right'].set_color('black')
            row_labels.spines['top'].set_visible(False)
            row_labels.spines['left'].set_visible(False)
            row_labels.spines['bottom'].set_visible(False)
            self._fig.add_subplot(row_labels, tight_layout=True)
            if row_labels.is_first_col():
                row_text = row_text.replace("_", " ")
                row_labels.set_ylabel(row_text, fontsize=7, rotation=0, loc='center', labelpad=50)
            if row_labels.is_first_row():
                row_labels.xaxis.set_label_position('top')
                row_labels.set_xlabel(f"Test image {self.img_counter}", fontsize=7, rotation=0, labelpad=2)
                self.img_counter += 1
            for enum, i in enumerate(inner_grid):
                last_grid = i.subgridspec(ncols=3, nrows=1, wspace=0.0, hspace=0.0)
                original = self._fig.add_subplot(last_grid[0], frameon=True)
                heat = self._fig.add_subplot(last_grid[1], frameon=True)
                overlay_img = self._fig.add_subplot(last_grid[2], frameon=True)
                for axis in ['top', 'bottom', 'left', 'right']:
                    original.spines[axis].set_linewidth(0.1)
                    heat.spines[axis].set_linewidth(0.1)
                    overlay_img.spines[axis].set_linewidth(0.1)

                if row_labels.is_last_row():
                    original.set_xlabel('Original', fontsize=7, rotation=65, labelpad=2)
                    heat.set_xlabel('Heatmap', fontsize=7, rotation=65, labelpad=2)
                    overlay_img.set_xlabel('Overlay', fontsize=7, rotation=65, labelpad=2)

                original.set_yticks([])
                original.set_xticks([])
                og_img = cv2.cvtColor(self.lst_of_figures[x]['original_img'], cv2.COLOR_BGR2RGB)
                self._fig.add_subplot(original, tight_layout=True)
                plt.imshow(og_img)


                heat.set_yticks([])
                heat.set_xticks([])
                heat_img = cv2.cvtColor(self.lst_of_figures[x]['heatmap'], cv2.COLOR_BGR2RGB)
                self._fig.add_subplot(heat, tight_layout=True)
                plt.imshow(heat_img)


                overlay_img.set_yticks([])
                overlay_img.set_xticks([])
                comb_img = cv2.cvtColor(self.lst_of_figures[x]['combined_img'], cv2.COLOR_BGR2RGB)
                self._fig.add_subplot(overlay_img, tight_layout=True)
                plt.imshow(comb_img)

    def save_fig(self, save_directory, save_name):
        #fig.tight_layout(rect=[0, 0.1, 0.85, 0.95])
        plt.gcf().subplots_adjust(left=0.35, right=0.9, top=0.88, bottom=0.1)
        plt.savefig(f"{os.path.join(save_directory, save_name)}.png", dpi=600)
        #plt.show()
