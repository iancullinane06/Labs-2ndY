import seaborn as sns
from matplotlib import pyplot as plt    
from matplotlib import font_manager

sns.set_theme(style="darkgrid", rc={"axes.facecolor": ".9"})
sns.set(font="Verdana")

def _style_axis(ax, title, xlabel, ylabel):
    ax.set(
        xlabel=xlabel,
        ylabel=ylabel
    )
    ax.set_title(title, fontdict={"fontsize": 14, "fontweight": "bold"})


def _finish_plot(fig, filename, save_directory):
    fig.tight_layout()
    fig.savefig(f"{save_directory}/{filename}")
    plt.show()
    
if __name__ == "__main__":
    pass