import seaborn as sns
from matplotlib import pyplot as plt    

sns.set_theme(style="darkgrid", rc={"axes.facecolor": ".9"})

def _style_axis(ax, title, xlabel, ylabel):
    ax.set(
        title=title,
        xlabel=xlabel,
        ylabel=ylabel
    )


def _finish_plot(fig, filename, save_directory):
    fig.tight_layout()
    fig.savefig(f"{save_directory}/{filename}")
    plt.show()