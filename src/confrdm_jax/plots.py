
import matplotlib.pyplot as plt

def plot_data_hist(data):
    rt = data[..., 0]
    choice = data[..., 1]

    fig, ax = plt.subplots()

    ax.hist(rt[choice == 1], color="darkgreen", alpha=0.5)
    ax.hist(rt[choice == 0], color="indianred", alpha=0.5)

    return fig, ax