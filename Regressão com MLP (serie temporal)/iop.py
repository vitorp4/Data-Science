import matplotlib.pyplot as plt
import numpy as np

def iop_rmsd(type, metrics, central):

    n_horizons = metrics[0].shape[1]
    iop = []

    for h in range(1,n_horizons+1):
        pers_stat = metrics[0].loc[type,f't+{h}']
        model_stat = metrics[1].loc[type,f't+{h}']
        iop = np.append(iop,100*(pers_stat-model_stat)/pers_stat)

    plt.figure(figsize=(8, 2.5))
    plt.plot(np.concatenate(([np.nan],iop)),marker='x',ms=5,label="MLP")
    plt.xticks(range(1,13))
    plt.grid(axis='y',ls='-.')
    plt.yticks(range(-5,46,10))
    plt.ylabel('IoP (%) - ' + type)
    plt.xlabel('Horizonte')
    plt.legend()
    plt.title("Central " + central)
    plt.savefig("iop_" + central + ".png", dpi=300, bbox_inches='tight')