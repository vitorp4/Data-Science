import skill_metrics as sm
import numpy as np
import matplotlib.pyplot as plt

def diagram(std_obs, metrics, names, title, savename):

    n_models = len(metrics)
    n_horizons = metrics[0].shape[1]

    STD = [std_obs] + [metrics[i].loc['std_ratio',f't+{h}']*std_obs for i in range(n_models) for h in range(1,n_horizons+1)]
    RMSD = [0] + [metrics[i].loc['rmsd',f't+{h}'] for i in range(n_models) for h in range(1,n_horizons+1)]
    CORR = [1] + [metrics[i].loc['corr_coef',f't+{h}'] for i in range(n_models) for h in range(1,n_horizons+1)]

    STD = np.array(STD)
    RMSD = np.array(RMSD)
    CORR = np.array(CORR)
    
    markerLabel = ['O'] + [f'{names[i]}-{h}' for i in range(n_models) for h in range(1, n_horizons+1)]

    plt.figure(figsize=(15, 8))
    plt.rc('font', size=14)

    sm.taylor_diagram(STD, RMSD, CORR, markerobs = 'o', styleobs = '-', 
                    titleobs = 'obs', colcor = 'k', 
                    widthcor= 0.5, widthrms = 1.3, colrms = 'g', 
                    markerLabel = markerLabel,
                    alpha=0.7, checkStats='on', tickrmsangle=135, 
                    markersize=10, colobs='#fc03df')
    plt.title(title)

    plt.savefig(f"{savename}.png")