import matplotlib.pyplot as plt 
import numpy as np
import copy
import cmocean
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator



def get_group_idx(group_string,ds):
    return ds.groupby('id').groups[group_string]

def get_groups(ds):
    return list(ds.groupby('id').groups.keys())

def show_sample(ds):
    fig,axes = plt.subplots(1,5,figsize=(15,7.5))
    fig.set_facecolor('w')
    cmaps = ['Blues','turbo','Spectral_r','Greys_r']

    vmins = [-2.5,-2.5,0,0]
    vmaxs = [2,2,20,4]

    for i,ax in enumerate(axes[0:4]):
        ax.imshow(ds.features[:,:,i],cmap=cmaps[i],vmin=vmins[i],vmax=vmaxs[i])
        ax.axis('off')
        ax.set_title(ds.n_channel[i].values)

    ax = axes[4]
    ax.imshow(np.log10(ds.label_2d_reg[:,:]),cmap='magma')
    ax.axis('off')
    ax.set_title('glm')
    
    fig.suptitle('class label: {}, n flashes: {}'.format(ds.label_1d_class.values,ds.label_1d_reg.values),y=0.75)

    return ds.id.values

def make_plot_saliency(image, mask=None, maskalpha=0.7,
              cmap=cmocean.cm.balance,
              title=None):

    fig,axes = plt.subplots(nrows=2,ncols=4, figsize=(10,5), dpi=170)
    fig.set_facecolor('w')

    titles = ['WV','IR','VIL','VIS']
    vmins = [-2.5,-2.5,0,0]
    vmaxs = [2,2,20,4]

    cmaps = ['Blues','turbo','Spectral_r','Greys_r']
    top_row = axes[0,:]    
        
    for i,ax in enumerate(top_row):   
        ax.imshow(image[:,:,i],
                  cmap=cmaps[i],vmin=vmins[i],vmax=vmaxs[i], alpha=0.9, aspect='auto')
        ax.axis('off')
    last_row = axes[1,:]
    for i,ax in enumerate(last_row):
        im = ax.imshow(mask[:,:,i], cmap=cmocean.cm.balance, alpha=0.8)
        color =  'k'
        
        ax.annotate(f'Saliency min = {np.min(mask[:,:,i]): 0.03e}', 
                    (0.1, 0.1), xycoords='axes fraction', color=color, fontsize=6, 
                   bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=0.5'))
        ax.annotate(f'Saliency max = {np.max(mask[:,:,i]): 0.03e}', 
                    (0.1, 0.2), xycoords='axes fraction', color=color, fontsize=6, 
                   bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=0.5'))
        
        ax.axis('off')
        ax.set_title(titles[i], fontsize=15)
    
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()

    return ax


def make_plot(image, mask=None, maskalpha=0.7,
              cmap=cmocean.cm.balance,normalize=False, 
              mask_thresh=None, title=None, mode='SHAP'):

    fig,axes = plt.subplots(nrows=2,ncols=2, figsize=(8,7), dpi=170)
    fig.set_facecolor('w')

    titles = ['WV','IR','VIL','VIS']
    vmins = [-2.5,-2.5,0,0]
    vmaxs = [2,2,20,4]

    cmaps = ['Blues','turbo','Greys','Greys_r']
    
    vmax = round(np.nanpercentile(mask, 99.9),4)
    vmin = -vmax
    
    levels = MaxNLocator(nbins=11).tick_values(vmin,vmax)
    
    ##levels = np.linspace(vmin, vmax, 11)
    cmap = plt.colormaps['seismic']       
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
    if mask_thresh is not None:
        mask = np.ma.masked_where(abs(mask)<=mask_thresh, mask)
        title=fr'Abs({mode} Values) $\geq$ {mask_thresh}'

    for i,ax in enumerate(axes.flat):
        if i == 2:
            im = np.ma.masked_where(image[:,:,i] <= 0, image[:,:,i])
            alpha=0.7
        else:
            im = image[:,:,i]
            alpha=0.8
            
        ax.imshow(im, cmap=cmaps[i],vmin=vmins[i],vmax=vmaxs[i], alpha=alpha, aspect='auto')
    
        if normalize:
            im = ax.pcolormesh(mask[:,:,i], cmap='seismic', alpha=1.0, norm=norm)
        else:
            mdata = np.ma.filled(mask[:,:,i], np.nan)
            vmax = round(np.nanpercentile(mdata, 99.9),4)
            vmin = -vmax
            levels = MaxNLocator(nbins=11).tick_values(vmin,vmax)
            norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
            im = ax.pcolormesh(mask[:,:,i], cmap='seismic', alpha=1.0, norm=norm)
            fig.colorbar(im, ax=ax, shrink=0.9, label=f'{mode} Values')
        
        color = 'k' if i < 3 else 'w'
        
        ax.annotate(f'{mode} min = {np.min(mask[:,:,i]): 0.03f}', 
                    (0.1, 0.9), xycoords='axes fraction', color=color, fontsize=8)
        ax.annotate(f'{mode} max = {np.max(mask[:,:,i]): 0.03f}', 
                    (0.1, 0.85), xycoords='axes fraction', color=color, fontsize=8)
        
        ax.axis('off')
        ax.set_title(titles[i], fontsize=15)

    # Add colorbar
    if normalize:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9, label=f'{mode} Values')    
    
    
    if title is not None:
        fig.suptitle(title)
    #plt.tight_layout()

    return ax

def minmax(X):
  X_new = copy.deepcopy(X)
  for i in np.arange(0,4):
    X_new[:,:,i] = (X_new[:,:,i] - np.min(X_new[:,:,i])) / (np.max(X_new[:,:,i]) - np.min(X_new[:,:,i]))
  return X_new

def standardanom(X,axis=True):
  X_new = copy.deepcopy(X)
  if (X_new.shape[-1] == 4) and (axis):
    for i in np.arange(0,4):
      X_new[:,:,i] = (X_new[:,:,i] - np.mean(X_new[:,:,i])) / np.std(X_new[:,:,i])
  elif (X_new.shape[-1] == 4):
      mu = np.mean(X_new)
      sigma = np.std(X_new)
      X_new = (X_new - mu) / sigma
  else:
    X_new = (X_new - np.mean(X_new)) / np.std(X_new)

  return X_new

