import matplotlib.pyplot as plt 
import numpy as np
import copy
import cmocean


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


def make_plot(image,mask=None,maskalpha=0.7,title=None,vmin=None,vmax=None,
              figsize=(6.4*2, 4.8*2)):

  fig,axes = plt.subplots(2,4,figsize=figsize)
  fig.set_facecolor('w')

  titles = ['WV','IR','VIL','VIS']

  axes_top = axes[0,:]

  cmaps = ['Blues','turbo','Spectral_r','Greys_r']
  for i,ax in enumerate(axes_top):
    ax.imshow(im[:,:,i],cmap=cmaps[i])
    ax.axis('off')
    ax.set_title(titles[i])

  axes_bottom = axes[1,:]

  cmaps = ['Greys','Greys','Greys_r','Greys_r']

  vmins = [-2.5,-2.5,0,0]
  vmaxs = [2,2,20,4]

  for i,ax in enumerate(axes_bottom):
    ax.imshow(im[:,:,i],cmap=cmaps[i],vmin=vmins[i],vmax=vmaxs[i])
    if mask is not None:
      if mask.shape[-1] == 4:
        if (vmin is None) or (vmax is None):
          #determine color limits so white is 0 
          absmax = np.max(np.abs(mask[:,:,i]))
          vmin_t = -1*absmax
          vmax_t = absmax
        else:
          vmin_t = vmin
          vmax_t = vmax
        pm = ax.imshow(mask[:,:,i],cmap='seismic',alpha=maskalpha,vmin=vmin_t,vmax=vmax_t)
        plt.colorbar(pm,ax=ax,shrink=0.375)
      else:
        if (vmin is None) or (vmax is None):
          #determine color limits so white is 0 
          absmax = np.max(np.abs(mask))
          vmin = -1*absmax
          vmax = absmax 
        pm = ax.imshow(mask,cmap='seismic',alpha=maskalpha,vmin=vmin,vmax=vmax)


    ax.axis('off')
    

  if title is not None:
    fig.suptitle(title,y=1)
  plt.tight_layout()

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

def make_plot(image,mask=None,maskalpha=0.7,title=None,
              cmap=cmocean.cm.balance,normalize=True):

  fig,axes = plt.subplots(3,4)

  fig.set_facecolor('w')

  titles = ['WV','IR','VIL','VIS']
  vmins = [-2.5,-2.5,0,0]
  vmaxs = [2,2,20,4]

  axes_top = axes[0,:]

  cmaps = ['Blues','turbo','Spectral_r','Greys_r']
  for i,ax in enumerate(axes_top):
    ax.imshow(image[:,:,i],cmap=cmaps[i],vmin=vmins[i],vmax=vmaxs[i])
    ax.axis('off')
    ax.set_title(titles[i])

  axes_middle = axes[1,:]

  if normalize:
    mask_orig = copy.deepcopy(mask)
    mask = standardanom(mask)
  for i,ax in enumerate(axes_middle):
    # ax.imshow(im[:,:,i],cmap=cmaps[i],vmin=vmins[i],vmax=vmaxs[i])
    if mask is not None:
      if mask.shape[-1] == 4:
        
        pm= ax.imshow(mask[:,:,i],cmap=cmap,alpha=maskalpha,vmin=-4,vmax=4)
      else:
        pm= ax.imshow(mask,cmap=cmap,alpha=maskalpha)

    ax.axis('off')
    # plt.colorbar(pm,ax=ax)

  axes_bottom = axes[2,:]

  if normalize:
    mask = standardanom(mask_orig,axis=False)
  for i,ax in enumerate(axes_bottom):
    # ax.imshow(im[:,:,i],cmap=cmaps[i],vmin=vmins[i],vmax=vmaxs[i])
    if mask is not None:
      if mask.shape[-1] == 4:
        pm= ax.imshow(mask[:,:,i],cmap=cmap,alpha=maskalpha,vmin=-4,vmax=4)
      else:
        pm= ax.imshow(mask,cmap=cmap,alpha=maskalpha)

    ax.axis('off')
    # plt.colorbar(pm,ax=ax)
    
  if title is not None:
    fig.suptitle(title,y=1)
  plt.tight_layout()

  return ax,mask
