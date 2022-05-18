# -*- coding: utf-8 -*-
#
#****************************************************************************#
#                                                                            #
#  Copyright (c) 2015, by University of Birmingham. All rights reserved.     #
#                                                                            #
#  Redistribution and use in source and binary forms, with or without        #
#  modification, are permitted provided that the following conditions        #
#  are met:                                                                  #
#                                                                            #
#      * Redistributions of source code must retain the above copyright      #
#        notice, this list of conditions and the following disclaimer.       #
#      * Redistributions in binary form must reproduce the above copyright   #
#        notice, this list of conditions and the following disclaimer in the #
#        documentation and/or other materials provided with the distribution.#
#      * The name 'University of Birmingham' may not be used to endorse or   #
#        promote produces derived from this software without specific prior  #
#        written permission.                                                 #
#                                                                            #
#****************************************************************************#
#
"""
Modified Taylor diagram [Elvidge 2014, Taylor 2001] implementation code.
A set of classes and functions for constructiong a "modified Taylor diagram".
A diagram which plots the model standard deviation, correlation,
bias, error standard deviation and mean square error to observation data
with r=stddev and theta=arccos(correlation).
This code can be used for creating your own diagrams, and an example use
is shown at the bottom of the program. If you use this approach for model
comparison we ask that you reference:
Elvidge, S., M. J. Angling, and B. Nava (2014), On the Use of Modified Taylor 
Diagrams to Compare Ionospheric Assimilation Models, Radio Sci., 
doi:10.1002/2014RS005435.
The orignal Taylor paper is:
Taylor, K. E. (2001), Summarizing multiple aspects of model performance in a 
single diagram, J. Geophys. Res., 106(7), 7183–7192.
The format of use is (a more detailed example can be found at bottom of 
the file):
>>> dia = ModifiedTaylorDiagram(refstd, label="Observation")
>>> colors = dia.calc_colors(bias_data)
>>> cbar = dia.add_colorbar(fig, bias_data.min(), bias_data.max())
>>> contours = dia.add_contours(colors='darkblue', levels=np.linspace(0,1.5,7))
>>> plt.clabel(contours, contours.levels[0:len(contours.levels)-2],
               inline=1, fontsize=10, use_clabeltext=1)
>>> plt.text(0.1, 0.9, 'Error Std. Dev.', transform=dia.ax.transAxes, 
             rotation=30, color='darkblue', fontsize=10)
>>> for i,(stddev,corrcoef,bias) in enumerate(data):
        dia.add_point(stddev, corrcoef, mod_lab[i], marker='o', markersize=10., 
                      ls='', c=cm.jet(colors[i]), label=labels)
>>> legend = dia.add_legend(str_artist, labels, prop=dict(size='small'), 
                            loc=(0.8,0.9))
>>> plt.show()
The code is split up in suich as way to give the user as much flexible in 
presentation (color etc. ) as wanted. 
Modification History
-------
Created on Mon Nov 01 2014 by Sean at SERENE, University of Birmingham
Contact: serene@contacts.bham.ac.uk
This code is an expansion of an original piece of work by Yannick Copin 
(yannick.copin@laposte.net) created on 2012-02-17.
03/08/15  Added 'offset' variable to add_point, and 'fontsize' to add_colorbar
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.floating_axes as fa
import mpl_toolkits.axisartist.grid_finder as gf
import matplotlib.colors as mpl_colors
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

class ArtistObject(object):
    def __init__(self, text):
        self.my_text = text

class LegendHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        import matplotlib.colors as mpl_colors
        import matplotlib.text as mpl_text
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpl_text.Text(x=0, y=0, text=orig_handle.my_text, 
                              verticalalignment=u'baseline', 
                              horizontalalignment=u'left', multialignment=None, 
                              fontproperties=None, rotation_mode=None)      
        
        handlebox.add_artist(patch)
        return patch
        

class ModifiedTaylorDiagram(object):
    """
    Set up the modified Taylor diagram axes, i.e. single quadrant polar
    plot, using mpl_toolkits.axisartist.floating_axes. 
    
    Parameters
    ----------
    datastd : float
        Standard deviation of the observation data set
    fig : figure object
        A figure where the diagram is to be plotted.
        The default is none, in which case a new figure is created.
    rect : int
        For creating subplots in the figure with the given grid definition.
        Default is 111, i.e. no subplot.
    normalize : True/False
        Determines whether the data should be normalized by datastd. Default is True. 
    sd_axis_frac : float
        Scale of std. dev. axis (as a fraction of datastd)
        Default is 1.5
    fontsize : int
        Size of font for axis labels
        Default is 12
    """
    def __init__(self, datastd, fig=None, rect=111, normalize=True, 
                 sd_axis_frac=1.5, fontsize=10):
        
        # Check if we are going to be normalizing values or not, and set
        # values appropriately.
        if normalize:        
            self.datastd = datastd / datastd
            self.normfactor = datastd
        else:
            self.datastd = datastd           
            self.normfactor = 1
        
        # Standard deviation axis extent
        self.smin = 0
        self.smax = sd_axis_frac*self.datastd

        tr = PolarAxes.PolarTransform()

        # Correlation labels
        rlocs = np.concatenate((np.arange(10)/10.,[0.95,0.99]))
        tlocs = np.arccos(rlocs)        # Conversion to polar angles
        gl1 = gf.FixedLocator(tlocs)    # Positions
        tf1 = gf.DictFormatter(dict(zip(tlocs, map(str,rlocs))))
        # Std. Dev. labels
        sdlocs = np.linspace(self.smin,self.smax,7)
        
        # Round the output (particularly import when normalize='N')
        sdlocs = [round(sd, -int(np.floor(np.log10(sd)))+2) for sd in sdlocs[1:]]
        sdlocs.append(0)
        
        gl2 = gf.FixedLocator(sdlocs)
        tf2 = gf.DictFormatter(dict(zip(sdlocs, map(str,sdlocs))))

        self.ghelper = fa.GridHelperCurveLinear(tr,
                                           extremes=(0,np.pi/2, # 1st quadrant
                                                     self.smin,self.smax),
                                           grid_locator1=gl1,
                                           grid_locator2=gl2,
                                           tick_formatter1=tf1,
                                           tick_formatter2=tf2
                                           )

        fig, ax = plt.subplots(dpi=200, subplot_kw={'projection': 'polar'})
        fig.set_facecolor("w")
        ax.set_xlim([0, np.pi/2])
        
        # Check if an existing figure has been passed to the program
        #if fig is None:
        #    fig = plt.figure(dpi=300, figsize=(6,6))
         
        #fig, ax = plt.subplots(dpi=300)
        
        #ax = fa.FloatingSubplot(fig, rect, grid_helper=self.ghelper)
        #fig.add_subplot(ax)
        
        # Adjust axes
        """
        ax.axis["top"].set_axis_direction("bottom")  # Azimuthal axis
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].major_ticklabels.set_color("darkgreen")
        ax.axis["top"].label.set_color("darkgreen")
        ax.axis["top"].label.set_text("Correlation")
        ax.axis["top"].label.set_fontsize(fontsize)

        ax.axis["left"].set_axis_direction("bottom") # X axis
        ax.axis["left"].toggle(ticklabels=True)

        ax.axis["right"].set_axis_direction("top")   # Y axis
        ax.axis["right"].toggle(ticklabels=True, label=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")
        if normalize:
            ax.axis["right"].label.set_text("Normalized standard deviation")
        else:
            ax.axis["right"].label.set_text("Standard deviation")
            
        ax.axis["right"].label.set_fontsize(fontsize)
        ax.axis["bottom"].set_visible(False)  # Don't want it
        """
        trans, _ , _ = ax.get_xaxis_text1_transform(-10)
        ax.text(np.deg2rad(45), -0.18, "Correlation", transform=trans, 
         rotation=45-90, ha="center", va="center", color='k')
        
        
        # Contours for each standard deviation
        ax.grid(False)

        if normalize:
            ax.set_ylabel("Normalized standard deviation")
        else:
            ax.set_ylabel("Standard deviation")
        
        
        self._ax = ax                   # Graphical axes
        self.ax = ax
        #self.ax = ax.get_aux_axes(tr)   # Polar coordinates

        # Add reference point and ref stddev contour
        l, = self.ax.plot([0], self.datastd, 'k*',
                          ls='', ms=10)
        t = np.linspace(0, np.pi/2)
        r = np.zeros_like(t) + self.datastd
        self.ax.plot(t,r, 'k--', label='_', alpha=0.55)

        # Collect data points for latter use (e.g. legend)
        self.samplePoints = [l]
        
        #Add "correlation lines"
        #corr_labels = np.concatenate((np.arange(1,10)/10.,[0.99]))
        corr_labels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        for i in corr_labels:
            self.ax.plot([np.arccos(i),np.arccos(i)], [0,self.smax],
                          c='k',alpha=0.25, lw=0.75)
        
    
        ax.set_xticks(np.arccos(corr_labels))
        ax.set_xticklabels(corr_labels, color='k')
        ##ax.set_xlabel('Correlation')
        
                
    def add_point(self, stddev, corrcoef, stddev_err=None, cc_err=None, modlab=None, offset=(8,-12), 
                  *args, **kwargs):
        """
        Add point (stddev, corrcoeff) to the modified Taylor diagram.
        
        The "original" (non-normalized) values should be given to the function.
        If the "normalize" flag was passed when setting up the mTd then the
        values will be normalized in this function.
        
        Parameters
        ----------
        stddev : float
            The standard deviation of the point
        corrcoef : float
            The correlation coefficient of the data point to be plotted.
        modlab : string
            The name of the annonation for the point.
        offset : 2-tuple
            The text offset for the labels.
            Default is (-12,-12)
        *args *kwargs :
            Arguments and keyword arguments to pass to the plt.plot command.
            For example the label, for when constructing the legend later on.
        """
        # Normalize data if necessary
        stddev = stddev / self.normfactor
        
        # Plot the point.
        #l, = self.ax.plot(np.arccos(corrcoef), stddev,
        #                  *args, **kwargs) # (theta,radius)
        
        l = self.ax.errorbar(np.arccos(corrcoef), stddev,
                              yerr=stddev_err, xerr=cc_err, 
                          *args, **kwargs) # (theta,radius)
        
        
        self.samplePoints.append(l)

        # Label the point
        #xoff = np.random.choice([7, 8, 10])
        #yoff = np.random.choice([-8, -10, -12]) 
        
        #self.ax.annotate(modlab, xy = (np.arccos(corrcoef),stddev), 
        #                 xytext = (xoff, yoff), textcoords = 'offset points', fontsize=9, 
        #                color = 'xkcd:bright orange')
        
        return l

    def add_contours(self, levels=6, **kwargs):
        """
        Add the error standard deviation contours.
        
        Parameters
        ----------
        levels : int
            The number of error std. dev. semi-circles to show.
        **kwargs :
            Keyword agruments accepted in plt.contour
        """

        rs,ts = np.meshgrid(np.linspace(self.smin,self.smax),
                            np.linspace(0,np.pi/2))
        # Compute error std dev
        rms = np.sqrt(self.datastd**2 + rs**2 - 2*self.datastd*rs*np.cos(ts))

        contours = self.ax.contour(ts, rs, rms, levels, alpha=0.65,  
                                   **kwargs)

        return contours

    def calc_colors(self, bias):
        """
        Calculate the colours for the symbols when plotting their bias.
        
        This should be the original values, not normalized values.
        
        Parameters        
        ----------
        bias : array
            The bias values for the models, compared to the observation.
        """
        # Normalize the bias values if necessary
        bias = bias / self.normfactor
        
        # Set the values to the range [0,1] for colouring
        #largest = np.array([abs(bias.min()),abs(bias.max())]).max()
        largest = 8.0
        # Subtraction done, because color 0 = color 1
        bias_col = (bias + largest)/(2*largest) - 1e-10
        
        return bias_col
        
    def add_colorbar(self, bias, num_ticks=7, fontsize=10):
        """
        Add the colorbar to the diagram.
        
        Parameters
        ----------
        bias : array
            The (non-normalized) bias values for the models, compared to the 
            observation.
        num_ticks : int
            The number of ticks for the color bar.
            Default is 5.
        fontsize : int
            The fontsize for the 'Bias' label. 
            Default is 10.
        """
        #ax = plt.gca()
        
        # Normalize the bias values
        bias = bias / self.normfactor
        
        largest = np.ceil(bias.max())        
        largest = np.array([abs(bias.min()),abs(bias.max())]).max()
        #largest = 8.0 
        cnorm = mpl_colors.Normalize(vmin=-largest,vmax=largest)
        tick_locator = ticker.MaxNLocator(nbins=num_ticks)
        
        # Plot the colorbar
        sm = plt.cm.ScalarMappable(norm=cnorm, cmap=cm.rainbow)
        # The array for ScalarMappable has to be faked :(
        sm._A=[]
        
        # Label the colorbar
        title = 'Bias' if self.normfactor == 1 else 'Normalized Bias'
    
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes('right', size='5%', pad=0.05, grid_helper=self.ghelper)
        #fig = ax.get_figure()
        #cbar = fig.colorbar(sm, cax=cax, label=title)
        
        cbar = plt.colorbar(sm,
                            orientation='horizontal',
                            fraction=0.042,pad=0.12)
        
        
        # Update the number of ticks
        cbar.locator = tick_locator
        cbar.update_ticks()
        
        # Remove colorbar container frame (just for style):
        cbar.outline.set_visible(False)
        cbar.set_label(title, fontsize=fontsize)


    #def add_legend(self, artist, labels, *args, **kwargs):
    def add_legend(self, *args, **kwargs):   
        """
        Add the legend to the mTd.
        
        Python legends do not accept strings for the artist by default
        so a custom handler_map is required.
        
        Parameters
        ----------
        artist : string list
            The list of the annonations to be put in the legend.
        legend : string list
            The labels for the artist to put on the legend.
        """
        # Define new aritst objects
        #obj = [ArtistObject(st) for st in artist]
        #if self.normfactor != 1:
        #    if np.float(self.normfactor) > 0.01:
        #        obj.append(ArtistObject("{0:4.2f}".format(self.normfactor)))
        #        labels.append('     Norm factor')
        #    else:
        #        obj.append(ArtistObject("{0:4.2e}".format(self.normfactor)))
        #        labels.append('        Norm factor')
        # Create a dictionary with the arists and the new handler_map
        #dic = dict.fromkeys(obj, LegendHandler())
    
        # Add the legend
        #self.ax.legend(obj, labels, handler_map = dic, *args, **kwargs)
        
        self.ax.legend(*args, **kwargs) 

if __name__=='__main__':
    import string
    
    # "Fake" data/observation
    x = np.linspace(0,4*np.pi,100)
    data = np.sin(x)
    
    datastd = data.std(ddof=1)           # Observation standard deviation

    # Models
    m1 = data + 0.2*np.random.randn(len(x))          # Model A
    m2 = (0.8*data + .1*np.random.randn(len(x))) - 2 # Model B
    m3 = np.sin(x-np.pi/10)                          # Model C
    m4 = np.roll(data*1.1,10) + 1                    # Model D

    #Labels for the models
    model_label = list(string.ascii_uppercase)[0:4]

    # Compute stddev, correlation coefficient and bias of models
    points = np.array([ [m.std(ddof=1), np.corrcoef(data, m)[0,1], 
                         m.mean()-data.mean()] for m in (m1,m2,m3,m4)])
    # points[:,0] is the std. dev.
    # points[:,1] is the correlation
    # points[:,2] is the bias
    
    # If you want to save the figure, I've found that passing
    # fig=plt.figure(figsize(5.5,3.5)) to the ModifiedTaylorDiagram class
    # gives you a nice result.
    
    # Set up the mTd
    dia = ModifiedTaylorDiagram(datastd, normalize='Y', fontsize=14, fig=plt.figure(figsize=(8., 8.)))
                                
    #Calcuate the colours for the points on the diagram, by passing the bias                             
    colors = dia.calc_colors(points[:,2])

    # Add colorbar to the diagram
    cbar = dia.add_colorbar(points[:,2], fontsize=14)

    # Add the error std. dev. contours, and label them
    contours = dia.add_contours(colors='darkblue',levels=np.linspace(0,1.5,7),
                                linestyles='dotted')
    plt.clabel(contours, contours.levels[0:len(contours.levels)-2],
               inline=1, use_clabeltext=1)
    plt.text(0.05, 0.93, 'Model Error Std. Dev.', transform=dia.ax.transAxes, 
             rotation=30, color='darkblue', fontsize=12)

    # Add points to modified Taylor diagram
    for i,(stddev,corrcoef,bias) in enumerate(points):
        dia.add_point(stddev, corrcoef, model_label[i], marker='o', 
                      markersize=10., c=cm.jet(colors[i]), 
                      label="Model %s" % list(string.ascii_uppercase)[i])

    # Add a legend to the diagram
    dia.add_legend(model_label, ["Model %s" 
                   % list(string.ascii_uppercase)[i] for i,j in 
                   enumerate(model_label)], prop=dict(size='small'), 
                   loc=(0.8,0.9))

    plt.savefig('test_taylordiagram.png', dpi=300, facecolor='w',
            edgecolor='w')                        
