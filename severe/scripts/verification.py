#import shapely.geometry

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np 
#from descartes import PolygonPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import warnings

from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, f1_score, brier_score_loss
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from scipy import interpolate

#import warnings
#from shapely.errors import ShapelyDeprecationWarning
#warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 

def plot_verification(estimators, X, y, 
                      baseline_estimator=None, 
                      X_baseline=None, 
                      n_boot=10, style='classification'):
    """Plot Classification- or Regression-based verification."""
    verify = VerificationDiagram()
    fig, axes = plt.subplots(dpi=300, figsize=(8,8), ncols=2, nrows=2)
    
    if style == 'classification':
        metrics = ['reliability', 'roc', 'performance']
    elif style == 'regression': 
        metrics = ['taylor']
    
    for ax, metric in zip(axes.flat, metrics):

        xp = {}
        yp = {} 
        scores = {}
        pred = {}
        for name, model in estimators:
            if style == 'classification':
                predictions = model.predict_proba(X)[:,1]
            else:
                predictions = model.predict(X)
                
            _x, _y, _scores = sklearn_curve_bootstrap(
                                    y, 
                                    predictions, 
                metric=metric,
                n_boot=n_boot)
    
            xp[name] = _x
            yp[name] = _y
            pred[name] = predictions
            scores[name] = _scores
       
        if baseline_estimator is not None:
            for name, model in estimators:
                predictions = model.predict(X)
                
                _x, _y, _scores = sklearn_curve_bootstrap(
                                    y, 
                                    predictions, 
                metric=metric,
                n_boot=n_boot)
    
                xp[name] = _x
                yp[name] = _y
                pred[name] = predictions
                scores[name] = _scores
    
        verify.plot(diagram=metric, x=xp, y=yp, ax=ax, scores=scores, pred=pred)
    
    axes.flat[-1].remove()
    plt.subplots_adjust(wspace=0.2)

class VerificationDiagram:
    mpl.rcParams["axes.titlepad"] = 15
    mpl.rcParams["xtick.labelsize"] = 10
    mpl.rcParams["ytick.labelsize"] = 10
    
    def _add_major_and_minor_ticks(self, ax):
        """Add minor and major tick marks"""
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5, prune="lower"))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5, prune="lower"))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
    
    def _set_axis_limits(self, ax,):
        """Sets the axis limits"""
        ax.set_ylim([0,1])
        ax.set_xlim([0,1])
    
    def _make_reliability(self, ax, **diagram_kwargs):
        """
        Make the Receiver Operating Characterisitc (ROC) Curve.
        """
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.plot([0,1], [0,1], ls='dashed', color='k', alpha=0.7)
        ax.set_xlabel('Mean Forecast Probability')
        ax.set_ylabel('Conditional Event Frequency')
        
        return ax 
    
    
    def _plot_inset_ax(self, ax, pred, line_colors, inset_yticks = [1e1, 1e3] ): 
        """Plot the inset histogram for the attribute diagram."""
        import math
        def orderOfMagnitude(number):
            return math.floor(math.log(number, 10))
        
        mag = orderOfMagnitude(len(pred))
        # Check if the number is even. 
        if mag % 2 == 0:
            mag+=1 

        inset_yticks = [10**i for i in range(mag)]
        inset_ytick_labels = [f'{10**i:.0e}' for i in range(mag)]
    
        # Histogram inset
        small_ax = inset_axes(
            ax,
            width="50%",
            height="50%",
            bbox_to_anchor=(0.15, 0.58, 0.5, 0.4),
            bbox_transform=ax.transAxes,
            loc=2,
        )

        small_ax.set_yscale("log", nonpositive="clip")
        small_ax.set_xticks([0, 0.5, 1])
        #small_ax.set_yticks(inset_yticks)
        #small_ax.set_yticklabels(inset_ytick_labels)
        #small_ax.set_ylim([1e0, np.max(inset_yticks)])
        small_ax.set_xlim([0, 1])
        
        bins=np.arange(0, 1.1, 0.1)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        
        for k, color in zip(pred.keys(), line_colors):
            p = pred[k]
            fcst_probs = np.round(p, 5)
            n, x = np.histogram(a=fcst_probs, bins=bins)
            n = np.ma.masked_where(n==0, n)
            small_ax.plot(bin_centers, n, color=color, linewidth=0.6)
        
        return small_ax 
        
    
    def _make_roc(self, ax, **diagram_kwargs):
        """
        Make the Receiver Operating Characterisitc (ROC) Curve.
        """
        pss_contours = diagram_kwargs.get('pss_contours', True)
        cmap = diagram_kwargs.get('cmap', 'Blues')
        alpha = diagram_kwargs.get('alpha', 0.6)
        
        x=np.arange(0,1.1,0.1)
        if pss_contours:
            # Compute the Pierce Skill Score (PSS)
            pod,pofd=np.meshgrid(x,x)
            pss = pod-pofd
            contours = ax.contourf(pofd, pod, pss, levels=x, cmap=cmap, alpha=alpha)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig = ax.get_figure()
            fig.colorbar(contours, cax=cax, label='Pierce Skill Score (POD-POFD)')
            
        # Plot random classifier/no-skill line 
        ax.plot(x,x,linestyle="dashed", color="gray", linewidth=0.8)
        
        return ax, contours 

    def _make_performance(self, ax, **diagram_kwargs):
        """
        Make a performance diagram (Roebber 2009). 
        """
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        xx = np.linspace(0.001,1,100)
        yy = xx
        xx,yy = np.meshgrid(xx,xx)
        csi = 1 / (1/xx + 1/yy -1)
        cf = ax.contourf(xx,yy,csi, cmap='Blues', alpha=0.3, levels=np.arange(0,1.1,0.1))
        ax.set_xlabel('Success Ratio (SR; 1-FAR)')
        ax.set_ylabel('Probability of Detection (POD)')
        biasLines = ax.contour(
                    xx,
                    yy,
                    yy/xx,
                    colors="k",
                    levels=[0.5, 1.0, 1.5, 2.0, 4.0],
                    linestyles="dashed",
                    linewidths=0.5,
                    alpha=0.9
                    )
        ax.clabel(biasLines, levels=[0.5, 1.0, 1.5, 2.0, 4.0], fontsize=6, inline=True, fmt="%1.1f")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig = ax.get_figure()
        fig.colorbar(cf, cax=cax, label='Critical Success Index (CSI)')

        return ax, cf 
        
    def plot(self, diagram, x, y,
             pred=None, 
             add_dots=True, 
             scores=None, 
             add_high_marker=False,
             line_colors=None,
             diagram_kwargs={}, 
             plot_kwargs={}, ax=None): 
        """
        Plot a performance, attribute, or ROC Diagram. 
        
        Parameters
        ---------------
            diagram : 'performance', 'roc', or 'reliability'
            
            x,y : 1-d array, 2-d array or dict 
                The X and Y coordinate values. When plotting multiple 
                curves, then X and Y should be a dictionary. 
                E.g., x = {'Model 1' : x1, 'Model 2' : x2}
                      y = {'Model 1' : y1, 'Model 2' : y2}
                      
                If x or y are 2-d array, it is assumed the first
                dimension is from bootstrapping and will be used 
                to create confidence intervals. 
                      
            add_dots : True/False
            
            add_table : True/False
        """
        plot_kwargs['color'] = plot_kwargs.get('color', 'r')
        plot_kwargs['alpha'] = plot_kwargs.get('alpha', 0.7)
        plot_kwargs['linewidth'] = plot_kwargs.get('linewidth', 1.5)
        
        line_colors = ['r', 'b', 'g', 'k']
        
        if ax is None:
            mpl.pyplot.subplots
            f, ax = plt.subplots(dpi=600, figsize=(4,4))
        
        self._set_axis_limits(ax)
        self._add_major_and_minor_ticks(ax)
        
        contours=None
        for_performance_diagram=False
        if diagram == 'performance':
            for_performance_diagram=True
            ax, contours = self._make_performance(ax=ax, **diagram_kwargs)
        elif diagram == 'roc':
            ax, contours = self._make_roc(ax=ax, **diagram_kwargs)
        elif diagram == 'reliability':
            ax = self._make_reliability(ax=ax, **diagram_kwargs)
            if pred is not None:
                self._plot_inset_ax(ax, pred, line_colors)
            
        else:
            raise ValueError(f'{diagram} is not a valid choice!')
    
        if not isinstance(x, dict):
            x = {'Label' : x}
            y = {'Label' : y}
        
        keys = x.keys()
        
        error_bars=False
        for line_label, color in zip(keys, line_colors):
            _x = x[line_label]
            _y = y[line_label]
            plot_kwargs['color'] = color
            
            if _x.ndim == 2:
                error_bars=False
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    _x = np.nanmean(_x, axis=0)
                    _y = np.nanmean(_y, axis=0)
                    
            line_label = None if line_label == 'Label' else line_label
    
            ax.plot(_x, _y, label=line_label,**plot_kwargs)

            if diagram in ['roc', 'performance'] and add_dots:
                # Add scatter points at particular intervals 
                ax.scatter(_x[::10], _y[::10], s=15, marker=".", **plot_kwargs)
            
            
            if add_high_marker:
                if diagram == 'roc':
                    highest_val = np.argmax(_x - _y)
                else:
                    highest_val = np.argmax(csi)
            
                ax.scatter(
                        _x[highest_val],
                        _y[highest_val],
                        s=65,
                        marker = "X", 
                        **plot_kwargs, 
                        )
            
        if error_bars:
            # Adds the 95% confidence interval.
            for line_label, color in zip(keys, line_colors):
                _x = x[line_label]
                _y = y[line_label]
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    x_coords_bottom, x_coords_top = np.nanpercentile(_x, (2.5, 97.5), axis=0)
                    y_coords_bottom, y_coords_top = np.nanpercentile(_y, (2.5, 97.5), axis=0)
                
                polygon_object = _confidence_interval_to_polygon(
                    x_coords_bottom,
                    y_coords_bottom,
                    x_coords_top,
                    y_coords_top,
                    for_performance_diagram=for_performance_diagram,
                )   
            
                polygon_colour = mpl.colors.to_rgba(color, 0.4)
            
                polygon_patch = PolygonPatch(
                    polygon_object, lw=0, ec=polygon_colour, fc=polygon_colour
                )   
            
                ax.add_patch(polygon_patch)
        
        if scores is not None:
            if diagram == 'performance':
                loc = 'upper center'
            elif diagram == 'reliability':
                loc = 'lower right'
            elif diagram == 'roc':
                loc = 'center right'
            
            
            table_data, rows, columns = to_table_data(scores)
            
            rows = [f' {r} ' for r in rows]
            
            add_table(ax, table_data,
                    row_labels=rows,
                    column_labels=columns,
                    col_colors= None,
                    row_colors = {name : c for name,c in zip(rows, line_colors)},
                    loc=loc,
                    colWidth=0.16,
                    fontsize=8)

def brier_skill_score(y_values, forecast_probabilities, **kwargs):
    """Computes the brier skill score"""
    climo = np.mean((y_values - np.mean(y_values)) ** 2)
    return 1.0 - brier_score_loss(y_values, forecast_probabilities) / climo


def sklearn_curve_bootstrap(y_true, y_pred, metric, n_boot, scorers=None, **kws):
    """Apply bootstrapping to the sklearn verification curves"""
    N = 200
    if metric == 'performance':
        func = precision_recall_curve
        if scorers is None:
            scorers = {'AUPDC' : average_precision_score,}
    elif metric == 'roc':
        func = roc_curve
        if scorers is None:
            scorers = {'AUC' : roc_auc_score}
    elif metric == 'reliability':
        N = 10
        func = reliability_curve
        if scorers is None:
            scorers = {'BSS' : brier_skill_score} 
        
    curves = []
    scores = {k : [] for k in scorers.keys()}
    for i in range(n_boot):
        idx = resample(range(len(y_true)), replace=True)
        curves.append(func(y_true[idx], y_pred[idx], **kws))
        
        for k in scorers.keys():
            scores[k].append(scorers[k](y_true[idx], y_pred[idx]))
        
    sampled_thresholds = np.linspace(0.001, 0.99, N)
    sampled_x = []
    sampled_y = []
    # assume curves is a list of (precision, recall, threshold)
    # tuples where each of those three is a numpy array
    for pair in curves:
        if metric in ['performance', 'roc']:
            x, y, threshold = pair
            x_fp = x[:-1] if metric == 'performance' else x
            y_fp = y[:-1] if metric == 'performance' else y
            
            #x = np.interp(sampled_thresholds, threshold, x_fp)
            #y = np.interp(sampled_thresholds, threshold, y_fp)
            fx = interpolate.interp1d(threshold, x_fp, fill_value='extrapolate')
            fy = interpolate.interp1d(threshold, y_fp, fill_value='extrapolate')
            
            x = fx(sampled_thresholds)
            y = fy(sampled_thresholds)
        else:
            x, y, _ = pair
        
        sampled_x.append(x)
        sampled_y.append(y)
    
    return np.array(sampled_x), np.array(sampled_y), scores



def add_table(ax, table_data, row_labels, column_labels, row_colors, col_colors, loc='best',
        fontsize=3., extra=0.75, colWidth=0.16, ):
    """
    Adds a table
    """
    #[0.12]*3
    col_colors = plt.cm.BuPu(np.full(len(column_labels), 0.1))
    the_table = ax.table(cellText=table_data,
               rowLabels=row_labels,
               colLabels=column_labels,
               colWidths = [colWidth]*len(column_labels),
               rowLoc='center',
               cellLoc = 'center' , 
               loc=loc, 
               colColours=col_colors,
               alpha=0.6,
               zorder=5
                )
    the_table.auto_set_font_size(False)
    table_props = the_table.properties()
    table_cells = table_props['children']
    i=0; idx = 0
    for cell in table_cells: 
        cell_txt = cell.get_text().get_text()

        if i % len(column_labels) == 0 and i > 0:
            idx += 1
        
        if is_number(cell_txt):
            cell.get_text().set_fontsize(fontsize + extra)
            cell.get_text().set_color(row_colors[row_labels[idx]]) 
            
        else:
            cell.get_text().set_fontsize(fontsize-extra)
            
        if cell_txt in column_labels:
            cell.get_text().set_color('k') 
            if len(cell.get_text().__dict__['_text']) > 3:
                cell.get_text().set_fontsize(fontsize-3.25)
        else:
            pass
            #cell.get_text().set_color('grey')
        
        if cell_txt in row_labels:
            cell.get_text().set_color(row_colors[cell_txt]) 
            cell.get_text().set_fontsize(fontsize)
        
        i+=1
        
    for key, cell in the_table.get_celld().items():
        cell.set_linewidth(0.25)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

    
def to_table_data(scores):
    """Convert scores to tabular data format"""
    model_names = scores.keys()
    table_data = []
    for k in model_names:
        scorer_names = scores[k].keys()
        rows=[]
        for name in scorer_names:
            rows.append(np.nanmean(scores[k][name]))
        table_data.append(rows)
    
    table_data = np.round(table_data, 2)
    
    return table_data, list(model_names), list(scorer_names)
    



def reliability_curve(targets, predictions, n_bins=10):
    """
    Generate a reliability (calibration) curve. 
    Bins can be empty for both the mean forecast probabilities 
    and event frequencies and will be replaced with nan values. 
    Unlike the scikit-learn method, this will make sure the output
    shape is consistent with the requested bin count. The output shape
    is (n_bins+1,) as I artifically insert the origin (0,0) so the plot
    looks correct. 
    """
    bin_edges = np.linspace(0,1, n_bins+1)
    bin_indices = np.clip(
                np.digitize(predictions, bin_edges, right=True) - 1, 0, None
                )

    indices = [np.where(bin_indices==i+1)
               if len(np.where(bin_indices==i+1)[0]) > 0 else np.nan for i in range(n_bins) ]

    mean_fcst_probs = [np.nan if i is np.nan else np.nanmean(predictions[i]) for i in indices]
    event_frequency = [np.nan if i is np.nan else np.sum(targets[i]) / len(i[0]) for i in indices]

    # Adding the origin to the data
    mean_fcst_probs.insert(0,0)
    event_frequency.insert(0,0)
        
    return np.array(mean_fcst_probs), np.array(event_frequency), indices

def _confidence_interval_to_polygon(
    x_coords_bottom,
    y_coords_bottom,
    x_coords_top,
    y_coords_top,
    for_performance_diagram=False,
):
    """Generates polygon for confidence interval.
    P = number of points in bottom curve = number of points in top curve
    :param x_coords_bottom: length-P np with x-coordinates of bottom curve
        (lower end of confidence interval).
    :param y_coords_bottom: Same but for y-coordinates.
    :param x_coords_top: length-P np with x-coordinates of top curve (upper
        end of confidence interval).
    :param y_coords_top: Same but for y-coordinates.
    :param for_performance_diagram: Boolean flag.  If True, confidence interval
        is for a performance diagram, which means that coordinates will be
        sorted in a slightly different way.
    :return: polygon_object: Instance of `shapely.geometry.Polygon`.
    """

    nan_flags_top = np.logical_or(np.isnan(x_coords_top), np.isnan(y_coords_top))
    if np.all(nan_flags_top):
        return None

    nan_flags_bottom = np.logical_or(
        np.isnan(x_coords_bottom), np.isnan(y_coords_bottom)
    )
    if np.all(nan_flags_bottom):
        return None

    real_indices_top = np.where(np.invert(nan_flags_top))[0]
    real_indices_bottom = np.where(np.invert(nan_flags_bottom))[0]

    if for_performance_diagram:
        y_coords_top = y_coords_top[real_indices_top]
        sort_indices_top = np.argsort(y_coords_top)
        y_coords_top = y_coords_top[sort_indices_top]
        x_coords_top = x_coords_top[real_indices_top][sort_indices_top]

        y_coords_bottom = y_coords_bottom[real_indices_bottom]
        sort_indices_bottom = np.argsort(-y_coords_bottom)
        y_coords_bottom = y_coords_bottom[sort_indices_bottom]
        x_coords_bottom = x_coords_bottom[real_indices_bottom][sort_indices_bottom]
    else:
        x_coords_top = x_coords_top[real_indices_top]
        sort_indices_top = np.argsort(-x_coords_top)
        x_coords_top = x_coords_top[sort_indices_top]
        y_coords_top = y_coords_top[real_indices_top][sort_indices_top]

        x_coords_bottom = x_coords_bottom[real_indices_bottom]
        sort_indices_bottom = np.argsort(x_coords_bottom)
        x_coords_bottom = x_coords_bottom[sort_indices_bottom]
        y_coords_bottom = y_coords_bottom[real_indices_bottom][sort_indices_bottom]

    polygon_x_coords = np.concatenate(
        (x_coords_top, x_coords_bottom, np.array([x_coords_top[0]]))
    )
    polygon_y_coords = np.concatenate(
        (y_coords_top, y_coords_bottom, np.array([y_coords_top[0]]))
    )

    return vertex_arrays_to_polygon_object(polygon_x_coords, polygon_y_coords)


def vertex_arrays_to_polygon_object(
    exterior_x_coords,
    exterior_y_coords,
    hole_x_coords_list=None,
    hole_y_coords_list=None,
):
    """Converts polygon from vertex arrays to `shapely.geometry.Polygon` object.
    V_e = number of exterior vertices
    H = number of holes
    V_hi = number of vertices in [i]th hole
    :param exterior_x_coords: np array (length V_e) with x-coordinates of
        exterior vertices.
    :param exterior_y_coords: np array (length V_e) with y-coordinates of
        exterior vertices.
    :param hole_x_coords_list: length-H list, where the [i]th item is a np
        array (length V_hi) with x-coordinates of interior vertices.
    :param hole_y_coords_list: Same as above, except for y-coordinates.
    :return: polygon_object: `shapely.geometry.Polygon` object.
    :raises: ValueError: if the polygon is invalid.
    """

    exterior_coords_as_list = _vertex_arrays_to_list(
        exterior_x_coords, exterior_y_coords
    )
    if hole_x_coords_list is None:
        return shapely.geometry.Polygon(shell=exterior_coords_as_list)

    num_holes = len(hole_x_coords_list)
    outer_list_of_hole_coords = []
    for i in range(num_holes):
        outer_list_of_hole_coords.append(
            _vertex_arrays_to_list(hole_x_coords_list[i], hole_y_coords_list[i])
        )

    polygon_object = shapely.geometry.Polygon(
        shell=exterior_coords_as_list, holes=tuple(outer_list_of_hole_coords)
    )

    if not polygon_object.is_valid:
        raise ValueError("Resulting polygon is invalid.")

    return polygon_object


def _vertex_arrays_to_list(vertex_x_coords, vertex_y_coords):
    """Converts vertices of simple polygon from two arrays to one list.
    x- and y-coordinates may be in one of three formats (see docstring at top of
    file).
    V = number of vertices
    :param vertex_x_coords: See documentation for _check_vertex_arrays.
    :param vertex_y_coords: See documentation for _check_vertex_arrays.
    :return: vertex_coords_as_list: length-V list, where each element is an
        (x, y) tuple.
    """
    num_vertices = len(vertex_x_coords)
    vertex_coords_as_list = []
    for i in range(num_vertices):
        vertex_coords_as_list.append((vertex_x_coords[i], vertex_y_coords[i]))

    return np.array(vertex_coords_as_list)