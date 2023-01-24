#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 14:08:23 2021

@author: xinlin.chen@duke.edu

Last edited 2023/01/24
"""

from matplotlib import colors, cm
from matplotlib import pyplot as plt
import numpy as np
from general_functions import Array
from typing import Union


def split_cmap(cmap_a_str,cmap_b_str,y,split,a_range=[0,1],b_range=[0,1],flip_a=False,flip_b=True):
    """Set up dual colormaps (e.g. one for pos, one for neg results).
    
    Args:
        cmap_a_str (str): name of colormap A (must be built-in colormap), e.g., 'Greys'
        cmap_b_str (str): name of colormap B, e.g., 'winter'
        y (Array[float]): data being plotted
        split (float): point at which colormap changes
        a_range (list[Union[float,int]]): range of colormap A to show
        b_range (list[Union[float,int]]): range of colormap B to show
        flip_a (bool): flip colormap A?
        flip_b (bool): flip colormap B?
    
    Returns:
        new_cmap (colors.LinearSegmentedColormap): dual colormap
    """
    cmap_a = cm.get_cmap(cmap_a_str)
    cmap_b = cm.get_cmap(cmap_b_str)
    
    a_inds, b_inds = np.where(y<-1e-10)[0], np.where(y>=-1e-10)[0]
    part_a = np.linspace(a_range[0]*(cmap_a.N-1),a_range[1]*(cmap_a.N-1),len(a_inds)).astype('int')
    part_b = np.linspace(b_range[0]*(cmap_b.N-1),b_range[1]*(cmap_b.N-1),len(b_inds)).astype('int')
    if flip_a:
        part_a = np.flip(part_a)
    if flip_b:
        part_b = np.flip(part_b)
    new_cmap = colors.ListedColormap(np.concatenate([cmap_a(part_a),
                                                     cmap_b(part_b)]))
    return new_cmap

def plot_bar(y,**kwargs):
    """
    Plot multiple variables in bar plot. Axis and figure settings can be passed
    in as dictionary. Figure setting keys have 'fig_' prepended
    Args:
        y (Array['m,n',float]): bar heights. Each column is a different data series.
    """
    width=0.7/np.shape(y)[1]
    step=width+0.5*(1-np.shape(y)[1]*width)/np.shape(y)[1]
    
    if 'bar' in kwargs:
        bar_args = kwargs.pop('bar')
    else:
        bar_args = {}
    
    for col in range(np.shape(y)[1]):
        bar_arg = {k: v[col] for k,v in bar_args.items()}
        plt.bar(np.arange(col*step,col*step+np.shape(y)[0]),y[:,col],width=width,
                **bar_arg)
    fig_kw = [key[4:] for key in kwargs if 'fig_' in key]
    fig_kwargs = {}
    for arg in fig_kw:
        fig_kwargs[arg] = kwargs.pop('fig_'+arg)
    fig = plt.gcf()
    ax = plt.gca()
    
    for key,args in kwargs.items():
        if type(args)==dict:
            getattr(plt,key)(**args)
        # Assume tuple inputs should be passed in one by one. To avoid this
        # behavior, use a list instead
        elif type(args)==tuple:
            getattr(plt,key)(*args)
        else:
            getattr(plt,key)(args)
    
    for key,args in fig_kwargs.items():
        if type(args)==dict:
             getattr(fig,key)(**args)
        else:
            getattr(fig,key)(args)