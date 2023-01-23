#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 14:29:28 2021

@author: xinlin.chen@duke.edu

Last edited 23/01/23
"""

import numpy as np
import math
#from pyedflib import highlevel
from scipy.interpolate import griddata as gd
from custom_topomap import mneviz_topomap_fast, mneviz_topomap
from warnings import filterwarnings
from general_functions import deal
from torch import Tensor, LongTensor
from mne.io import read_raw_edf
from dataclasses import field


def findEEGCh(headers):
    """
    Finds EDF signal channels corresponding to EEG channels
    """
    bl = np.zeros(len(headers),dtype='bool')
    for idx,el in enumerate(headers):
        bl[idx] = el['label'][0:3]=='EEG'
    return bl

def getEEGCoordDF(ch_info,input_ch_names):
    """
    Get rows of Pandas dataframe (containing channel names and coordinates)
    corresponding to channel names present in 'input_ch_names'
    Args:
        ch_info (Pandas dataframe with columns 'ch_names', 'x', and 'y')
        input_ch_names (list of str)
    Returns:
        chs_to_import (Pandas dataframe: columns <ch_col>, 'x', and 'y'):
            filtered rows of ch_info corresponding to elements in input_ch_names
    """
    inds = ch_info['ch_names'].searchsorted(input_ch_names)
    return ch_info.iloc[inds]

def getEEGCoord(ch_names_dict,input_ch_names):
    """
    Get values of dict (key: channel name, value: 2D coordinates)
    corresponding to channel names in 'input_ch_names'
    Args:
        ch_names_dict (dict)
        input_ch_names (N-length list of str)
    Returns:
        ch_coords ((N,2) array): 2D EEG electrode .coordinates
    """
    ch_coords = np.zeros([len(input_ch_names),2])
    for row, ch in enumerate(input_ch_names):
        ch_coords[row,:] = ch_names_dict.get(ch)
    return ch_coords

def getGraphEdgeMatrix(node_posns):
    """From mx2 array of 2D node positions, get mxm matrix of node-node edges
    """
    # For EEG, this is electrode-electrode distances from eegkit cap coordinates
    metric = np.zeros([len(node_posns),len(node_posns)])
    for ind in range(len(node_posns)):
        metric[ind][ind:] = np.linalg.norm(
            node_posns[ind,:]-node_posns[ind:,:],axis=1)
    # Matrix is symmetrical
    metric += metric.transpose()
    # Replace zeros in diagonal with np.nan (sorted last in np.argsort)
    # so that electrode X's distance of 0 from itself does not matter
    np.fill_diagonal(metric,np.nan)
    return metric

def getChannelsEDF(path):
    """Get channel names from EDF file. Read _read_edf_header() in mne.io for
    inspiration.
    """
    with open(path,'rb') as f:
        f.read(252) # Skip irrelevant info
        nchan = int(f.read(4).decode())
        channels = list(range(nchan))
        ch_names = [f.read(16).strip().decode() for ch in channels]
    return ch_names
        

def getSortedChSubset(sorted_ch_inds,sorted_ch_distances,channels):
    """
    For matrices 'sorted_ch_inds' and 'sorted_ch_distances', extract
    information relevant to the channel subset in 'channels'. Basically,
    since we have information about 91 electrode coordinates but our EEG
    data were recorded from a subset, we only care about finding what channels
    WITHIN that subset are closest to EACH channel in that subset.
    
    Channels in output 'n' are renumbered according to their order of
    appearance in 'channels'. E.g. sorted_ch_inds is 16x16. channels is a
    list containing channels 5, 2, and 6 in that order. n and d will be 3x3
    arrays containing values ranging from 0 to 2, where 5->0, 2->1, 6->2
    
    Args:
        sorted_ch_inds (num_total_channels,num_total_channels-1 array):
            for each row X, contains indices of channels, sorted by proximity
            to channel X
        sorted_ch_distances (num_total_channels,num_total_channels-1 array)
            for each row X, contains sorted distances between channel X and
            other channels
        channels (num_channels array): subset of channels to extract
    
    Returns:
        n (num_channels,num_channels array): subset of sorted_ch_inds
            for channels in 'channels'. Original channel indices are replaced
            according to their order of appearance in 'channels'.
        d (num_channels, num_channels array): subset of sorted_ch_distances
            for channels in 'channels'
    """
    n = np.zeros([len(channels),len(channels)],dtype='int')
    d = np.zeros([len(channels),len(channels)],dtype='float')
    for i,ch in enumerate(channels):
        bools = np.in1d(sorted_ch_inds[ch,:],channels,assume_unique=True)
        n[i,:] = sorted_ch_inds[ch,bools]
        d[i,:] = sorted_ch_distances[ch,bools]
    # Renumber channels with the negative of index, then return the negative
    # as ouput (to avoid potential renumbering clashes)
    for i,ch in enumerate(channels):
        n[n==ch]=-i
    return -n, d

def mne_readedf(path,**kwargs):
    """Return data, channel names, sampling frequency
    """
    data = read_raw_edf(path,**kwargs)
    return data._data,data.ch_names,data.info['sfreq']

def getGraphEdges(sorted_ch_inds,sorted_ch_distances,boundary=1.0, \
                  boundary_type='below',maxN=3):
    """
    Obtain EEG graph connectivity in COO (coordinate) format.
    For an 'undirected' edge, two index tuples are defined to account for 
    both directions.
    
    Args:
        sorted_ch_inds (num_nodes,num_nodes-1 array): channel neighbors.
            Each row X contains electrode channel indices, sorted by proximity
            to electrode X
        sorted_ch_inds (num_nodes,num_nodes-1 array): similar to
            sorted_ch_inds, except each element corresponds to the inter-
            channel distance in cm, rather than the channel index
        radius (float): neighborhood radius in cm
        maxN (int): maximum number of edges per node
    Returns:
        edge_index (2,num_edges LongTensor): edge-edge connections in COO
            format
        edge_attr (num_edges,1 Tensor): edge feature vector containing
            inter-channel distances
    """
    edge_index = np.zeros([2,np.shape(sorted_ch_inds)[0]*maxN],dtype='int')
    edge_attr = np.zeros(np.shape(sorted_ch_inds)[0]*maxN)
    comp_fun = np.less_equal if boundary_type == 'below' else np.greater_equal
    # Total number of edges
    numE = 0
    for ch in range(np.shape(sorted_ch_inds)[0]):
        # Number of edges for current node
        chE = 0
        # Check that neighborhood has not been exited and the number of edges
        # for this node has not gone past maxN
        while comp_fun(sorted_ch_distances[ch,chE],boundary) and chE<maxN:
            edge_index[:,numE] = [ch,sorted_ch_inds[ch,chE]]
            edge_attr[numE] = sorted_ch_distances[ch,chE]
            numE += 1
            chE += 1
    return edge_index[:,:numE],edge_attr[:numE]

def mnemap2img_fast(signals,ch_coords,sample_axis,image_interp='bicubic',map_size=50,num_cols=6):
    """
    Preferred topographic map generation method
    Convert EEG signals with known 2D channel coordinates to image using
    function adapted from mne.viz.plot_topomap() v0.23.0
    Args:
        signals ((obs,sample,channel) or (obs,channel,sample) array):
            EEG epochs
        ch_coords ((channel,2) array): EEG channel coordinates (2D)
        sample_axis (int): axes corresponding to sample
        interp (str): interpolation method ('none', 'nearest', 'bilinear',
                      'bicubic', 'spline16','spline36', 'hanning', 'hamming',
                      'hermite', 'kaiser', 'quadric','catrom', 'gaussian',
                      'bessel', 'mitchell', 'sinc', 'lanczos')
        map_size (int): square map size in pixels
        num_cols (int): number of maps per row of 'img'
    Returns:
        img ((sample//num_cols+1)*l,l*num_cols array): image containing series
            of topographic maps (1 per sample)
    """
    # Find regions to mask
    mask, circle, _ = drawCircle(map_size)
    circle *= 255
    # Generate maps for signal
    # For plots, do np.flip, axis=0
    # Min-max normalization
    mne_maps = mneviz_topomap_fast(
            signals,ch_coords,sample_axis,cmap='Greys',
            sphere=np.array([0,0,0,12.2]),vmin=0,vmax=255,res=map_size)
    map_min, map_max = np.min(mne_maps),np.max(mne_maps)
    mne_maps -= map_min
    # In case of divide by zero warnings
    filterwarnings('error','divide by zero')
    try:
        mne_maps *= 255/(map_max-map_min)
    except:
        print(f'Maps size: {np.shape(mne_maps)}, min: {map_min}, max: {map_max}')
        filterwarnings('always','divide by zero')
        mne_maps *= 255/(map_max-map_min)
    
    num_rows = math.ceil(np.shape(signals)[sample_axis]/num_cols)
    # Initialize image
    img = np.zeros([map_size*num_rows,map_size*num_cols])
    # Fill up image with masked maps
    """
    # For loop by sample
    for sample in range(np.shape(signals)[sample_axis]):
        # Masks area outside skull
        mne_maps[:,:,sample][mask] = circle
        # Assign map to image region
        img[sample//num_cols*l:(sample//num_cols+1)*l,
            (sample%num_cols)*l:(sample%num_cols+1)*l] = mne_maps[:,:,sample]
        # np.flip(mne_maps[:,:,sample],axis=0)
    """
    # Vectorized masking (faster)
    mne_maps[mask[0],mask[1],:] = circle[:,None]
    # For loop by row
    for row_ind in range(num_rows):
        img[row_ind*map_size:(row_ind+1)*map_size,:] = np.reshape(
            mne_maps[:,:,row_ind*num_cols:(row_ind+1)*num_cols],(map_size,map_size*num_cols),
            order='F')
    return img

def mnemap2img(signals,ch_coords,sample_axis,image_interp='bicubic',map_size=50,num_cols=6):
    """
    Slow version of mnemap2img_fast that uses mneviz_topomap() from
    custom_topomap. This function is closer to the original, 
    mne.viz.plot_topomap()
    """
    img = np.zeros([math.ceil(map_size*np.ceil(np.shape(signals)[sample_axis]/num_cols)),map_size*num_cols])
    signals = minmax(signals,axis=None)*255
    # MNE viz topomaps are 64x64
    mask, circle, circle_ind = drawCircle(map_size)
    
    for obs in range(np.shape(signals)[sample_axis]):
        im,_,_ = mneviz_topomap(
            np.take(signals,obs,axis=sample_axis),
            ch_coords,cmap='Greys',
            sphere=np.array([0,0,0,12.2]),vmin=0,vmax=255,res=map_size,show=False)
        data = np.flip(im.get_array(),axis=0)
        data[mask] = circle*255
        img[obs//num_cols*map_size:(obs//num_cols+1)*map_size,(obs%num_cols)*map_size:(obs%num_cols+1)*map_size] = data
    return img

def eeg2img(signals,ch_coords,sample_axis,interp='cubic',map_size=49,num_cols=6,pad=False,pad_coords=[]):
    """
    Convert EEG signals with known 2D channel coordinates to image using
    scipy's griddata()
    Args:
        signals ((sample,channel,observation) array): EEG signals
        ch_coords ((channel,2) array): EEG channel coordinates (2D)
        sample_axis (int): axes corresponding to sample
        interp (str): interpolation method ('cubic','linear','nearest')
        l (int): square map size in pixels
        num_cols (int): number of maps per row of 'img'
        pad (bool): whether to pad 'nans' outside of interpolated regions when
                    using 'cubic' and 'linear'
        pad_coords ((X,2) array): coordinates of positions to pad
    Returns:
        img (array): image representations of EEG signals, where each sample
                     becomes a topographic map of the skull
    """
    # Initialize image
    img = np.zeros([math.ceil(map_size*np.ceil(np.shape(signals)[sample_axis]/num_cols)),map_size*num_cols])
    # Min-max normalize each channel of signals
    signals = minmax(signals,axis=sample_axis)*255
    # Find regions to mask
    mask, circle, circle_ind = drawCircle(map_size)
    # Fill 'img' with topographic map for each sample
    for obs in range(np.shape(signals)[sample_axis]):
        data = eeg2map(ch_coords,np.take(signals,obs,axis=sample_axis),interp,pad,pad_coords)
        # Masks area outside skull
        data[mask] = circle*255
        img[obs//num_cols*map_size:(obs//num_cols+1)*map_size,(obs%num_cols)*map_size:(obs%num_cols+1)*map_size] = data
    return img

def eeg2map(ch_coords,signal_sample,interp='cubic',pad=False,pad_coords=[],bounds=[-11,11,-12,12],steps=[0.45,0.49]):
    # Results in 49x49 grid
    # Y and X flipped to get correct output
    grid_y, grid_x = np.mgrid[bounds[2]:bounds[3]:steps[1],
                              bounds[0]:bounds[1]:steps[0]]
    """
    if pad:
        # Fill in missing electrodes as zeros/mean value - results in artifacts
        ch_coords = np.concatenate([ch_coords,pad_coords],axis=0)
        signal_sample = np.concatenate([signal_sample,
                                        np.zeros(len(pad_coords))+np.mean(signal_sample)])
    """
    grid = gd(ch_coords,signal_sample,(grid_x,grid_y),method=interp)#fill_value=0)
    if pad:
        # Fill in nan values with nearest interpolation
        nan_inds = np.where(np.isnan(grid))
        grid2 = gd(ch_coords,signal_sample,(grid_x,grid_y),method='nearest')
        grid[nan_inds] = grid2[nan_inds]
    return np.flip(grid,axis=0)

def minmax(signal,axis=1,scale_mask=None,replace_nans_with_zeros=False):
    if scale_mask is None:
        if axis is None:
            # Overall min-max normalization
            maxes, mins = np.max(signal), np.min(signal)
            return (signal-mins)/(maxes-mins)
        else:
            # Dimension-specific min-max normalization
            maxes = np.expand_dims(np.max(signal,axis=axis),axis=axis)
            mins = np.expand_dims(np.min(signal,axis=axis),axis=axis)
            norm_signal = (signal-mins)/(maxes-mins)
            zero_locs = np.where(maxes-mins==0)[np.invert(axis)]
            if len(zero_locs)>0:
                # If n-th element is constant along axis, min-maxed result is np.nan
                # Replace np.nan results with original constant value OR with zeros
                np.put_along_axis(
                    norm_signal,zero_locs[:,None],
                    np.zeros(np.shape(signal)[axis]) if replace_nans_with_zeros else
                    np.take(signal,zero_locs,np.invert(axis)),np.invert(axis))
            return norm_signal
    else: # Only scale values at indices where mask is 1
    # ! Not yet tested with scale mask
        if np.shape(scale_mask != np.shape(signal)):
            raise ValueError('Mask does not match shape of signal')
        if axis is None:
            maxes, mins = np.max(signal), np.min(signal)
            for i in range(len(signal)):
                if np.any(scale_mask[i]):
                    signal[i][scale_mask[i]] = (signal[i][scale_mask[i]]-mins)/(maxes-mins)
        else:
            maxes, mins = np.zeros([np.shape(signal)[axis],1]), np.zeros([np.shape(signal)[axis],1])
            for i in range(np.shape(signal)[axis]):
                signal_segment = np.take(signal,i,axis)[scale_mask]



def drawCircle(circle_size,centroid=None,radius=None,threshold=None):
    if centroid is None:
        center = (circle_size-1)/2
    else:
        center = centroid
    if radius is None:
        radius = min(center, circle_size-center)
    if threshold is None:
        threshold = 0.48
    x, y = np.ogrid[:circle_size, :circle_size]
    dist_from_centroid = np.sqrt((x-center)**2 + (y-center)**2)
    
    # 'Mask' area corresponding to skull outline and outside the skull.
    # Circle line becomes 1s, outside circle becomes 0s
    # Indices to mask (tuple)
    mask_ind = np.where(dist_from_centroid >= radius-threshold)
    # Circle array
    circle = (abs(dist_from_centroid - radius) <= threshold).astype('int')
    # Circle indices to pad (if padding for interpolation; array)
    circle_ind = np.where(circle)
    circle_ind = np.concatenate([circle_ind[0][:,None],circle_ind[1][:,None]],axis=1)
    circle_mask = circle[mask_ind]
    # Masking performed like so: img[mask_ind] = circle
    # EEG Map padding performed like so: concatenate circle_ind to ch_coords,
    # concatenate zeros to signal_sample
    return mask_ind, circle_mask, circle_ind
