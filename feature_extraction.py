#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 02:16:24 2021

@author: xinlin.chen@duke.edu

Last edited 2023/01/24
"""

import math
import numpy as np
import pycwt
from general_functions import Array    

def extract_ba_feats(signal,decimation_f=20,fs=256):
    """
    Extract block-averaged features from an array of P300 speller EEG epochs
    
    Args:
    	signal (Array['num_obs, num_samples, num_channels',float]): EEG data
    	decimation_f (int): decimation frequency (Hz)
    	fs (int): sampling frequency (Hz)
    Returns:
    	features (Array['num_obs, num_features',float]): extracted features
    """
    # Each observation already contains the correct time window
    
    total_num_obs = np.shape(signal)[0]
    sampling_factor = math.ceil(fs/decimation_f)
    feats_per_channel = np.floor(np.shape(signal)[1]/sampling_factor)
    sample_window = int(feats_per_channel*sampling_factor)
    try:
        nch = np.shape(signal)[2]
    except: # Only one channel
        nch = 1
    num_feats = int(nch*feats_per_channel)
    features = np.zeros((total_num_obs,num_feats))
    if nch == 1:
        for obs in range(total_num_obs):
            # Block average, concatenate data across channels
            features[obs,:] = np.mean(np.reshape(np.squeeze(signal[obs,0:sample_window]),(sampling_factor,-1),order='F'),axis=0)
    else:
        for obs in range(total_num_obs):
            # Block average, concatenate data across channels
            features[obs,:] = np.mean(np.reshape(np.squeeze(signal[obs,0:sample_window,:]),(sampling_factor,-1),order='F'),axis=0)
        
    return features

def extract_wv_feats(signal,fs,nscales=20,freqlims=[0.5,30]):
    """ Compute CWT using Morlet wavelet
    
    Extract time-frequency features for windows of EEG data within a signal.
    
    Args:
        signal (Array['num_obs,num_samples',float]): EEG signal segments
        fs (float): sampling frequency (Hz)
        nscales (int): number of wavelet scales (affects resolution in
                                                 frequency domain)
        freqlims (Array[2,int]): frequency limits in Hz for wavelet transform
        
    Returns:
        features (Array['num_obs,num_channels,nscales,sample_window',float]):
            feature matrix
        
    """
    signal = signal.squeeze()
    if signal.ndim>1 and signal.shape[1]>1:
        total_num_obs=signal.shape[0]
        sample_window = signal.shape[1]
    else: # Only one observation
        total_num_obs = 1
        sample_window = signal.shape[0]
        signal = np.expand_dims(signal,0)
    
    freqs = np.linspace(freqlims[0],freqlims[1],nscales)
    
    
    features = np.zeros((total_num_obs,freqs.size,sample_window),dtype='complex128')
    
    # Morlet wavelet is continuous
    features[0,:,:],scales,_,_,_,_ = pycwt.cwt(signal[0,:],1/fs,wavelet='morlet',freqs=freqs)
    # If more than one observation
    for obs in range(1,total_num_obs):
        features[obs,:,:],_,_,_,_,_ = pycwt.cwt(signal[obs,:],1/fs,wavelet='morlet',freqs=freqs)
    
    # Get the squared magnitude
    features = np.reshape(np.abs(features)**2/scales[:,None],[total_num_obs,-1])
    return features