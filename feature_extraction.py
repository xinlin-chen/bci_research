#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 02:16:24 2021

@author: xinlin.chen@duke.edu

Last edited 23/01/23
"""

import math
import numpy as np
import pycwt
    

def extract_ba_feats(signal,decimation_f=20,fs=256):
    """
    Extract block-averaged features from an array of P300 speller EEG epochs
    
    Args:
    	signal ((num_obs, num_samples, num_channels) array): EEG data
    	decimation_f (int): decimation frequency (Hz)
    	fs (int): sampling frequency (Hz)
    Returns:
    	features ((num_obs, num_features) array): extracted features
    """
    # Each observation already contains the correct time window
    
    total_num_obs = np.shape(signal)[0]#sum([np.shape(i)[0] for i in df['task_onset']])
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
        signal ((num_obs,num_samples) array): EEG signal segments
        fs (float): sampling frequency (Hz)
        nscales (int): number of wavelet scales (affects resolution in
                                                 frequency domain)
        freqlims ((2,) array): frequency limits in Hz for wavelet transform
        
    Returns:
        features (num_obs x num_channels x nscales x sample_window array):
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