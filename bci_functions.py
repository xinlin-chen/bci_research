#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: xinlin.chen@duke.edu

Last edited 2023/01/24
"""

import pandas as pd
import numpy as np
from re import search
from os import walk
from glob import glob
from mne.io import read_raw_edf
import pickle
import math
from scipy.stats import mode
from scipy.stats import gaussian_kde as kde
from statistics import multimode
from swlda import Clfr
from general_functions import deal, offset_range
from sklearn.metrics import roc_auc_score as get_auc
import string
from language_model import viterbi_alg
from matplotlib import pyplot as plt
from general_functions import Array

def get_sample_window(fs=256,wlen=0.8,decimation_f=20):
    sampling_factor = math.ceil(fs/decimation_f)
    feats_per_channel = math.floor(wlen*fs/sampling_factor)
    sample_window = int(feats_per_channel*sampling_factor)
    return sample_window

def get_p300sp_context_windows(task_onset,obs_window_samples,
                               model_stride_samples,max_context_window_samples):
    """
    Calculate start and end of context windows for a series of stimuli. Each
    context window ends at obs_window_samples*current_task_onset and starts
    an integer multiple of model_stride_samples before that. The context
    window will not be longer than max_context_window_samples, and will not
    stretch to before the first stimulus onset.
    
    Args:
        task_onset (Array['N',int]): EEG signal sample indices at which N stimuli
            were presented (i.e. onset of flash)
        obs_window_samples (int): length of observation window in samples
        model_stride_samples (int): sample stride (effective sampling
            frequency) of model that will process the EEG
        max_context_window_samples (int): maximum length (in samples) of EEG
            context window. This context window ends at obs_window_samples
            after a flash onset
    
    Returns:
        eeg_context (Array['N,3',int]): each of N rows contains
        	(task_onset,context_window_start,context_window_end) for the N-th stimulus
    """
    eeg_context = np.zeros((len(task_onset),3),dtype='int')
    # Max context window size must be multiple of model stride
    max_context_window_samples = max_context_window_samples-(max_context_window_samples%model_stride_samples)
    # Assume EEG data valid from start of first flash
    spelling_start_sample = task_onset[0]
    eeg_context[:,0] = task_onset
    for ind,start_sample in enumerate(task_onset):
        # Actual sample window should be integer multiple of the model stride,
        # so that the last X embeddings of the model output correspond to the
        # relevant EEG observation
        eeg_context[ind,2] = start_sample+obs_window_samples-(obs_window_samples%model_stride_samples)
        # Get beginning of context window. Observation length must be integer
        # multiple of model stride
        eeg_context[ind,1] = max(spelling_start_sample,eeg_context[ind,2]-max_context_window_samples)
    return eeg_context

def extract_eeg_epochs(signal,task_onset,sample_window=195,offset=0,path=''):
    """Extract EEG epochs.
    
    Extract time-locked EEG epochs from user data, given an
    array of signals and information about flash onset.
    """
    
    eeg_signals = np.zeros([len(task_onset),sample_window,np.shape(signal)[1]])
    err_inds = []
    for ind,obs in enumerate(task_onset):
        try:
            eeg_signals[ind,:,:] = signal[obs+offset:obs+offset+sample_window,:]
        except:
            print(f'Epoch too short for path: {path} ind {ind}'))
            continue
    
    return eeg_signals

def segment_signal(signal,fs,wlen=0.8,wstep=0.25):
    """Split up signal into segments.
    
    Split signal up into segments of length <wlen>, taking a step of <wstep>
    each time.
    
    Args:
        signal (Array['num_samples', float]): EEG signal (single-channel)
        fs (int): sampling frequency (Hz)
        wlen (float): window length (s)
        wstep (float): window step (s)
    Returns:
        sig_frags (Array['num_obs,sample_window',float])): segmented EEG signals
    """
    
    signal = signal.squeeze()
    sample_window = math.ceil(wlen*fs)
    sample_step = math.ceil(wstep*fs)
    # Indices of each EEG fragment
    start_inds = np.arange(0,signal.shape[0]-sample_window,sample_step,dtype='uint64')
    end_inds=start_inds+sample_window
    total_num_obs = len(start_inds)
    
    # Signal fragment
    sig_frags = np.zeros([total_num_obs,sample_window])
    for i in range(total_num_obs):
        sig_frags[i,:] = signal[start_inds[i]:end_inds[i]]
    return sig_frags


class Speller:
	"""
	For performing offline spelling simulations with brain signal data. Presently
    implemented for data collected using the checkerboard paradigm. This speller can use
    static or dynamic stopping. The classifier may be static or adaptive (i.e., supports
    being updated with new data).
    
    Attributes:
    	num_sessions (int): max number of sessions that a user went in for
        num_subjects (int): number of subjects in study
        adaptive (bool): adaptive speller (i.e. can be re-trained)?
        dynamic (bool): use dynamic stopping
        ds_threshold_val (bool): for dynamic stopping, at what probability
            threshold do we stop spelling?
        use_row_column (bool): row-column (true) or other stimulus presentation paradigm?
        label_type (str): 'scores' (sum of classifier scores) or
            'likelihoods' (Bayesian algorithm)
        set_ada_threshold (bool): set character score threshold for
            expanding training set with unsupervised data (unlabelled
            'online' data and classifier-predicted labels)?
        ada_threshold_val (float): threshold value for above
        get_performance (bool): collect data to calculate performance
            measures (e.g. sum of true positives for each trial)?
	"""
    def __init__(self,num_sessions=1, num_subjects=1,adaptive=False,
                 dynamic=False,ds_threshold_val=0.9,
                 use_row_column=False,label_type='likelihoods',
                 set_ada_threshold=True,ada_threshold_val=None,
                 get_performance=False):
        """Set up speller.
        """
        self.spelling_acc = np.zeros([num_sessions,num_subjects])-1
        self.num_correctly_spelled_chars = np.zeros([num_sessions,num_subjects],dtype='int')
        self.tr_pdfs = [None,None]
        self.tr_classes = np.arange(2)
        self.is_adaptive = adaptive
        self.is_dynamic = dynamic
        self.use_row_column = use_row_column
        self.label_type = 'likelihoods'
        self.get_performance = get_performance
        if adaptive:
            self.Clfr = [None]*num_subjects
            # ETS: expanded training set
            self.ets_size_chars, self.ets_num_correctly_spelled_chars = \
                deal(np.zeros,2,shape=[num_sessions,num_subjects],dtype='int')
            self.ets_accuracy = np.zeros([num_sessions,num_subjects])
            self.set_ada_threshold = set_ada_threshold
            
            if set_ada_threshold:
                self.poor_chars = np.zeros([num_sessions,num_subjects],dtype='int')
            if ada_threshold_val is None:
                if label_type[0] == 's': #scores
                    self.ada_threshold = 0.15
                else: #likelihoods
                    self.ada_threshold = ds_threshold_val
            else:
                self.ada_threshold = ada_threshold_val
        else:
            self.Clfr = [Clfr() for _ in range(num_subjects)]
        if dynamic:
            self.ds_threshold_val = ds_threshold_val
    
    def fit(self,sub_ind,data,labels):
        # Train classifier, estimate pdfs of training scores
        self.Clfr[sub_ind].fit(data,labels)
        self.update_pdfs(sub_ind)
    
    def unsupervised_update(self,sub_ind,sess_ind,data,labels,char_ranges,char_inds_overall):
        """Update adaptive classifier with trial(s) if appropriate.
        
        Discard trial if imposing threshold and that threshold is not met by
        the trial.
        """
        include_trial = np.array([(not self.set_ada_threshold) or val>self.ada_threshold
                                  for val in self.threshold_vals[char_inds_overall].flatten()])
        self.included_trials_session[char_inds_overall] = include_trial
        if any(include_trial):
            include_inds = np.zeros(len(labels),dtype='bool')
            # Add unlabelled trials to expanded training set of classifier
            for i in range(len(include_trial)):
                if include_trial[i]:
                    include_inds[char_ranges[i]] = True
                    self.ets_size_chars[sess_ind,sub_ind] += 1
            self.Clfr[sub_ind].update(data[include_inds,:],labels[include_inds])
            self.update_pdfs(sub_ind)
        
    
    def update_pdfs(self,sub_ind):
        # Get non-target and target pdfs of training data
        self.tr_pdfs = [kde(self.Clfr[sub_ind].tr_scores[
            np.invert(self.Clfr[sub_ind].tr_labels)]),
                        kde(self.Clfr[sub_ind].tr_scores[self.Clfr[sub_ind].tr_labels])]
    
    def eval_pdfs(self,score):
        # Evaluate non-target and target pdfs at the point of the current
        # classifier score
        
        return [self.tr_pdfs[i].evaluate(score) for i in self.tr_classes]
    
    def test(self,sub_ind,data,labels=[]):
        # Every time test() is called, predictions are reset
        # Get testing scores
        te_scores = self.Clfr[sub_ind].test(data,labels)
        # Reset predictions for current testing set (e.g. session, word or
        # trial)
        self.reset_predictions(len(te_scores))
        return te_scores
    
    def copy(self,sub_ind,speller_obj):
        # Shallow copies classifier, training parameters from other Speller object
        self.Clfr[sub_ind] = speller_obj.Clfr[sub_ind].copy()
        self.tr_pdfs = speller_obj.tr_pdfs
    
    def initialize_session(self,num_chars,M,label_type):
        if self.label_type[0] == 's': # scores
            # CCRF: cumulative character response function (e.g. character probability)
            self.ccrf = np.zeros([M,num_chars])
        elif self.label_type[0] == 'l': # likelihoods
            self.ccrf = np.ones([M,num_chars])*1/M
        self.correctly_spelled = np.zeros(num_chars,dtype='bool')
        if self.is_adaptive:
            # 'Included' means used as an unlabelled observation to expand
            # training set
            self.included_trials_session = np.zeros(num_chars,dtype='bool')
            self.threshold_vals = np.zeros(num_chars)
        if self.get_performance:
            self.num_true_pos = np.zeros(num_chars,dtype='int')
            # Number of target predictions will always be the same as number of targets
            # self.num_targ_preds = np.zeros(num_chars,dtype='int')
            self.num_false_pos = np.zeros(num_chars,dtype='int')
            self.num_true_neg = np.zeros(num_chars,dtype='int')
            self.num_targs = np.zeros(num_chars,dtype='int')
            self.num_ntargs = np.zeros(num_chars,dtype='int')
    
    def initialize_trial(self):
        self.not_selected = True
    
    def simulate_epoch(self,score,flashed_chars_bool,char_ind):
        # If using dynamic stopping and a character's CCRF has exceeded the
        # spelling threshold, don't update CCRF anymore
        if self.not_selected or not self.is_dynamic:
            l = self.eval_pdfs(score)
            self.ccrf[:,char_ind] = bayesian_ccrf(
                self.ccrf[:,char_ind],l[0],l[1],flashed_chars_bool)
        if self.is_dynamic and np.max(self.ccrf[:,char_ind])>self.ds_threshold_val:
            self.ds_ind = char_ind
            self.not_selected = False
    
    def spell(self,trial_ind_overall,flash_groups,target_char_code,prediction_epoch_inds=[],truth_labels=[]):
        
        # Find character that maximizes CCRF
        sorted_ccrf_ind = np.argsort(self.ccrf[:,trial_ind_overall])
        sorted_ccrf = self.ccrf[sorted_ccrf_ind,trial_ind_overall]
        # Is this character the true target?
        self.correctly_spelled[trial_ind_overall] = sorted_ccrf_ind[-1] == target_char_code
        
        # For adaptive spellers, collect predictions and threshold values
        if self.is_adaptive:
            # Get value to compare against adaptive classifier threshold
            if self.label_type[0] == 's':
                self.threshold_vals[trial_ind_overall] = 1-sorted_ccrf[-2]/sorted_ccrf[-1]
            elif self.label_type[0] == 'l':
                self.threshold_vals[trial_ind_overall] = sorted_ccrf[-1]
                
        # Note: flash group character codes are one-indexed, so add one
        # to location of character that maximises CCRF
        if prediction_epoch_inds:
            self.predictions[prediction_epoch_inds] = predict_labels(flash_groups,sorted_ccrf_ind[-1]+1)
            if self.get_performance and np.size(truth_labels):
                if (not (hasattr(self,'set_ada_threshold') and self.set_ada_threshold)) or sorted_ccrf[-1]>=self.ada_threshold:
                    self.evaluate_performance(trial_ind_overall,prediction_epoch_inds,truth_labels)
    
    
    def evaluate_performance(self,trial_ind_overall,prediction_epoch_inds,truth_labels):
        """
        Args:
            trial_ind_overall (int): index of trial being analyzed (relative
                to the start of the user's spelling session)
            prediction_epoch_inds (Array['M',int]): indices of self.predictions
                corresponding to trial
            truth_labels (Array['N',bool]): truth labels for epochs in trial
        """
        # Number of actual target flash groups that were correctly predicted
        self.num_true_pos[trial_ind_overall] = np.sum(
            self.predictions[prediction_epoch_inds][truth_labels])
        self.num_false_pos[trial_ind_overall] = np.sum(
            self.predictions[prediction_epoch_inds][np.invert(truth_labels)])
        # True positives + false positives. Should always be the same as num_targs
        # self.num_targ_preds[trial_ind_overall] = np.sum(self.predictions[prediction_epoch_inds])
        # Number of actual targets (i.e. true positives + false negatives)
        self.num_targs[trial_ind_overall] = np.sum(truth_labels)
        # Number of actual non-target flash groups that were correctly predicted
        self.num_true_neg[trial_ind_overall] = np.sum(
            np.invert(self.predictions[prediction_epoch_inds][np.invert(truth_labels)]))
        self.num_ntargs[trial_ind_overall] = len(truth_labels)-np.sum(truth_labels)
    
    def apply_lang_model(self,lang_models,grid_objects,trial_inds_overall,
                         flash_groups_wd,trial_ranges_wd,prediction_epoch_inds,
                         target_chars,truth_labels_wd):
        """
        Apply language model once a full word has been spelled. Get new
        predicted targets for each trial corresponding to the word, new
        predicted labels for all epochs corresponding to the word. Finally,
        update spelling accuracy using new predicted targets.
        
        Args:
            lang_models (list[LM_letter]): list of language model objects to use in
                Viterbi algorithm
            grid_objects (str): string of potential grid selections (in order)
            trial_inds_overall (list[int]): indices of Speller.ccrf to apply language
            	model to
            flash_groups_wd (Array['W,H',int]): flash groups corresponding to current word
            trial_ranges_wd (list[range]): each element contains the range of epoch
                indices in flash_groups_wd corresponding to a trial (used to
                spell a letter from the current word)
            prediction_epoch_inds (Array[int]): indices of predictions to updates
            target_chars (list[str]): actual target characters for current word
            truth_labels_wd (array): ground truth labels for current word
        """
        # Get predicted targets for each trial after applying language model
        self.predicted_targets_lm = viterbi_alg(
            lang_models,grid_objects,self.ccrf[
                :,trial_inds_overall])+1
        # Get predicted labels for all epochs corresponding to current word
        self.predictions[prediction_epoch_inds] = predict_word(
            flash_groups_wd,self.predicted_targets_lm,trial_ranges_wd)
        # Re-calculate spelling accuracy
        self.correctly_spelled[trial_inds_overall] = \
            target_chars == self.predicted_targets_lm
        
        if self.get_performance:
            for trial_ind_wd,epoch_inds_wd in enumerate(trial_ranges_wd):
                self.evaluate_performance(
                    trial_inds_overall[trial_ind_wd],
                    prediction_epoch_inds[epoch_inds_wd],
                    truth_labels_wd[epoch_inds_wd])
    
    def eval_session(self,sub_ind,sess_ind,test_labels,chars_spelled_bool):
        # A trial may have been skipped if an error occurred during recording
        self.num_correctly_spelled_chars[sess_ind,sub_ind] = np.sum(self.correctly_spelled[
                chars_spelled_bool])
        self.spelling_acc[sess_ind,sub_ind] = self.num_correctly_spelled_chars[sess_ind,sub_ind]\
            /np.sum(chars_spelled_bool)
        if self.is_adaptive:
            if self.set_ada_threshold:
                # Discarded characters (out of all characters that were not skipped)
                self.poor_chars[sess_ind,sub_ind] = np.sum(chars_spelled_bool)-\
                np.sum(self.included_trials_session)
            # Characters must have been spelled and added to expanded
            # training set.
            if self.ets_size_chars[sess_ind,sub_ind] > 0:
                self.ets_num_correctly_spelled_chars[sess_ind,sub_ind] = \
                    np.sum(self.correctly_spelled[self.included_trials_session])
                self.ets_accuracy[sess_ind,sub_ind] = self.ets_num_correctly_spelled_chars[sess_ind,sub_ind]/\
                    self.ets_size_chars[sess_ind,sub_ind]
            else:
                self.ets_accuracy[sess_ind,sub_ind] = np.nan
    
    def reset_predictions(self,num_epochs):
        # Reset at beginning of session and after each update of adaptive
        # classifier parameters
        self.predictions = np.zeros(num_epochs,dtype='bool')


class SpellingTracker:
    """
    Keeps track of spelling as a session is simulated. Tracks index of current
    epoch, trial and word. Also identifies if trials are invalid and keeps
    track of associated invalid epochs in a session.
    """
    def __init__(self,df,num_epochs=None,use_row_column=False):
        # Get size of spelling board
        self.M = int(df.iloc[0].nrows*df.iloc[0].ncols)
        # Index of word, index of character within all of testing set
        self.word_ind, self.char_ind_overall = 0, 0
        # Get info about testing set
        self.num_chars_per_wd,self.test_set_size_chars,_,_,_,_ = \
                    p300_file_summarize(df[['text','task_onset','num_seq']])
        self.skipped_chars_overall = np.zeros(self.test_set_size_chars,dtype='bool')
        self.cumsum_chars = np.cumsum(self.num_chars_per_wd)
        self.task_onsets = [i.task_onset for i in df.itertuples()]
        self.use_row_column = use_row_column
        self.session_ended = False
        # Keep track of all target characters over session
        if use_row_column:
            # First column: row index. Second column: column index.
            self.test_target = np.zeros([self.test_set_size_chars,2],dtype='int')
        else:
            # Character index
            self.test_target = np.zeros(self.test_set_size_chars,dtype='int')
        self.word_start_flash_ind = 0
        self.num_epochs = num_epochs
        self.start_word()
    
    def start_next_trial(self):
        # Increment character indices
        # Character index (starting from beginning of current word)
        self.char_ind_wd += 1
        # Character index (starting from beginning of session)
        self.char_ind_overall += 1
        # Check if we have completed spelling the previous word
        if self.char_ind_overall==(self.cumsum_chars[self.word_ind]):
            # Signals that it is time to update an adaptive-by-word classifier
            self.word_ended = True
            
            # Gather information about previous word
            self.epoch_inds_prev_wd = np.array(self.epoch_inds_curr_wd)
            self.valid_epochs_prev_wd = self.valid_epochs_wd
            ###
            # Only necessary for speller using language model
            self.chars_spelled_prev_wd_bool = np.invert(self.skip_char_in_wd)
            # Characters spelled (index 0 corresponds to beginning of current session)
            self.chars_spelled_prev_wd_inds_overall = np.where(
                self.chars_spelled_prev_wd_bool)[0]+self.word_start_char_ind
            ###
            self.valid_trial_ranges_prev_wd = [
                self.trial_ranges_wd[i] for i in range(len(self.trial_ranges_wd))
                if self.chars_spelled_prev_wd_bool[i]]
            self.prev_test_target = self.test_target[self.word_start_char_ind:self.char_ind_overall]
            
            self.word_ind += 1
            # Check if there are still words to spell
            if self.word_ind<len(self.num_chars_per_wd):
                # Prepare to start spelling the next word
                self.word_start_flash_ind += np.sum(self.true_num_char_flashes)
                self.start_word()
            else:
                # Otherwise, start a new session
                self.session_ended = True
        else:
            # Continue spelling current word
            self.word_ended = False
            self.epoch_inds_curr_trial = offset_range(self.trial_ranges_wd[self.char_ind_wd],
                                                self.word_start_flash_ind)
            # Get number of epochs in next trial
            if not self.num_epochs:
                self.num_epochs_trial = self.true_num_char_flashes[self.char_ind_wd]
            else:
                self.num_epochs_trial = self.num_epochs
            
    
    def invalidate_trial(self):
        # Invalidate further epochs from current word if desired
        self.valid_epochs_wd[self.trial_ranges_wd[self.char_ind_wd]] = False
        self.skip_char_in_wd[self.char_ind_wd] = True
        self.skipped_chars_overall[self.char_ind_overall] = True
        
    def start_word(self):
        # Index of character in current word
        self.char_ind_wd = 0
        # Index of first character in word (overall, i.e. out of all words in
        # testing set)
        self.word_start_char_ind = self.char_ind_overall
        # Sometimes, the actual number of stimuli presented in an experiment
        # may differ because of human (researcher) error
        _,self.true_num_char_flashes = p300_file_breakdown(self.task_onsets[self.word_ind])
        # Get number of epochs in next trial
        if not self.num_epochs:
            self.num_epochs_trial = self.true_num_char_flashes[self.char_ind_wd]
        else:
            self.num_epochs_trial = self.num_epochs
        
        # Trial indices corresponding to current word
        self.epoch_inds_curr_wd = range(self.word_start_flash_ind,
                                    self.word_start_flash_ind+
                                    np.sum(self.true_num_char_flashes))
        # Skip character (aka trial) if epochs are missing
        self.skip_char_in_wd = self.true_num_char_flashes<mode(self.true_num_char_flashes)[0][0]
        self.skipped_chars_overall[self.char_ind_overall:self.char_ind_overall+
                                   len(self.true_num_char_flashes)] = self.skip_char_in_wd
        # Get index at which data are collected for each character in the
        # current word, relative to the first epoch of the current word
        trial_edges_wd = np.hstack([0,np.cumsum(self.true_num_char_flashes)])
        self.trial_ranges_wd = [range(trial_edges_wd[i],
                                        trial_edges_wd[i+1])
                                  for i in range(len(self.true_num_char_flashes))]
        # Indices corresponding to current trial
        self.epoch_inds_curr_trial = offset_range(
            self.trial_ranges_wd[self.char_ind_wd],self.word_start_flash_ind)
        # Find the epochs in the current word that are valid
        self.valid_epochs_wd = np.ones(len(self.task_onsets[self.word_ind]),dtype='bool')
        for ii in range(len(self.true_num_char_flashes)):
            if self.skip_char_in_wd[ii]:
                self.valid_epochs_wd[self.trial_ranges_wd[ii]] = False
            if not self.num_epochs:
                self.valid_epochs_wd[self.trial_ranges_wd[ii][self.num_epochs:]] = False
        
    
    def get_target_char(self,flash_groups,test_labels):
        self.test_target[self.char_ind_overall] = get_targchar_code(
            flash_groups[:,1:],test_labels,self.M,self.use_row_column)
        return self.test_target[self.char_ind_overall]


def bayesian_ccrf(priors,likelihood_nt,likelihood_t,flashed_chars_bool,eps=1e-10):
    """
    Bayesian cumulative character response function
    
    Update probability of each character being the target character based on
    the classifier score for an EEG epoch. Get likelihood*priors for each
    character based on whether they were in the flash group. Modify in place.
    
    Spelling board has M characters.
    
    Args:
        priors (Array['M',float]): prior probability of each character being the
                            target character
        likelihood_nt (float): non-target probability density function value
        likelihood_t (float): target probability density function value
        flashed_chars_bool (Array['M',bool]): whether a character was
                            included in the flashed group
    """
    priors[flashed_chars_bool] *= likelihood_t+eps
    priors[np.invert(flashed_chars_bool)] *= likelihood_nt+eps
    
    if sum(priors)==0:
        print('\nDiv by zero in priors')
    
    return priors/sum(priors)

def p300_file_summarize(df):
    
    num_chars_per_word = df['text'].apply(lambda i: len(i)).to_numpy()
    num_chars_overall = np.sum(num_chars_per_word)
    num_flashes_per_word = np.sum(df['task_onset'].apply(lambda i: len(i)).to_numpy())
    num_flashes_per_char = np.divide(num_flashes_per_word,num_chars_per_word).astype('int')
    num_flashes_per_seq = np.divide(num_flashes_per_char,df['num_seq']).astype('int')
    num_flashes_overall = np.sum(np.multiply(num_flashes_per_char,num_chars_overall))
    
    return num_chars_per_word,num_chars_overall,num_flashes_per_word,\
        num_flashes_per_char,num_flashes_per_seq,num_flashes_overall

def p300_file_breakdown(task_onset):
    """
    Extracted beginning of stimulus presentation for each character (aka
    trial) in a set of trials for a P300 speller experiment. Numbers of
    stimuli (aka flashes) presented per character also extracted.
    """
    onset_diff = np.diff(task_onset,axis=0)
    char_change_points = np.append(np.where(onset_diff>np.mean(onset_diff))[0]+1,len(task_onset))
    num_char_flashes = np.insert(np.diff(char_change_points,axis=0),0,char_change_points[0])
    return char_change_points[:-1],num_char_flashes
    
def get_targchar_code(flash_groups,labels,M,use_row_column=False):
    """
    Infer the code of the current target character, given a series of flash
    groups and their associated labels.
    Note: we might find out as soon as 2 flash groups in that one character
    must be the target character. However, checking all flash groups means
    we can flag if something is wrong with the selected spelling data (e.g.
    accidentally spills across more than one trial)
    
    BCI2000 character codes are one-indexed
    """
    target_fg_vector = flash_groups[np.where(labels)[0],:].flatten()
    # target_mode = mode(target_fg,axis=None,nan_policy='omit')
    target_fg_vector = target_fg_vector[~np.isnan(target_fg_vector)]
    # Find most frequently occurring character
    targ_code = multimode(target_fg_vector)
    # Error if 0 or >1 characters appear across all target flash groups
    if sum(target_fg_vector==targ_code[0]) < np.sum(labels) or len(targ_code)>1:
        raise BaseException('Error finding target character')
    return targ_code[0]

def fg2bool(flash_group,M):
    """
    Convert P300 speller-recorded flash group on a size-M spelling board to a
    vector of booleans
    """
    flash_bool = np.zeros(M,dtype='bool')
    flash_bool[flash_group] = True
    return flash_bool

def predict_labels(flash_groups,char_code):
    # Predict labels from flash group
    predictions = np.zeros(np.shape(flash_groups)[0],dtype='bool')
    predictions[np.argwhere(flash_groups==char_code)[:,0]] = True
    return predictions

def predict_word(flash_groups,char_codes,char_ranges):
    """
    Args:
        flash_groups (Array['M,N',int]): flash groups corresponding to M epochs;
            each flash group has up to N-1 characters
        char_codes (Array['P',int]): character codes corresponding to P trials
        char_start (Array['P',int]): start index for each trial (letter) in word
    """
    predictions = np.zeros(np.shape(flash_groups)[0],dtype='bool')
    for char_ind,char_code in enumerate(char_codes):
        predictions[char_ranges[char_ind]] = \
            predict_labels(flash_groups[char_ranges[char_ind]],char_code)
    return predictions

def get_spelled_chars(spelling_grid_objects,ccrf):
    # Find the characters that are spelled, given a spelling board and
    # associated character cumulative response functions
    char_inds = np.argmax(ccrf,axis=0)
    chars = ''.join([spelling_grid_objects[i] for i in char_inds])
    return chars

def update_pdfs(scores,labels):
    return [kde(scores[np.invert(labels)]),kde(scores[labels])]

def display_spelled_words(spelling_grid_objects,char_locs,word_len,spelled_chars_bool):
    ranges = [range(word_len*i,word_len*(i+1)) for i in range(int(len(char_locs)/word_len))]
    spelled_chars_bool = np.reshape(spelled_chars_bool,[int(len(char_locs)/word_len),-1])
    chars = []
    for i in range(len(ranges)):
        curr_chars = ''.join([spelling_grid_objects[j] for j in char_locs[ranges[i]]])
        chars += [''.join([curr_chars[s] for s in range(word_len) if spelled_chars_bool[i][s]])]
    return ', '.join(chars)

def plot_ccrf(ccrf,target_inds,ylims=None,label_type='Probability',highlight_chars=True,spelling_grid_objects=None):
    # Plot heatmap of cumulative character response functions for X trials on
    # a size Y spelling board
    if not spelling_grid_objects and np.shape(ccrf)[0] == 36:
        spelling_grid_objects = string.ascii_uppercase+string.digits[1:]+' '
    spelled_chars = [spelling_grid_objects[i] for i in np.argmax(ccrf,axis=0)]
    if highlight_chars:
        y_inds = np.sort(target_inds)
        y_labels = [spelling_grid_objects[i] for i in y_inds]
        y_inds = y_inds + 0.5
    else:
        y_inds = np.arange(0.5,np.shape(ccrf)[0]+0.5)
        y_labels = spelling_grid_objects
    
    plt.pcolormesh(ccrf);c=plt.colorbar();c.set_label(label_type)
    ax = plt.gca();
    ax.set_xlabel('Character',fontsize=15);ax.set_xticks(np.arange(0.5,np.shape(ccrf)[1]+0.5))
    ax.set_xticklabels(spelled_chars)
    ax.set_ylabel('Spelling Board',fontsize=15);ax.set_yticks(y_inds)
    ax.set_yticklabels(y_labels)
    ax.tick_params(axis='both',which='major',labelsize=14)
    plt.gca().invert_yaxis()
    if not ylims:
        plt.ylim(ylims)
    if label_type == 'Probability':
        plt.clim([0,0.1])
    
    for c in range(len(target_inds)):
        plt.plot([c,c+1,c+1,c,c],[
            target_inds[c],target_inds[c],target_inds[c]+1,
            target_inds[c]+1,target_inds[c]],color='k')
    
    plt.gcf().set_size_inches(6,5)