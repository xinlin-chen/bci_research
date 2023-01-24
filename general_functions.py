#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 16:16:30 2021

@author: xinlin.chen@duke.edu

Last edited 2023/01/24
"""

import csv
from pickle import load, dump
from matplotlib import pyplot as plt
import numpy as np
from time import time, sleep
from string import ascii_uppercase
from typing import TypeVar, Generic

Shape = TypeVar("Shape")
DType = TypeVar("DType")

class Array(np.ndarray, Generic[Shape, DType]):
    pass

def read_csv(file,delimiter=','):
    with open(file,newline='') as f:
        reader = csv.reader(f,delimiter=delimiter,quoting=csv.QUOTE_NONE)
        data = list(reader)
    return data

def write_txt(file,data,newline):
    with open(file,'w') as f:
        for row in data:
            _ = f.write(row)
            _ = f.write(newline)

def deal(func,number,**kwargs):
    output = []
    for _ in range(number):
        output.append(func(**kwargs))
    return output

def count_torch_model_params(model):
    count = 0
    for param in model.parameters():
        pcount = 1
        for s in param.size():
            pcount *= s
        count += pcount
    return count

def offset_range(range_obj,offset):
    return range(range_obj.start+offset,range_obj.stop+offset)

def list2str(list_var,format_opt=None,delimiter=','):
    if format_opt:
        return '['+delimiter.join([f'%{format_opt}'%el for el in list_var])+']'
    else:
        return '['+delimiter.join([str(el) for el in list_var])+']'

def get_available_gpu(mem_req=9000, interval=60,verbose=False):
    """Wait for GPU fitting memory constraints.
    
    Query GPUs every <interval> seconds until there is a GPU with at least
    <mem_req> MB free.
    """
    if verbose:
        print('Querying GPU memory...')
    go, i = False, 0
    st = time()
    while not go:
        if i > 0:
           sleep(st+i*interval-time())
        nvsmi = nvidia_smi.getInstance()
        mem = nvsmi.DeviceQuery('memory.free')
        gpu_mems = [mem['gpu'][dev]['fb_memory_usage']['free']>mem_req
                        for dev in range(len(mem['gpu']))]
        go = any(gpu_mems)
        i += 1
    device_to_use = gpu_mems.index(True)
    if verbose:
        print('GPU with sufficient memory (%d MB) found after %d minutes'%(
            mem['gpu'][device_to_use]['fb_memory_usage']['free'],i))
    return device_to_use

def retrieve_or_save_abbrev(path,full_string,save=True,delimiter=';'):
    """Abbreviate filenames into code.
    
    Running into issue of filenames being too long. Pick some variable to
    abbreviate. It will be replaced by a code ('A', 'B', ...'Z', 'AA', ...)
    
    Codes saved as <code>;<full_string> in <path>, e.g. EXAMPLE.txt:
        A;[17,23]
        B:[0,8,15,97]
    
    Args:
        filename (str): filename to save or retrieve abbreviated code
        full_string (str): string to store in file
        save (bool): whether or not to save a new abbreviation. If not, just
                tells user abbreviation was not found
    Returns:
        code (str): Code of abbreviated filename

    """
    try:
        # If previous abbreviations exist, load them in
        with open(path,'r') as f:
            codes, abbrevs = zip(*[(row.split(delimiter)[0],
                                    row.split(delimiter)[1].strip())
                                   for row in f.readlines()])
    except:
        codes, abbrevs = [], []
    
    try:
        # Has full_string already been assigned a code?
        index = abbrevs.index(full_string)
        abbrev_code = codes[index]
        return abbrev_code
    except:
        if save:
            # If full_string has not been assigned a code, increment last saved
            # code in file and assign that to full_string
            if not codes:
                last_code = ''
            elif codes[-1] == '' and len(codes)>1:
                last_code_ind = len(codes)-2
                while codes[last_code_ind] == '' and last_code_ind > 0:
                    last_code_ind -=1
                last_code = codes[last_code_ind]
            else:
                last_code = codes[-1]
            
            abbrev_code = increment_alphabet_code(last_code)
            with open(path,'a') as f:
                f.write(f'{abbrev_code}{delimiter}{full_string}\n')
            return abbrev_code
        else:
            return None

def increment_alphabet_code(curr_code):
    if len(curr_code) == 0:
        new_code = 'A'
    else:
        final_index = ascii_uppercase.index(curr_code[-1])
        if final_index == (len(ascii_uppercase)-1):
            new_code = increment_alphabet_code(curr_code[:-1])
            new_code += ascii_uppercase[0]
        else:
            new_code = curr_code[:-1] + ascii_uppercase[final_index+1]
    return new_code

def filter_abbrevs(path,filter_fun,delimiter=';'):
    """Filter strings by filter_fun and return abbreviated codes of those that pass.
    
    Check un-abbreviated strings in file against filter_fun. For strings
    that match, return the string as well as the code used to abbreviate the 
    string
    Args:
        path (str): path to which abbreviated codes were saved
        filter_fun (Callable[str,bool]): function to use to filter abbreviations
        	(e.g. abbreviation must be list of X numbers)
        delimiter (str): column delimiter
    Returns:
        filtered_abbrevs (list[tuple]): list of (code,abbrev) that are
            passed by filter
    """
    with open(path,'r') as f:
        codes, abbrevs = zip(*[(row.split(delimiter)[0],
                                row.split(delimiter)[1].strip())
                               for row in f.readlines()])
    filtered_abbrevs = [(codes[ind],a) for ind,a in enumerate(abbrevs) if filter_fun(a)]
    return filtered_abbrevs

def unabbrev(path,code,delimiter=';'):
	"""Decode abbreviation.
	"""
    with open(path,'r') as f:
        codes, abbrevs = zip(*[(row.split(delimiter)[0],
                                row.split(delimiter)[1].strip())
                               for row in f.readlines()])
    ind = codes.index(code)
    return abbrevs[ind]

def dict2str(in_dict,sep_kv='',sep_entries='_'):
    out_list = []
    for k,v in in_dict.items():
        if isinstance(v,(list,tuple)):
            v = ','.join(map(str,v))
        elif isinstance(v,(str,float,int,bool)):
            v = str(v)
        else:
            v = type(v).__name__
        out_list += [k+sep_kv+v]
    return sep_entries.join(out_list)