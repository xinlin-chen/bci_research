"""
Stepwise linear discriminant analysis classifier. Based on stepwisefit from
MATLAB.

@author: xinlin.chen@duke.edu

Last updated 2023/01/24
"""

import numpy as np
from numpy import matlib as mb
import scipy.linalg as sla
from scipy import stats #- for SWLDA stats
from sklearn.metrics import roc_auc_score as get_auc
from general_functions import Array

class Clfr:
    """Step-wise linear discriminant analysis classifier.
    
    Attributes:
    	penter (float): probability to enter
    	premove (float): probability to remove
    	max_iter (int): max number of iterations
    	zmuv_normalize (bool): zero-mean unit-variance normalize?
    """
    def __init__(self,penter=0.1,premove=0.15,max_iter=60,zmuv_normalize=False):
        self.penter = penter
        self.premove = premove
        self.max_iter = 60
        self.zmuv_normalize = zmuv_normalize
        self.trained = False
    
    def fit(self,data,labels):
        """Train classifier on labelled data.
        """
        self.weights,self.model_terms = stepwisefit(data,labels,self.pentatlaber,self.premove,self.max_iter,self.zmuv_normalize)
        self.tr_scores = np.matmul(data[:,self.model_terms],self.weights[self.model_terms])
        self.tr_data = data
        if labels.ndim == 2:
            self.tr_labels = labels.flatten()
        else:
            self.tr_labels = labels
        self.trained = True
        self.te_scores = np.array([])
    
    def update(self,new_data,new_labels):
        """Re-trains classifier with additional training data.
        
        Does not update 'te_labels' according to the new parameters.
        """
        self.tr_data = np.vstack((self.tr_data,new_data))
        if new_labels.ndim == 2:
            new_labels = new_labels.flatten()
        self.tr_labels = np.hstack((self.tr_labels,new_labels))
        self.weights,self.model_terms = stepwisefit(self.tr_data,self.tr_labels,self.penter,self.premove,self.max_iter,self.zmuv_normalize)
        self.tr_scores = np.matmul(self.tr_data[:,self.model_terms],self.weights[self.model_terms])
    
    def test(self,data,labels=[]):
        """Apply classifier to testing data. If labels are provided, calculate AUC.
        """
        self.te_scores = np.matmul(data[:,self.model_terms],self.weights[self.model_terms])
        if len(labels)>0:
            self.auc = get_auc(labels,self.te_scores)
        return self.te_scores
    
    def copy(self):
        """Copy classifier parameters over to a new classifier object.
        """
        clfr = Clfr()
        for key in self.__dict__.keys():
            setattr(clfr,key,self.__dict__[key])
        return clfr

def stepwisefit(data,labels,penter=0.5,premove=0.15,max_iter=60,zmuv_normalize=False):
    """ Step-wise linear discriminant analysis (SWLDA). Adapted from stepwisefit in MATLAB.
    
    Args:
    	data (Array['num_obs,num_feats',float]): extracted features
    	labels (Array['num obs',Union[bool,float]): truth labels
    	zmuv_normalize (bool): whether or not to normalize the data (zero mean unit variance)
    Returns:
    	all_weights (Array['num_feats',float]): SWLDA model weights
    	model_terms (Array['num_feats',bool]): features to include in model
    """
    
    model_terms = np.zeros(np.shape(data)[1],dtype=bool)
    keep = np.zeros(np.shape(data)[1],dtype=bool)
    P = np.shape(data)[1]
    labels = np.squeeze(labels)
    if zmuv_normalize:
        means = np.mean(data,axis=0)
        stds = np.std(data,axis=0)
        stds[stds==0] = 1
        data = data/stds[:,None].T
    step = 0
    while True:
        all_weights,SE,pval = stepcalc(data,labels,model_terms)
        nextstep,_ = stepnext(model_terms,pval,all_weights,penter,premove,keep)
        if step>=max_iter:
            break
        step += 1
        
        if nextstep==0:
            break
        model_terms[nextstep] = np.invert(model_terms[nextstep])
    return all_weights,model_terms

def stepcalc(data,labels,model_terms):
    N = np.size(labels)
    P = np.size(model_terms)
    X = np.hstack((np.ones((N,1)),data[:,model_terms]))
    num_terms = sum(model_terms)+1
    tol = max(N,P+1)*np.finfo('double').eps
    ex_data = data[:,np.invert(model_terms)] # excluded data
    ex_ssq = np.sum(ex_data**2,0)
    # Scipy linalg's qr
    Q,R,perm=sla.qr(X,mode='economic',pivoting=True)
    if R.size:
        Rrank = np.sum(abs(np.diag(R))>tol*abs(R[0]))
    else:
        Rrank = 0
    if Rrank < num_terms:
        R = R[0:Rrank,0:Rrank]
        Q = Q[:,0:Rrank]
    bb = np.zeros(num_terms)
    Qb = np.matmul(Q.T,labels)
    Qb[abs(Qb)<(tol*np.max(abs(Qb)))] = 0
    bb[perm],_,_,_ = np.linalg.lstsq(R,Qb,rcond=None)
    
    r=labels-np.matmul(X,bb).T
    dfe = np.size(X,0)-Rrank
    df0 = Rrank-1
    SStotal=np.linalg.norm(labels-np.mean(labels))**2
    SSresid = np.linalg.norm(r)**2
    perfectfity = dfe==0 or SSresid<(tol*SStotal)
    if perfectfity:
        SSresid = 0
        r = 0
    rmse = np.sqrt(safedivide(SSresid,dfe))
    Rinv,_,_,_ = np.linalg.lstsq(R,np.eye(np.shape(R)[0],np.shape(R)[1]),rcond=None)
    se = np.zeros(num_terms)
    #covb = np.zeros([num_terms,num_terms])
    #!!! behavior of simultaneous row and column indexing is different in matlab
    covb = rmse**2 * np.matmul(Rinv,Rinv.T)
    covb[:]=covb[:,perm]
    covb[:]=covb[perm,:]
    # To prevent shape from changing (in case of empty se), use [:]
    se[:] = np.sqrt(np.diag(covb))
    
    xr = ex_data-np.matmul(Q,np.matmul(Q.T,ex_data))
    yr = r
    xx = np.sum(xr**2,axis=0)
    perfectfitx = xx<(tol*ex_ssq)
    if perfectfitx.any():
        xr[:,perfectfitx] = 0
        xx[perfectfitx] = 1
    b2 = safedivide(np.matmul(yr.T,xr),xx)
    r2 = mb.repmat(yr[:,None],1,np.sum(np.invert(model_terms)))-np.multiply(
        xr,mb.repmat(b2,N,1))
    df2 = max(0,dfe-1)
    s2 = safedivide(np.sqrt(safedivide(np.sum(r2**2,axis=0),df2)),np.sqrt(xx))

    B = np.zeros(P) # shape (P,)
    B[model_terms] = bb[1:]
    B[np.invert(model_terms)] = b2.T
    SE = np.zeros(P)
    SE[model_terms] = se[1:]
    SE[np.invert(model_terms)] = s2.T
    
    # Numpy handles multi-row/column indexing differently than MATLAB
    # Commented out code not tested
    """
    COVB = np.zeros([P,P])
    COVB = covb[1:,1:]
    if COVB.size:
        COVB[:] = COVB[:,model_terms]
        COVB[:] = COVB[model_terms,:]"""
    # Stats
    PVAL = np.zeros(P)
    tstat = np.zeros(P)
    if model_terms.any():
        tval = safedivide(B[model_terms],SE[model_terms])
        ptemp = 2*stats.t.cdf(-abs(tval),dfe)
        PVAL[model_terms] = ptemp
        tstat[model_terms] = tval
    if np.invert(model_terms).any():
        if dfe>1:
            tval = safedivide(B[np.invert(model_terms)],SE[np.invert(model_terms)])
            ptemp=2*stats.t.cdf(-abs(tval),dfe-1)
        else:
            tval = np.nan
            ptemp = np.nan
    PVAL[np.invert(model_terms)] = ptemp
    tstat[np.invert(model_terms)] = tval
    """
    # Summary statistics
    MSexplained = safedivide(SStotal-SSresid,df0)
    fstat = safedivide(MSexplained,rmse**2)
    pval = 1-stats.f.cdf(fstat,df0,dfe)
    """
    return B,SE,PVAL

def stepnext(model_terms,pval,B,penter,premove,keep):
    swap = 0
    p = np.nan
    
    termsout = np.argwhere(np.logical_and(np.invert(model_terms),np.invert(keep)))
    # If termsout is not empty
    if termsout.size:
        minval = np.min(pval[termsout])
        minind = np.argmin(pval[termsout])
        if minval<penter:
            swap = termsout[minind]#[0]]
            p = minval
    if swap==0:
        termsin = np.argwhere(np.logical_and(model_terms,np.invert(keep)))
        if termsin.size:
            badterms = termsin[np.isnan(pval[termsin])]
            if badterms.size:
                swap = np.isnan(B[badterms])
                if any(swap):
                    swap = badterms[swap]
                    swap = swap[0]
                else:
                    # if multiple terms contribute to a perfect fit, remove the
                    # one that contributes the least (select the one with the
                    # smallest coeff)
                    swap = np.argmin(abs(B[badterms]))
                    swap = badterms[swap]
                p = np.nan
            else:
                maxval = max(pval[termsin])
                maxind = np.argmax(pval[termsin])
                if maxval>premove:
                    swap=termsin[maxind]
                    p = maxval
    return swap,p

def safedivide(numer,denom):
    t = denom==0
    if not np.any(t) or not numer.size:
        quotient = np.divide(numer,denom)
    else:
        quotient = np.array(np.divide(np.zeros(np.shape(numer)),np.ones(np.shape(denom))))
        if np.size(numer)==1 and np.size(denom)>1:
            numer = mb.repmat(numer,np.size(denom))
        elif np.size(denom)==1 and np.size(numer)>1:
            denom = mb.repmat(denom,np.size(numer))
            t = denom==0
        quotient[np.invert(t)] = np.divide(numer[np.invert(t)],denom[np.invert(t)])
        quotient[t] = np.multiply(np.inf,np.sign(numer[t]))
    return quotient
