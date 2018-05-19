# -*- coding: utf-8 -*-
"""
Credits: https://github.com/ttricco/cta200h

https://github.com/ttricco/cta200h/blob/master/lecture8/monte_carlo.ipynb

"""

import numpy as np

#Define a multimodal distribution (remember that the normalization doesn't matter):
def multimodal(samp,cov,invcov):
    x = samp[0]
    y = samp[1]
    return twoDGauss(samp,cov,invcov) + 0.1*np.exp(-0.5*((x-5.)**2 + (y-5)**2))

#Define the D-dimensional Gaussian:      
def twoDGauss(samp,cov,invcov):
    x = samp[0]
    y = samp[1]
    det = np.linalg.det(cov)
    norm = 1./(2.*np.pi)/det**0.5
    return norm*np.exp(-0.5*(x*(invcov[0,0]*x + invcov[0,1]*y) + y*(invcov[1,0]*x + invcov[1,1]*y)))

#Define the proposal step (we are using a symmetric Gaussian proposal density):
def proposal(oldsamp,sigmaprop,D):
    newsamp = oldsamp + np.random.normal(0.,sigmaprop,D)
    return newsamp

#Define the acceptance-rejection step (return the new sample and a boolean that tells us whether
#or not the new sample was accepted):
def accept(newsamp,oldsamp,paramrange,P,*Pargs):
    if not ((np.array([p1 - p2 for p1, p2 in zip(newsamp, np.transpose(paramrange)[:][0])])>0).all() and (np.array([p1 - p2 for p1, p2 in zip(np.transpose(paramrange)[:][1],newsamp)])>0).all()):
        acc = False
        return acc, oldsamp # make sure the samples are in the desired range
    newprob = P(newsamp,*Pargs)
    oldprob = P(oldsamp,*Pargs)
    if newprob >= oldprob:
        acc = True
        return acc, newsamp
    else:
        prob = newprob/oldprob
        acc = np.random.choice([True,False],p=[prob,1.-prob])
        return acc, acc*newsamp + (1. - acc)*oldsamp #Note that this is either newsamp or oldsamp

#Define function that runs an entire chain:
def run_chain(steps,paramrange,sigmaprop,D,P,*Pargs):
    oldsamp=np.array([np.random.uniform(paramrange[d][0],paramrange[d][1]) for d in range(D)])#Draw a random starting point
    count = 0 #Count the number of accepted samples
    samples = [oldsamp] #Store all samples
    for i in range(steps):
        newsamp = proposal(oldsamp,sigmaprop,D) #Propose a new sample
        acc, newsamp = accept(newsamp,oldsamp,paramrange,P,*Pargs) #decide whether or not to accept it
        samples.append(newsamp) #Add the sample to the list of samples
        if acc:
            count += 1
        oldsamp = newsamp #Move to the new sample
    ar = 1.*count/steps #compute the acceptance ratio
    return np.array(samples), ar
    