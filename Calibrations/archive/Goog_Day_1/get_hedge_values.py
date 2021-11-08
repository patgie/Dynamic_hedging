import sys
import os
sys.path.append(os.path.dirname('__file__'))
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import math
import time
import copy
import argparse
import random

from py_vollib.black_scholes.implied_volatility import implied_volatility 
from py_vollib.black_scholes import black_scholes as price_black_scholes 

from networks import *



class Net_LSV(nn.Module):
    """
    Calibration of LSV model to vanilla prices at different maturities
    """
    def __init__(self, device,n_strikes, n_networks):
        
        super(Net_LSV, self).__init__()
        self.device = device

        
        # initialise price diffusion neural network (different neural network for each maturity)
        self.S_vol =  Net_timegrid(dim=3, nOut=1, n_layers=3, vNetWidth=100, n_networks=1, activation_output="softplus")
        
        # initialise vanilla hedging strategy neural networks 
        """
        network for each maturity is used to hedge only options for that maturity, for example network corresponding to final maturity
        is used to simulate hedging strategy (from time 0) for vanilla options at the final maturity
        """
        self.vanilla_hedge = Net_timegrid(dim=2, nOut=n_strikes, n_layers=2, vNetWidth=20, n_networks=n_maturities)

        # initialise stochastic volatility drift and diffusion neural networks
        self.V_drift = Net_timegrid(dim=1, nOut=1, n_layers=2, vNetWidth=20, n_networks=1)
        self.V_vol = Net_timegrid(dim=1, nOut=1, n_layers=2, vNetWidth=20, n_networks=1, activation_output="softplus")
        
        # initialise stochastic volatility correlation and initial value parameters
        self.v0 = torch.nn.Parameter(torch.rand(1)*0.1)
        self.rho = torch.nn.Parameter(torch.ones(1)*(-3))
        
        # initialise exotic hedging strategy neural networks 
        """
        network for each maturity is used only to simulate hedging strategy for the timesteps in between previous and current maturity
        so that "n_maturities" networks are used to simulate hedging strategy for exotic option with maturity at the end of considered time horizon
        """
        self.exotic_hedge_straddle = Net_timegrid(dim=2, nOut=1, n_layers=3, vNetWidth=100, n_networks=1)
        self.exotic_hedge_lookback = Net_timegrid(dim=2, nOut=1, n_layers=3, vNetWidth=100, n_networks=1)

        
    def forward(self, timegrid): 
        # initialisation
        ones = torch.ones(1, 1, device=self.device)
        R = np.arange(750, 850.01, 1)
        Hedge = torch.zeros(len(R),len(timegrid)).to(device)
        for [idxr,price] in enumerate(R):
            for [idxt,t] in enumerate(timegrid):
                if args.lookback_exotic==True:
                    Hedge[idxr,idxt] = self.exotic_hedge_lookback.forward_idx(0, torch.cat([ones* t,torch.log(ones * price)],1))
                elif args.straddle_exotic==True:
                    Hedge[idxr,idxt] = self.exotic_hedge_straddle.forward_idx(0, torch.cat([ones* t,torch.log(ones * price)],1))
              
        return Hedge

     
                
def get_hedge(model,args):
    
    model = Net_LSV( device=device,n_strikes=n_strikes, n_networks=1)
    model.to(device)
    
    if args.straddle_exotic==True:
        checkpoint_str= "NSDE_test_hedge_straddle_seed_{}_cal_day_{}.pth.tar".format(seed,cal_day)
    elif args.lookback_exotic==True:
        checkpoint_str= "NSDE_test_hedge_lookback_seed_{}_cal_day_{}.pth.tar".format(seed,cal_day)
    checkpoint=torch.load(checkpoint_str)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    
    with torch.no_grad():
        Hedge = model(timegrid_exotic)
        if args.straddle_exotic==True:   
            print("Hedge surface straddle:",Hedge)
        elif args.lookback_exotic==True: 
            print("Hedge surface lookback:",Hedge)

    Hedge = Hedge.cpu().numpy()
    Hedge = pd.DataFrame(Hedge)
    if args.straddle_exotic==True: 
        hedge_str= "Hedges_straddle_seed_{}_cal_day_{}.csv".format(seed,cal_day) 
    if args.lookback_exotic==True: 
        hedge_str= "Hedges_lookback_seed_{}_cal_day_{}.csv".format(seed,cal_day)    
    
    Hedge.to_csv(hedge_str)

    return    

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()                     
    parser.add_argument('--device', type=int, default=7)
    parser.add_argument('--lookback_exotic',action='store_true', default=True)
    parser.add_argument('--straddle_exotic',action='store_true', default=False)
    args = parser.parse_args()      

    if torch.cuda.is_available():
        device='cuda:{}'.format(args.device)
        torch.cuda.set_device(args.device)
    else:
        device="cpu"
        
    cal_day = np.load("cal_day.npy")
    strikes = np.load("strikes.npy")
    maturity_values = np.load("maturities.npy")

    n_maturities = len(maturity_values)
    n_strikes = len(strikes[0,:])

    timegrid_exotic = np.load("timegrid_exotic.npy")[cal_day:11]
    timegrid = torch.tensor(timegrid_exotic).to(device)

    seed=456

    
    torch.manual_seed(seed) # fixed for reproducibility
    model = Net_LSV( device=device,n_strikes=n_strikes, n_networks=1)
    model.to(device)
        
    checkpoint_str= "NSDE_test_hedge_lookback_seed_{}_cal_day_{}.pth.tar".format(seed,cal_day) 
    checkpoint=torch.load(checkpoint_str)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

                               
    # get lookback hedge values    
    get_hedge(model,args)
    
    checkpoint_str= "NSDE_test_hedge_straddle_seed_{}_cal_day_{}.pth.tar".format(seed,cal_day) 
    checkpoint=torch.load(checkpoint_str)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    
    parser = argparse.ArgumentParser()                     
    parser.add_argument('--device', type=int, default=7)
    parser.add_argument('--lookback_exotic',action='store_true', default=False)
    parser.add_argument('--straddle_exotic',action='store_true', default=True)
    args = parser.parse_args()      

    # get straddle hedge values     
    get_hedge(model,args)

    
    
    