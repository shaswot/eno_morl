#!/usr/bin/env python
# coding: utf-8

# python ./c_morl.py --env=cenp --gamma=0.997 --noise=0.7 --pref=0.5 --seed=123

# In[1]:
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import random
import os
import os.path
import argparse
import sys

import gym
gym.logger.set_level(40) # remove gym warning about float32 bound box precision

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt


# In[2]:


import common.env_lib
from common.env_utils import sorl_plot
from common.rl_lib import ReplayBuffer, OUNoise, ENoise, ValueNetwork, PolicyNetwork, ddpg_update, calc_values_of_states


# # In[3]:
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="csense", type=str, help="Environment. Specified in ./common/env_lib.py")
parser.add_argument("--gamma", default=0.996, type=float, help="Discount Factor")
parser.add_argument("--noise", default=0.8, type=float, help="Action Noise")
parser.add_argument("--seed", default=238, type=int, help="Set seed [default: 238]")
parser.add_argument("--pref", default=0.5, type=float, help="Set preference [default: 0.5]")


args = parser.parse_args()


# arguments
seed      = args.seed
env_name  = args.env
GAMMA     = args.gamma
max_noise = args.noise
pref      = args.pref

# In[4]:


# set seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# In[5]:

################################################################################
# If using GPU
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

################################################################################
# Setup experiment parameters
timeslots_per_day = 24
REQ_TYPE = "random"

cur_folder = os.getcwd()
model_folder = os.path.join(cur_folder,"models")

# Type of RL training used for the model that we are going to load
training_experiment = "base_g" + str(GAMMA) + "-n" + str(max_noise)
################################################################################

# Tags for ENP model
enp_env_tag = env_name + '_t' + str(timeslots_per_day) + '_' + REQ_TYPE
# name of folder to load model from
enp_model_folder = enp_env_tag  + "-" + training_experiment 

# model filename
enp_model_tag   = enp_model_folder +'-' + str(seed)
enp_pmodel_file = os.path.join(model_folder, enp_model_folder, (enp_model_tag + "-policy.pt"))
enp_qmodel_file = os.path.join(model_folder, enp_model_folder, (enp_model_tag + "-value.pt"))
################################################################################

# Tags for sense model
sense_env_tag = "csense" + '_t' + str(timeslots_per_day) + '_' + REQ_TYPE
# name of folder to load model from
sense_model_folder = sense_env_tag  + "-" + training_experiment 

# model filename
sense_model_tag   = sense_model_folder + '-' + str(seed)
sense_pmodel_file = os.path.join(model_folder, sense_model_folder, (sense_model_tag + "-policy.pt"))
sense_qmodel_file = os.path.join(model_folder, sense_model_folder, (sense_model_tag + "-value.pt"))

################################################################################
# Setting up runtime environment base
experiment = "morl_runtimev2_g" + str(GAMMA) + "-n" + str(max_noise) + "-p" + str(pref)
#env: cenp -> cenpsense
env = eval("common.env_lib."+env_name+"sense"+"()")
env.set_pref(pref) # preference over actions
################################################################################

state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
hidden_dim = 256

# Make model
sense_pnet = PolicyNetwork(state_dim, action_dim, hidden_dim, device).to(device)
enp_pnet = PolicyNetwork(state_dim, action_dim, hidden_dim, device).to(device)

sense_qnet = ValueNetwork(state_dim, action_dim, hidden_dim, device).to(device)
enp_qnet = ValueNetwork(state_dim, action_dim, hidden_dim, device).to(device)

sense_pnet.load_state_dict(torch.load(sense_pmodel_file))
sense_pnet.eval()

sense_qnet.load_state_dict(torch.load(sense_qmodel_file))
sense_qnet.eval()

enp_pnet.load_state_dict(torch.load(enp_pmodel_file))
enp_pnet.eval()

enp_qnet.load_state_dict(torch.load(enp_qmodel_file))
enp_qnet.eval();

# Tags for morl_runtime experiment
env_tag = env_name + '_t' + str(timeslots_per_day) + '_' + REQ_TYPE
model_tag = experiment +'-'+str(seed)

# experiment tag
# name of folder to save models and results
exp_tag = env_tag  + "-" + experiment 

# experiment+seed tag
# tensorboard tag / model filename
tag     = env_tag  + "-" + experiment +'-'+str(seed) 
print("Experiment TAG: ",tag)

# Folder/file to save test results
test_results_folder = os.path.join(cur_folder,"results", exp_tag, "test")
if not os.path.exists(test_results_folder): 
        os.makedirs(test_results_folder) 
test_log_file = os.path.join(test_results_folder, tag + '-test.npy')    

# Setup environment
env_location_list = ['tokyo']#['tokyo','wakkanai','minamidaito']
START_YEAR = 1995
NO_OF_YEARS = 24
# timeslots_per_day = 24
# REQ_TYPE = "random"
prediction_horizon = 10*timeslots_per_day
henergy_mean= 0.13904705134356052 # 10yr hmean for tokyo

exp_test_log = {} # dictionary to store test results
intrp_no =4
for env_location in env_location_list:
#     print(env_location)
    exp_test_log[env_location] = {}
    for year in range(START_YEAR, START_YEAR+NO_OF_YEARS):
        exp_test_log[env_location][year]={}
        
        env.set_env(env_location,year, timeslots_per_day, 
                    REQ_TYPE, offset=timeslots_per_day/2,
                    p_horizon=prediction_horizon,
                    hmean=henergy_mean)    
        state = env.reset()
        reward_rec = []
        
        intrp_actions_rec = []
        intrp_sense_value_rec = []
        intrp_enp_value_rec = []
        intrp_final_value_rec = []
        
        ep_done_rec = []
        done = False
        while not done:
            if env.RECOVERY_MODE:
                no_action = 0            
                next_state, reward, done, _ = env.step(no_action)       
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

                    # sense_model# sense_model: get action and q-value
                    # convert action to DC
                    raw_sense_action_tensor = sense_pnet(state_tensor)
                    sense_value = sense_qnet(state_tensor, raw_sense_action_tensor).cpu().item()
                    raw_sense_action = raw_sense_action_tensor.cpu().item()

                    # enp_model: get action and q-value
                    # convert action to DC
                    enp_action_tensor = enp_pnet(state_tensor)
                    enp_value = enp_qnet(state_tensor, enp_action_tensor).cpu().item()
                    raw_enp_action = enp_action_tensor.cpu().item()

                # Find two intermediate DC
                intrp_actions = np.linspace(start=min(raw_sense_action,raw_enp_action), 
                                                stop=max(raw_sense_action,raw_enp_action), 
                                                num=intrp_no)
                intrp_actions_rec.append(intrp_actions)

                # Convert intrp_actions to action
                intrp_action_tensor = torch.FloatTensor(intrp_actions).unsqueeze(1).to(device)
                batch_state_tensor = torch.FloatTensor([state]*intrp_no).to(device)

                # Get the q-values from sense and enp models for the intrp_actions
                intrp_sense_value = sense_qnet(batch_state_tensor, intrp_action_tensor).cpu().detach().numpy().squeeze()
                intrp_enp_value = enp_qnet(batch_state_tensor, intrp_action_tensor).cpu().detach().numpy().squeeze()
                intrp_sense_value_rec.append(intrp_sense_value)
                intrp_enp_value_rec.append(intrp_enp_value)
                
                # Calculate the node utility for all intrp_actions
                final_value = pref*intrp_sense_value + (1-pref)*intrp_enp_value
                intrp_final_value_rec.append(final_value)
                    
                # final_action is the action with the maximum final_value
                final_raw_action = intrp_actions[final_value.argmax()]
                tr_action = (final_raw_action*0.5 + 0.5)
                
                # Execute action
                next_state, reward, done, _ = env.step(tr_action)

            reward_rec.append(reward)
            ep_done = done or env.RECOVERY_MODE
            ep_done_rec.append(ep_done)
            state = next_state

        # Log the traces and summarize results
        iteration_result={}

        # Saving traces
        iteration_result['reward_rec'] = np.array(reward_rec)
        iteration_result['ep_done_rec'] = np.array(ep_done_rec)
        iteration_result['action_log'] = np.array(env.action_log)
        iteration_result['sense_dc_log'] = np.array(env.sense_dc_log)
        iteration_result['env_log'] = np.array(env.env_log)
        iteration_result['eno_log'] = np.array(env.eno_log)
        iteration_result['sense_reward_log'] = np.array(env.sense_reward_log)
        iteration_result['enp_reward_log'] = np.array(env.enp_reward_log)
        
        iteration_result['intrp_actions_rec'] = intrp_actions_rec
        iteration_result['intrp_sense_value_rec'] = intrp_sense_value_rec
        iteration_result['intrp_enp_value_rec'] = intrp_enp_value_rec
        iteration_result['intrp_final_value_rec'] = intrp_final_value_rec

        # Summarizing results
        env_log = iteration_result['env_log']

        # Get henergy metrics
        henergy_rec = env_log[:,1]
        avg_henergy = henergy_rec.mean()
        iteration_result['avg_henergy'] = avg_henergy

        # Get req metrics
        req_rec = env_log[:,5]
        avg_req = req_rec.mean()            
        iteration_result['avg_req'] = avg_req

        # Get reward metrics
        # In this case, the reward metrics directly reflect the conformity
        reward_rec = iteration_result['reward_rec']
        # negative rewards = -1000 correspond to downtimes
        # To find average reward, remove negative values
        index = np.argwhere(reward_rec<0)
        rwd_rec = np.delete(reward_rec, index)
        avg_rwd = rwd_rec.mean()
        iteration_result['avg_rwd'] = avg_rwd

        # Get downtime metrics
        ep_done_rec =  iteration_result['ep_done_rec']
        downtimes = np.count_nonzero(ep_done_rec[:-1] < ep_done_rec[1:]) - 1 # last env done is excluded
        iteration_result['downtimes'] = downtimes

        # Get ENP metrics
        eno_log = iteration_result['eno_log']
        enp_log = []
        enp_log.append(eno_log[0])
        for t in range(1,len(eno_log)):
            enp = enp_log[-1] + eno_log[t]
            enp = np.clip(enp,0,1)
            enp_log.append(enp)
        iteration_result['enp_log'] = np.array(enp_log)

#         print("Year:\t", year)            
        exp_test_log[env_location][year] = iteration_result
#     print("")


# In[19]:
################################################################################
# Save Test Results
np.save(test_log_file, exp_test_log)

# In[20]:
################################################################################
# summarize metrics and display
print("\n\n***TEST RESULTS****")
print("Tag:", tag)
print("Seed:", seed)


print("LOCATION".ljust(12), "YEAR".ljust(6), "HMEAN".ljust(8), "REQ_MEAN".ljust(8), "AVG_DC".ljust(8), 
      "SNS_RWD".ljust(8), "ENP_RWD".ljust(8), "AVG_RWD".ljust(8), "DOWNTIMES".ljust(9))

exp_result = exp_test_log
location_list = list(exp_result.keys())
for location in location_list:
    yr_list = list(exp_result[location].keys())
    for year in yr_list:
        run_log = exp_result[location][year]
        # Print summarized metrics
        print(location.ljust(12), year, end=' ')
        sense_avg_rwd = run_log['sense_reward_log'].mean()
        enp_avg_rwd = run_log['enp_reward_log'].mean()
    
        average_rwd = run_log['avg_rwd']
        total_downtimes = run_log['downtimes']
        hmean = run_log['avg_henergy']
        reqmean = run_log['avg_req']
        sense_dc_mean = run_log['sense_dc_log'].mean()

        print(f'{hmean:7.3f}',end='  ')
        print(f'{reqmean:7.3f}',end='  ')
        print(f'{sense_dc_mean:7.3f}',end='  ')
        print(f'{sense_avg_rwd:7.3f}',end='  ')
        print(f'{enp_avg_rwd:7.3f}',end='  ')
        print(f'{average_rwd:7.3f}',end='  ')
        print(f'{total_downtimes:5d}',end='  ')
        print("")

