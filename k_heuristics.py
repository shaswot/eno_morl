import random
import os
import os.path
import argparse

import gym
gym.logger.set_level(40) # remove gym warning about float32 bound box precision

import numpy as np
import matplotlib.pyplot as plt

import common.env_lib
from common.env_utils import sorl_plot

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="csense", type=str, help="Environment. Specified in ./common/env_lib.py")
parser.add_argument("--seed", default=238, type=int, help="Set seed [default: 238]")


args = parser.parse_args()


# arguments
seed      = args.seed
env_name  = args.env


# set seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)


timeslots_per_day = 24
env = eval("common.env_lib."+env_name+"()")

# non-linear batt-centric policy generator
def k_factor(batt,k=0.95):
    x = (batt-0.1)/0.9
    y = (x-k*x)/(k-2*k*x+1)
    action = y
    return action


exp_heuristics={'k1':0.1, # most linear
                'k2':0.2, 
                'k3':0.3, 
                'k4':0.4,
                'k5':0.5,
                'k6':0.6,
                'k7':0.7,
                'k8':0.8,
                'k9':0.9, # sharpest ascent
               }
                  

cur_folder = os.getcwd()
for experiment in list(exp_heuristics.keys()):
    env_location_list = ['tokyo']#,'wakkanai','minamidaito']
    START_YEAR = 1995
    NO_OF_YEARS = 24
    timeslots_per_day = 24
    REQ_TYPE = "random"
    prediction_horizon = 10*timeslots_per_day
    henergy_mean= 0.13904705134356052 # 10yr hmean for tokyo
    
    # Tags
    env_tag = env_name + '_t' + str(timeslots_per_day) + '_' + REQ_TYPE
    model_tag = experiment +'-'+str(seed)

    # experiment tag
    # name of folder to save models and results
    exp_tag = env_tag  + "-" + experiment 

    # experiment+seed tag
    # tensorboard tag / model filename
    tag     = exp_tag +'-' + str(seed) 
#     print("Experiment tag: ",tag)
    
    # Folder/file to save test results
    test_results_folder = os.path.join(cur_folder,"results", exp_tag, "test")
    if not os.path.exists(test_results_folder): 
            os.makedirs(test_results_folder) 
    test_log_file = os.path.join(test_results_folder, tag + '-test.npy')    
 
    
    exp_result = {}
    for env_location in env_location_list:
#         print(env_location, end='\t')
        exp_result[env_location] = {}
        for year in range(START_YEAR, START_YEAR+NO_OF_YEARS):
            env.set_env(env_location, year , timeslots_per_day, 
                        REQ_TYPE, offset=timeslots_per_day/2,
                        p_horizon=prediction_horizon,
                        hmean=henergy_mean)    
            state = env.reset()
            reward_rec = []
            ep_done_rec = []
            done = False
            while not done:
                if env.RECOVERY_MODE:
                    no_action = 0            
                    next_state, reward, done, _ = env.step(no_action)       
                else:
                    paction = k_factor(env.benergy_obs,k=exp_heuristics[experiment])
                    next_state, reward, done, _ = env.step(paction)                
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
            batt_rec = env_log[:,3]
            batt_rec[batt_rec>0.1]=0
            batt_rec[batt_rec!=0]=1
            downtimes = np.count_nonzero(batt_rec[:-1] < batt_rec[1:])
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
#             print(year,end=", ")            
            exp_result[env_location][year] = iteration_result
#     print('')
    np.save(test_log_file, exp_result)

# summarize metrics and display

for experiment in list(exp_heuristics.keys()):
    
    # Tags
    env_tag = env_name + '_t' + str(timeslots_per_day) + '_' + REQ_TYPE
    model_tag = experiment +'-'+str(seed)

    # experiment tag
    # name of folder to save models and results
    exp_tag = env_tag  + "-" + experiment 

    # experiment+seed tag
    tag     = exp_tag +'-'+str(seed)     
    
    # Folder/file to load test results
    test_results_folder = os.path.join(cur_folder,"results", exp_tag, "test")
    if not os.path.exists(test_results_folder): 
        print("Experiment has not been run")
    test_log_file = os.path.join(test_results_folder, tag + '-test.npy')    
 
    # Load data
    exp_result = np.load(test_log_file,allow_pickle='TRUE').item()    
    
    print("Experiment:", tag)
    print("LOCATION".ljust(12), "YEAR".ljust(6), "HMEAN".ljust(8), "REQ_MEAN".ljust(8), "AVG_DC".ljust(8), 
      "SNS_RWD".ljust(8), "ENP_RWD".ljust(8), "AVG_RWD".ljust(8), "DOWNTIMES".ljust(9))

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
    print('*'*90)
    print('\n')