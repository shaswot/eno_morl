import numpy as np
import gym
from gym import spaces

from common.harvester_fns import csv_solar_harvester
from common.requester_fns import request_gen
from common.env_utils import battery, tx_prob

from collections import deque
class csense(gym.Env):    
    def __init__(self):
        super(csense, self).__init__()
        
        # Actions = energy_consumed
        self.action_space = spaces.Box(low=0,
                                       high=1,
                                       shape=(1,))

        # Observation = [time, h_energy, p_energy, b_energy, m_energy, req]
        self.observation_space = spaces.Box(low=0, 
                                            high=1, 
                                            shape=(6,))
########################################################

    def set_env(self,location, year, timeslots_per_day, req_type, offset, p_horizon, hmean=None):
        self.location = location
        self.year = year
        self.REQ_TIMESLOTS_PER_DAY=timeslots_per_day
        self.HFACTOR = 5*1E-3*240/self.REQ_TIMESLOTS_PER_DAY
        self.DFACTOR = 5*1E-3*240/self.REQ_TIMESLOTS_PER_DAY
        self.rq_type = req_type
        self.offset = int(offset) # offset for request generator [when using random_day]
        self.MIN_BATT = 0.1
        self.MIN_DC = 0.1
        self.p_horizon = p_horizon # prediction horizons
        self.hmean = hmean # mean henergy if given
########################################################

    def reset(self):
        # Characterize the harvester
        self.READINGS_PER_DAY = 24       

        self.env_harvester = csv_solar_harvester(location=self.location,
                                                 year=self.year,
                                                 READINGS_PER_DAY = self.READINGS_PER_DAY,
                                                 SMAX=4.0, # Max GSR
                                                 HENERGY_NOISE=0.0, # henergy artifical noise
                                                 NORMALIZED_HMIN_THRES=1E-5, # henergy cutoff
                                                 REQ_TIMESLOTS_PER_DAY=self.REQ_TIMESLOTS_PER_DAY, # no. of timeslots per day
                                                 PREDICTION_HORIZON=self.p_horizon, # lookahead horizon to predict energy
                                                 PREDICTION_NOISE=0.0) # preidction noise
        self.env_timeslot_values = self.env_harvester.time_slots
        self.ENV_LIFETIME = self.env_harvester.no_of_days
        
        # Characterize the battery
        self.BINIT = 0.7
        self.BEFF  = 1.0
        self.env_battery = battery(self.BINIT,self.BEFF)

        # Characterize request generator
        hstream = np.array(self.env_harvester.henergy_stream)
        if self.hmean is None:
            self.hmean = hstream.mean()
        self.req_stream =  request_gen(self.rq_type,
                                       htrace=hstream,
                                       hmean=self.hmean,
                                       mindc=self.MIN_DC,
                                       timesteps_per_day=self.REQ_TIMESLOTS_PER_DAY,
                                       offset=self.offset)
       
        # Data logging variables
        self.env_log = [] # record all values in the environment
        self.action_log = [] # record all actions sent to the environment
        self.eno_log = []
        self.sense_dc_log = []
        self.batt_slice = deque(maxlen=self.REQ_TIMESLOTS_PER_DAY*10)
        self.sense_reward_log = []
        self.enp_reward_log = []

        # Observation variables
        self.time_obs = None
        self.henergy_obs = None
        self.penergy_obs = None
        self.benergy_obs = None
        self.menergy_obs = None
        self.req_obs = None
        
        # Environment Flags
        self.RECOVERY_MODE = False # battery is recovering to BINIT from complete discharge & node is suspended
        
        # Get observation
        self.time_obs, self.henergy_obs, self.penergy_obs, DAY_END, HARVESTER_END = self.env_harvester.step()
        self.benergy_obs = self.env_battery.get_batt_state()
        self.eno_log.append(self.benergy_obs)
        self.batt_slice.append(self.benergy_obs)
        self.menergy_obs = np.mean(self.batt_slice) # mean battery for given prediction horizon
        self.req_obs = self.req_stream[int(self.env_harvester.global_time)]
        self.obs = (self.time_obs/self.READINGS_PER_DAY,
                    self.henergy_obs,
                    self.penergy_obs,
                    self.benergy_obs,
                    self.menergy_obs,
                    self.req_obs)
        log_data = self.obs
        self.env_log.append(log_data)
        return np.array(self.obs)
########################################################

    def action2sensedc(self,action):
        if action < 0:
            sense_dc = 0
        else:
            sense_dc = max(self.MIN_DC,action*self.req_obs)
        return sense_dc    
########################################################

    def step(self, action):
        # if the node is in recovery mode at any time during step(), 
        # this flag is set
        recovery_flag = self.RECOVERY_MODE

        # Execute Action
        if self.RECOVERY_MODE: # Is node already in recovery mode?
            self.recovery_action()
            recovery_flag = True
        else:
            ACTION_VALID = self.verify_action(action)
            if ACTION_VALID:
                self.execute_action(action)
            else:
                self.recovery_action()
                recovery_flag = True

        # Get next observation
        next_obs, done = self.next_obs() # this might change self.RECOVERY_MODE
        
        # Did the node enter recovery mode after executing action even though the action was valid?
        recovery_flag = recovery_flag or self.RECOVERY_MODE
        
        # Get reward
        # If the action was valid but resulted in entering recovery mode, then
        # reward should not be given by the action but rather by recovery mode.
        if recovery_flag: 
            reward = self.reward(action=-1)
        else:
            reward = self.reward(action)
        info = {}
        
        return np.array(next_obs), reward, done, info
########################################################

    def verify_action(self, action): # check if actions are valid
        assert self.RECOVERY_MODE == False, "Action does not need to be verified in recovery mode"
        sense_dc = self.action2sensedc(action)
        
        surplus_energy = (self.henergy_obs*self.HFACTOR - (sense_dc)*self.DFACTOR)
        if (-surplus_energy) < self.benergy_obs: # if there is sufficient energy in the battery to extract
            return True # valid action
        else:
            self.RECOVERY_MODE = True # switch to recovery mode
            return False # invalid action
########################################################

    def recovery_action(self):
        assert self.RECOVERY_MODE == True, "Node is not in recovery mode"
#         charge_qty =self.henergy_obs*self.HFACTOR
        charge_qty = self.BINIT/2
        self.env_battery.charge(charge_qty)
        self.batt_slice.clear()
        self.eno_log.append(charge_qty)
        self.action_log.append(-1)
        self.sense_dc_log.append(0)
########################################################

    def execute_action(self, action): 
        assert self.RECOVERY_MODE==False, "Node is in recovery mode. Cannot execute action"

        sense_dc = self.action2sensedc(action)
        surplus_energy = (self.henergy_obs*self.HFACTOR - (sense_dc)*self.DFACTOR)
        
        if surplus_energy >= 0:
            self.env_battery.charge(surplus_energy)
        else:
            self.env_battery.discharge(surplus_energy)
        
        self.eno_log.append(surplus_energy)
        self.action_log.append(action)      
        self.sense_dc_log.append(sense_dc)
########################################################

    def next_obs(self): # update all observations

        self.time_obs, self.henergy_obs, self.penergy_obs, DAY_END, HARVESTER_END = self.env_harvester.step()

        if not HARVESTER_END:
            self.req_obs = self.req_stream[int(self.env_harvester.global_time)]
            self.benergy_obs = self.env_battery.get_batt_state() # updated battery observation
            self.batt_slice.append(self.benergy_obs)
            self.menergy_obs = np.mean(self.batt_slice)
            
            if self.benergy_obs < self.MIN_BATT: # Is battery less than a threshold?
                self.RECOVERY_MODE = True
            
            if self.RECOVERY_MODE:
                if self.benergy_obs > self.BINIT: 
                    self.RECOVERY_MODE = False # snap out of recovery mode
                else:
                    self.RECOVERY_MODE = True # remain in recovery mode
            
        self.obs = (self.time_obs/self.READINGS_PER_DAY,
                    self.henergy_obs,
                    self.penergy_obs,
                    self.benergy_obs,
                    self.menergy_obs,
                    self.req_obs)
        
        done = HARVESTER_END
        
        # don't log the values of the last next_state when harvester ends
        # this causes mismatch in array sizes with action_log/reward_rec etc.
        if not done: 
            log_data = self.obs
            self.env_log.append(log_data)

        return self.obs, done
########################################################

    def reward(self,action): # reward based on utility
        if action < 0:
            sense_reward = 0
            enp_reward = 0
            reward = 0
            
        else:
            sense_dc = self.action2sensedc(action)
            sense_reward = min(1.0,sense_dc/self.req_obs)
            
            batt_threshold = 0.8
            if self.menergy_obs > batt_threshold:
                enp_reward = 1
            else:
                enp_reward = (self.menergy_obs - self.MIN_BATT)/(batt_threshold-self.MIN_BATT)
        
        reward = sense_reward
        self.sense_reward_log.append(sense_reward)
        self.enp_reward_log.append(enp_reward)
        return reward
########################################################
# End of csense()
########################################################
class cenp(csense):
    
    def reward(self,action): # reward based on utility
            if action < 0:
                sense_reward = 0
                enp_reward = 0
                reward = 0

            else:
                sense_dc = self.action2sensedc(action)
                sense_reward = min(1.0,sense_dc/self.req_obs)

                batt_threshold = 0.8
                if self.menergy_obs > batt_threshold:
                    enp_reward = 1
                else:
                    enp_reward = (self.menergy_obs - self.MIN_BATT)/(batt_threshold-self.MIN_BATT)

            reward = enp_reward
            self.sense_reward_log.append(sense_reward)
            self.enp_reward_log.append(enp_reward)
            return reward
########################################################
# End of cenp()
########################################################
class cenpv2(csense):
    
    def reward(self,action): # reward based on utility
            if action < 0:
                sense_reward = 0
                enp_reward = 0
                reward = 0

            else:
                sense_dc = self.action2sensedc(action)
                sense_reward = min(1.0,sense_dc/self.req_obs)

                batt_threshold = 0.8
                BRWD = 0.85
                BMAX = 1.0
                if self.menergy_obs > batt_threshold:
                    enp_reward = (self.menergy_obs - BMAX)*(1-BRWD)/(batt_threshold-BMAX)
                else:
                    enp_reward = (self.menergy_obs - self.MIN_BATT)/(batt_threshold-self.MIN_BATT)

            reward = enp_reward
            self.sense_reward_log.append(sense_reward)
            self.enp_reward_log.append(enp_reward)
            return reward
########################################################
# End of cenpv2()
########################################################
class cenpsense(csense):
    
    def set_pref(self,preference):
        self.preference = preference
    
    def reward(self,action): # reward based on utility
            if action < 0:
                sense_reward = 0
                enp_reward = 0
                reward = 0

            else:
                sense_dc = self.action2sensedc(action)
                sense_reward = min(1.0,sense_dc/self.req_obs)

                batt_threshold = 0.8
                BRWD = 0.85
                BMAX = 1.0
                if self.menergy_obs > batt_threshold:
                    enp_reward = 1
                else:
                    enp_reward = (self.menergy_obs - self.MIN_BATT)/(batt_threshold-self.MIN_BATT)

            reward = self.preference*sense_reward + (1-self.preference)*enp_reward
            self.sense_reward_log.append(sense_reward)
            self.enp_reward_log.append(enp_reward)
            return reward
########################################################
# End of cenpsense()
########################################################
class cenpv2sense(csense):
    
    def set_pref(self,preference):
        self.preference = preference
    
    def reward(self,action): # reward based on utility
            if action < 0:
                sense_reward = 0
                enp_reward = 0
                reward = 0

            else:
                sense_dc = self.action2sensedc(action)
                sense_reward = min(1.0,sense_dc/self.req_obs)

                batt_threshold = 0.8
                BRWD = 0.85
                BMAX = 1.0
                if self.menergy_obs > batt_threshold:
                    enp_reward = (self.menergy_obs - BMAX)*(1-BRWD)/(batt_threshold-BMAX)
                else:
                    enp_reward = (self.menergy_obs - self.MIN_BATT)/(batt_threshold-self.MIN_BATT)

            reward = self.preference*sense_reward + (1-self.preference)*enp_reward
            self.sense_reward_log.append(sense_reward)
            self.enp_reward_log.append(enp_reward)
            return reward
########################################################
# End of cenpv2sense()
########################################################
class rsense(csense):
    #action = (0,1) --> sense_dc = (0.1,0.5)
    def action2sensedc(self,action):
        if action < 0:
            sense_dc = 0
        else:
            sense_dc = self.MIN_DC + action*0.4
        return sense_dc
    
    def verify_action(self, action): # check if actions are valid
        assert self.RECOVERY_MODE == False, "Action does not need to be verified in recovery mode"
        sense_dc = self.action2sensedc(action)
        
        surplus_energy = (self.henergy_obs*self.HFACTOR - (sense_dc)*self.DFACTOR)
        if (-surplus_energy) < self.benergy_obs: # if there is sufficient energy in the battery to extract
            return True # valid action
        else:
            self.RECOVERY_MODE = True # switch to recovery mode
            return False # invalid action
        
    def execute_action(self, action): 
        assert self.RECOVERY_MODE==False, "Node is in recovery mode. Cannot execute action"

        sense_dc = self.action2sensedc(action)
        surplus_energy = (self.henergy_obs*self.HFACTOR - (sense_dc)*self.DFACTOR)
        
        if surplus_energy >= 0:
            self.env_battery.charge(surplus_energy)
        else:
            self.env_battery.discharge(surplus_energy)
        
        self.eno_log.append(surplus_energy)
        self.action_log.append(action)
        self.sense_dc_log.append(sense_dc)
########################################################
# End of rsense()
########################################################
class renp(rsense):
    
    def reward(self,action): # reward based on utility
            if action < 0:
                sense_reward = 0
                enp_reward = 0
                reward = 0

            else:
                sense_dc = self.action2sensedc(action)
                sense_reward = min(1.0,sense_dc/self.req_obs)

                batt_threshold = 0.8
                if self.menergy_obs > batt_threshold:
                    enp_reward = 1
                else:
                    enp_reward = (self.menergy_obs - self.MIN_BATT)/(batt_threshold-self.MIN_BATT)

            reward = enp_reward
            self.sense_reward_log.append(sense_reward)
            self.enp_reward_log.append(enp_reward)
            return reward
########################################################
# End of renp()
########################################################
class renpv2(rsense):
    
    def reward(self,action): # reward based on utility
            if action < 0:
                sense_reward = 0
                enp_reward = 0
                reward = 0

            else:
                sense_dc = self.action2sensedc(action)
                sense_reward = min(1.0,sense_dc/self.req_obs)

                batt_threshold = 0.8
                BRWD = 0.85
                BMAX = 1.0
                if self.menergy_obs > batt_threshold:
                    enp_reward = (self.menergy_obs - BMAX)*(1-BRWD)/(batt_threshold-BMAX)
                else:
                    enp_reward = (self.menergy_obs - self.MIN_BATT)/(batt_threshold-self.MIN_BATT)

            reward = enp_reward
            self.sense_reward_log.append(sense_reward)
            self.enp_reward_log.append(enp_reward)
            return reward
########################################################
# End of renp()
########################################################
# ######################################################### 
# class tx(gym.Env):
#     """An ambient environment simulator for OpenAI gym."""
#     metadata = {'render.modes': ['human']}
    
#     def __init__(self):
#         super(tx, self).__init__()
        
#         # Actions = energy_consumed
#         self.action_space = spaces.Box(low=0,
#                                        high=1,
#                                        shape=(1,))

#         # Observation = [time, h_energy, p_energy, b_energy, m_energy, ch_noise]
#         self.observation_space = spaces.Box(low=0, 
#                                             high=1, 
#                                             shape=(6,)) #<<<<<<<<<
        
        
#     def set_env(self,location, year, timeslots_per_day, p_horizon, ch_noise_mean, chnoise_var):
#         self.location = location
#         self.year = year
#         self.REQ_TIMESLOTS_PER_DAY=timeslots_per_day
#         self.HFACTOR = 5*1E-3*240/self.REQ_TIMESLOTS_PER_DAY
#         self.DFACTOR = 5*1E-3*240/self.REQ_TIMESLOTS_PER_DAY
#         self.MIN_BATT = 0.1
#         self.MIN_DC = 0.1
#         self.p_horizon = p_horizon # prediction horizons
#         self.chnoise_mean = ch_noise_mean
#         self.chnoise_var = chnoise_var


#     def reset(self):
#         # Characterize the harvester
#         self.READINGS_PER_DAY = 24       

#         self.env_harvester = csv_solar_harvester(location=self.location,
#                                                  year=self.year,
#                                                  READINGS_PER_DAY = self.READINGS_PER_DAY,
#                                                  SMAX=4.0, # Max GSR
#                                                  HENERGY_NOISE=0.0, # henergy artifical noise
#                                                  NORMALIZED_HMIN_THRES=1E-5, # henergy cutoff
#                                                  REQ_TIMESLOTS_PER_DAY=self.REQ_TIMESLOTS_PER_DAY, # no. of timeslots per day
#                                                  PREDICTION_HORIZON=self.p_horizon, # lookahead horizon to predict energy
#                                                  PREDICTION_NOISE=0.0) # preidction noise
#         self.env_timeslot_values = self.env_harvester.time_slots
#         self.ENV_LIFETIME = self.env_harvester.no_of_days
        
#         # Characterize the battery
#         self.BINIT = 0.7
#         self.BEFF  = 1.0
#         self.env_battery = battery(self.BINIT,self.BEFF)

# #         # Characterize request generator
# #         hstream = np.array(self.env_harvester.henergy_stream)
# #         if self.hmean is None:
# #             self.hmean = hstream.mean()
# #         self.req_stream =  request_gen(self.rq_type,
# #                                        htrace=hstream,
# #                                        hmean=self.hmean,
# #                                        mindc=self.MIN_DC,
# #                                        timesteps_per_day=self.REQ_TIMESLOTS_PER_DAY,
# #                                        offset=self.offset)
       
#         # Data logging variables
#         self.env_log = [] # record all values in the environment
#         self.action_log = [] # record all actions sent to the environment
#         self.eno_log = []
#         self.tx_log = [] # [tx_dc,rx_dBm,snr,tx_prob,success]
#         self.batt_slice = deque(maxlen=self.REQ_TIMESLOTS_PER_DAY*10)
        
#         # Observation variables
#         self.time_obs = None
#         self.henergy_obs = None
#         self.penergy_obs = None
#         self.benergy_obs = None
#         self.menergy_obs = None
# #         self.req_obs = None        
#         self.chnoise_obs = None

        
#         # Environment Flags
#         self.RECOVERY_MODE = False # battery is recovering to BINIT from complete discharge & node is suspended
        
#         # Get observation
#         self.time_obs, self.henergy_obs, self.penergy_obs, DAY_END, HARVESTER_END = self.env_harvester.step()
#         self.benergy_obs = self.env_battery.get_batt_state()
#         self.eno_log.append(self.benergy_obs)
#         self.batt_slice.append(self.benergy_obs)
#         self.menergy_obs = np.mean(self.batt_slice)
# #         self.req_obs = self.req_stream[int(self.env_harvester.global_time)]
#         self.chnoise_obs = np.random.normal(loc=self.chnoise_mean, 
#                                             scale=self.chnoise_var)
#         self.obs = (self.time_obs/self.READINGS_PER_DAY,
#                     self.henergy_obs,
#                     self.penergy_obs,
#                     self.benergy_obs,
#                     self.menergy_obs,
# #                     self.req_obs
#                     self.chnoise_obs)
        
#         log_data = self.obs
#         self.env_log.append(log_data)
#         return np.array(self.obs)
    
#     def action2txdc(self,action):
#         # action=(0,1) --> tx_dc=(0.1,1)
#         tx_dc = 0.1 + 0.9*action
#         return tx_dc
    
#     def txdc2rxpwr(self,tx_dc):
#         e_pwr = 46.02 + (tx_dc-0.1)*(125.97-46.02)/(1.0-0.1)
#         ant_pwr = 12.34763 - 5265337*np.exp(-3.09181*np.log(e_pwr)) # in dBm
#         return ant_pwr
    
#     def snr2txprob(self,snr):
#          # rayleigh
#         snr_lin = 10**(snr/10)
#         ber = 0.5*(1-(snr_lin/(1+snr_lin))**0.5)

#         N = 100 * (8*1024) # 100 KB packet
#         tx_success = (1- ber)**N
#         return tx_success
    
#     def step(self, action):
#         # Is node already in recovery mode?
#         if self.RECOVERY_MODE:
#             self.recovery_action()
#             reward = self.reward(action=-1)
#         else:
#             ACTION_VALID = self.verify_action(action)
#             if ACTION_VALID:
#                 self.execute_action(action)
#                 reward = self.reward(action)
#             else:
#                 self.recovery_action()
#                 reward = self.reward(action=-1)
        
#         next_obs, done = self.next_obs()
#         info = {}
        
#         # Did the node enter recovery mode after executing action even though the action was valid?
#         # If the action was valid but resulted in entering recovery mode, then
#         # reward should be negative.
#         if self.RECOVERY_MODE: 
#             reward = self.reward(action=-1)        
        
#         return np.array(next_obs), reward, done, info
    
#     def verify_action(self, action): # check if actions are valid
#         assert self.RECOVERY_MODE == False, "Action does not need to be verified in recovery mode"
#         tx_dc = self.action2txdc(action)
#         surplus_energy = (self.henergy_obs*self.HFACTOR - (tx_dc)*self.DFACTOR)
#         if (-surplus_energy) < self.benergy_obs: # if there is sufficient energy in the battery to extract
#             return True # valid action
#         else:
#             self.RECOVERY_MODE = True # switch to recovery mode
#             return False # invalid action
        
#     def recovery_action(self):
#         assert self.RECOVERY_MODE == True, "Node is not in recovery mode"
# #         charge_qty =self.henergy_obs*self.HFACTOR
#         charge_qty = self.BINIT/2
#         self.env_battery.charge(charge_qty)
#         self.batt_slice.clear()
#         self.eno_log.append(charge_qty)
#         self.action_log.append(-1)
    
#     def execute_action(self, action): 
#         assert self.RECOVERY_MODE==False, "Node is in recovery mode. Cannot execute action"

#         tx_dc = self.action2txdc(action)
#         surplus_energy = (self.henergy_obs*self.HFACTOR - (tx_dc)*self.DFACTOR)
        
#         if surplus_energy >= 0:
#             self.env_battery.charge(surplus_energy)
#         else:
#             self.env_battery.discharge(surplus_energy)
        
#         self.eno_log.append(surplus_energy)
#         self.action_log.append(action)      
        

#     def next_obs(self): # update all observations
#         self.time_obs, self.henergy_obs, self.penergy_obs, DAY_END, HARVESTER_END = self.env_harvester.step()

#         if not HARVESTER_END:
#             self.chnoise_obs = np.random.normal(loc=self.chnoise_mean, 
#                                             scale=self.chnoise_var)
# #             self.req_obs = self.req_stream[int(self.env_harvester.global_time)]
#             self.benergy_obs = self.env_battery.get_batt_state() # updated battery observation
#             self.batt_slice.append(self.benergy_obs)
#             self.menergy_obs = np.mean(self.batt_slice)
            
#             if self.benergy_obs < self.MIN_BATT: # Is battery less than a threshold?
#                 self.RECOVERY_MODE = True
            
#             if self.RECOVERY_MODE:
#                 if self.benergy_obs > self.BINIT: 
#                     self.RECOVERY_MODE = False # snap out of recovery mode
#                 else:
#                     self.RECOVERY_MODE = True # remain in recovery mode
# #         else:
# #             self.req_obs = -1
# #             self.chnoise_obs = -1
# #             self.action_log.append(-1)

#         self.obs = (self.time_obs/self.READINGS_PER_DAY,
#                     self.henergy_obs,
#                     self.penergy_obs,
#                     self.benergy_obs,
#                     self.menergy_obs,
# #                     self.req_obs
#                     self.chnoise_obs)
        
#         log_data = self.obs
#         self.env_log.append(log_data)

#         done = HARVESTER_END
#         return self.obs, done
    
#     def reward(self,action): # reward based on utility
#         if action < 0:
#             self.tx_log.append([-1,-200,-200,-1,-1])
#             return -10_000.0
#         else:
#             tx_dc = self.action2txdc(action)
#             rx_dBm = self.txdc2rxpwr(tx_dc)
#             snr = rx_dBm - self.chnoise_obs
#             tx_prob = self.snr2txprob(snr)
#             success = np.random.binomial(n=1, p=tx_prob)
#             self.tx_log.append([tx_dc,rx_dBm,snr,tx_prob,success])
#             return success
#     # End of tx
# ########################################################

# class cnfrm(gym.Env):
#     """An ambient environment simulator for OpenAI gym."""
#     metadata = {'render.modes': ['human']}
    
#     def __init__(self):
#         super(cnfrm, self).__init__()
        
#         # Actions = energy_consumed
#         self.action_space = spaces.Box(low=0,
#                                        high=1,
#                                        shape=(1,))

#         # Observation = [time, h_energy, p_energy, b_energy, m_energy, req]
#         self.observation_space = spaces.Box(low=0, 
#                                             high=1, 
#                                             shape=(6,)) #<<<<<<<<<
        
        
#     def set_env(self,location, year, timeslots_per_day, req_type, offset, p_horizon, hmean=None):
#         self.location = location
#         self.year = year
#         self.REQ_TIMESLOTS_PER_DAY=timeslots_per_day
#         self.HFACTOR = 5*1E-3*240/self.REQ_TIMESLOTS_PER_DAY
#         self.DFACTOR = 5*1E-3*240/self.REQ_TIMESLOTS_PER_DAY
#         self.rq_type = req_type
#         self.offset = int(offset) # offset for request generator [when using random_day]
#         self.MIN_BATT = 0.1
#         self.MIN_DC = 0.1
#         self.p_horizon = p_horizon # prediction horizons
#         self.hmean = hmean # mean henergy if given


#     def reset(self):
#         # Characterize the harvester
#         self.READINGS_PER_DAY = 24       

#         self.env_harvester = csv_solar_harvester(location=self.location,
#                                                  year=self.year,
#                                                  READINGS_PER_DAY = self.READINGS_PER_DAY,
#                                                  SMAX=4.0, # Max GSR
#                                                  HENERGY_NOISE=0.0, # henergy artifical noise
#                                                  NORMALIZED_HMIN_THRES=1E-5, # henergy cutoff
#                                                  REQ_TIMESLOTS_PER_DAY=self.REQ_TIMESLOTS_PER_DAY, # no. of timeslots per day
#                                                  PREDICTION_HORIZON=self.p_horizon, # lookahead horizon to predict energy
#                                                  PREDICTION_NOISE=0.0) # preidction noise
#         self.env_timeslot_values = self.env_harvester.time_slots
#         self.ENV_LIFETIME = self.env_harvester.no_of_days
        
#         # Characterize the battery
#         self.BINIT = 0.7
#         self.BEFF  = 1.0
#         self.env_battery = battery(self.BINIT,self.BEFF)

#         # Characterize request generator
#         hstream = np.array(self.env_harvester.henergy_stream)
#         if self.hmean is None:
#             self.hmean = hstream.mean()
#         self.req_stream =  request_gen(self.rq_type,
#                                        htrace=hstream,
#                                        hmean=self.hmean,
#                                        mindc=self.MIN_DC,
#                                        timesteps_per_day=self.REQ_TIMESLOTS_PER_DAY,
#                                        offset=self.offset)
       
#         # Data logging variables
#         self.env_log = [] # record all values in the environment
#         self.action_log = [] # record all actions sent to the environment
#         self.eno_log = []
#         self.sense_dc_log = []
#         self.batt_slice = deque(maxlen=self.REQ_TIMESLOTS_PER_DAY*10)
        
#         # Observation variables
#         self.time_obs = None
#         self.henergy_obs = None
#         self.penergy_obs = None
#         self.benergy_obs = None
#         self.menergy_obs = None
#         self.req_obs = None
        
#         # Environment Flags
#         self.RECOVERY_MODE = False # battery is recovering to BINIT from complete discharge & node is suspended
        
#         # Get observation
#         self.time_obs, self.henergy_obs, self.penergy_obs, DAY_END, HARVESTER_END = self.env_harvester.step()
#         self.benergy_obs = self.env_battery.get_batt_state()
#         self.eno_log.append(self.benergy_obs)
#         self.batt_slice.append(self.benergy_obs)
#         self.menergy_obs = np.mean(self.batt_slice)
#         self.req_obs = self.req_stream[int(self.env_harvester.global_time)]
#         self.obs = (self.time_obs/self.READINGS_PER_DAY,
#                     self.henergy_obs,
#                     self.penergy_obs,
#                     self.benergy_obs,
#                     self.menergy_obs,
#                     self.req_obs)
#         log_data = self.obs
#         self.env_log.append(log_data)
#         return np.array(self.obs)
    
#     def action2sensedc(self,action):
#         if action < 0:
#             sense_dc = 0
#         else:
#             sense_dc = max(self.MIN_DC,action*self.req_obs)
#         return sense_dc    
    
#     def step(self, action):
#         # Is node already in recovery mode?
#         if self.RECOVERY_MODE: 
#             self.recovery_action()
#             reward = self.reward(action=-1)
#         else:
#             ACTION_VALID = self.verify_action(action)
#             if ACTION_VALID:
#                 self.execute_action(action)
#                 reward = self.reward(action)
#             else:
#                 self.recovery_action()
#                 reward = self.reward(action=-1)
        
#         next_obs, done = self.next_obs()
#         info = {}
        
#         # Did the node enter recovery mode after executing action even though the action was valid?
#         # If the action was valid but resulted in entering recovery mode, then
#         # reward should be negative.
#         if self.RECOVERY_MODE: 
#             reward = self.reward(action=-1)
#         return np.array(next_obs), reward, done, info
    
#     def verify_action(self, action): # check if actions are valid
#         assert self.RECOVERY_MODE == False, "Action does not need to be verified in recovery mode"
#         sense_dc = self.action2sensedc(action)
        
#         surplus_energy = (self.henergy_obs*self.HFACTOR - (sense_dc)*self.DFACTOR)
#         if (-surplus_energy) < self.benergy_obs: # if there is sufficient energy in the battery to extract
#             return True # valid action
#         else:
#             self.RECOVERY_MODE = True # switch to recovery mode
#             return False # invalid action
        
#     def recovery_action(self):
#         assert self.RECOVERY_MODE == True, "Node is not in recovery mode"
# #         charge_qty =self.henergy_obs*self.HFACTOR
#         charge_qty = self.BINIT/2
#         self.env_battery.charge(charge_qty)
#         self.batt_slice.clear()
#         self.eno_log.append(charge_qty)
#         self.action_log.append(-1)
#         self.sense_dc_log.append(0)
        
#     def execute_action(self, action): 
#         assert self.RECOVERY_MODE==False, "Node is in recovery mode. Cannot execute action"

#         sense_dc = self.action2sensedc(action)
#         surplus_energy = (self.henergy_obs*self.HFACTOR - (sense_dc)*self.DFACTOR)
        
#         if surplus_energy >= 0:
#             self.env_battery.charge(surplus_energy)
#         else:
#             self.env_battery.discharge(surplus_energy)
        
#         self.eno_log.append(surplus_energy)
#         self.action_log.append(action)      
#         self.sense_dc_log.append(sense_dc)

#     def next_obs(self): # update all observations

#         self.time_obs, self.henergy_obs, self.penergy_obs, DAY_END, HARVESTER_END = self.env_harvester.step()

#         if not HARVESTER_END:
#             self.req_obs = self.req_stream[int(self.env_harvester.global_time)]
#             self.benergy_obs = self.env_battery.get_batt_state() # updated battery observation
#             self.batt_slice.append(self.benergy_obs)
#             self.menergy_obs = np.mean(self.batt_slice)
            
#             if self.benergy_obs < self.MIN_BATT: # Is battery less than a threshold?
#                 self.RECOVERY_MODE = True
            
#             if self.RECOVERY_MODE:
#                 if self.benergy_obs > self.BINIT: 
#                     self.RECOVERY_MODE = False # snap out of recovery mode
#                 else:
#                     self.RECOVERY_MODE = True # remain in recovery mode
# #         else:
# #             self.req_obs = -1
# #             self.action_log.append(-1)
            
#         self.obs = (self.time_obs/self.READINGS_PER_DAY,
#                     self.henergy_obs,
#                     self.penergy_obs,
#                     self.benergy_obs,
#                     self.menergy_obs,
#                     self.req_obs)
#         log_data = self.obs
#         self.env_log.append(log_data)

#         done = HARVESTER_END
#         return self.obs, done
    
#     def reward(self,action): # reward based on utility
#         if action < 0:
#             return -10_000.0
#         else:
#             sense_dc = self.action2sensedc(action)
#             return min(1.0,sense_dc/self.req_obs)
#     # End of cnfrm
# ########################################################
# class rawdc(cnfrm):
#     #action = (0,1) --> sense_dc = (0.1,0.5)
#     def action2sensedc(self,action):
#         if action < 0:
#             sense_dc = 0
#         else:
#             sense_dc = self.MIN_DC + action*0.4
#         return sense_dc
    
#     def verify_action(self, action): # check if actions are valid
#         assert self.RECOVERY_MODE == False, "Action does not need to be verified in recovery mode"
#         sense_dc = self.action2sensedc(action)
        
#         surplus_energy = (self.henergy_obs*self.HFACTOR - (sense_dc)*self.DFACTOR)
#         if (-surplus_energy) < self.benergy_obs: # if there is sufficient energy in the battery to extract
#             return True # valid action
#         else:
#             self.RECOVERY_MODE = True # switch to recovery mode
#             return False # invalid action
        
#     def execute_action(self, action): 
#         assert self.RECOVERY_MODE==False, "Node is in recovery mode. Cannot execute action"

#         sense_dc = self.action2sensedc(action)
#         surplus_energy = (self.henergy_obs*self.HFACTOR - (sense_dc)*self.DFACTOR)
        
#         if surplus_energy >= 0:
#             self.env_battery.charge(surplus_energy)
#         else:
#             self.env_battery.discharge(surplus_energy)
        
#         self.eno_log.append(surplus_energy)
#         self.action_log.append(action)
#         self.sense_dc_log.append(sense_dc)
    
#     def reward(self,action): # reward based on utility
#         if action < 0:
#             return -10_000.0
#         else:
#             sense_dc = self.action2sensedc(action)
#             return min(1.0,sense_dc/self.req_obs)
#     # End of rawdc
# ########################################################