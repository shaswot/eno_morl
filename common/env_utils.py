import numpy as np
import matplotlib.pyplot as plt
########################################################
# Class for battery
########################################################
class battery(object):
    def __init__(self, BINIT, BEFF):
        self.batt = BINIT # battery starts with 70% charge
        self.BEFF = BEFF
        
        assert 0 <= self.batt <= 1, 'Incorrect Battery Initialization'
        assert 0 < self.batt <= 1 , 'Incorrect Battery Efficiency'
    def charge(self, energy):
        assert energy >= 0, "Charging Energy should be >= 0"
        self.batt += energy*self.BEFF
        self.batt = np.clip(self.batt,0,1)
        
    def discharge(self, energy): #energy will be in negative
        assert energy < 0, "Discharging Energy should be < 0"
        self.batt += energy
        self.batt =  np.clip(self.batt,0,1)
        
    def get_batt_state(self):
        return np.clip(self.batt,0,1).item()
# End of class battery
########################################################
def tx_prob(tx_power,ch_gain):
    k = 2
    diff = tx_power-ch_gain
    offset = 1
    x = diff+offset
    y = (1 + np.exp(-k*(1+offset))) / (1 + np.exp(-k*x)) 
    y = y*2-1.1
    y = np.clip(y,0,1)
    
    return np.where(tx_power<0.01,0,y)
########################################################
def sorl_plot(run_log, timeslots_per_day, START_DAY=0, NO_OF_DAY_TO_PLOT = 500):
    
   # Get Environment Log
    ################################################################    

    reward_rec = run_log['reward_rec']
    sense_dc_log = run_log['sense_dc_log']
    env_log = run_log['env_log']
    
    henergy_obs_rec=env_log[:,1]
    penergy_obs_rec=env_log[:,2]
    benergy_obs_rec=env_log[:,3]
    menergy_obs_rec=env_log[:,4]
    req_obs_rec=env_log[:,5]
    


    NO_OF_TIMESLOTS_PER_DAY = timeslots_per_day
    NO_OF_DAY_TO_PLOT = int(min(NO_OF_DAY_TO_PLOT, len(henergy_obs_rec)/NO_OF_TIMESLOTS_PER_DAY))
    END_DAY = START_DAY + NO_OF_DAY_TO_PLOT

    start_index = START_DAY*NO_OF_TIMESLOTS_PER_DAY
    end_index = END_DAY*NO_OF_TIMESLOTS_PER_DAY
    
    
    # Draw figure
    ##############
    fig, axs = plt.subplots(nrows=1,
                            ncols=1,
                            figsize=[20,4],
                            sharex=True)

    sense_dc_ax  = axs

    sense_dc_ax.grid(which='major', axis='x', linestyle='--')
    sense_dc_ax.set_ylim(-0.1,1.1)

    
    sense_dc_ax.plot(sense_dc_log[start_index:end_index], 
                     color='tab:blue', alpha=0.7,linewidth=1.0, label="sense_dc")
    sense_dc_ax.plot(req_obs_rec[start_index:end_index], 
                     color='tab:orange',linestyle='--',linewidth=1.0, label="req_dc")
    sense_dc_ax.plot(benergy_obs_rec[start_index:end_index], 
                     color='tab:red',linewidth=1.0, label="battery")
    sense_dc_ax.plot(menergy_obs_rec[start_index:end_index], 
                     color='tab:red',linestyle='--',linewidth=1.0, alpha=0.5,label="batt10d")
    sense_dc_ax.plot(reward_rec[start_index:end_index],
                     color='tab:purple', alpha=0.7,linewidth=1.0, label="reward")
#     sense_dc_ax.plot(henergy_obs_rec[start_index:end_index],
#                      color='k',linewidth=0.25,alpha=0.5, label="henergy")
#     sense_dc_ax.plot(penergy_obs_rec[start_index:end_index],
#                      color='k',linestyle='--',linewidth=0.25, label="penergy")

    
    sense_dc_ax.set_xlim([0,NO_OF_TIMESLOTS_PER_DAY*NO_OF_DAY_TO_PLOT])
    xtick_resolution = max(1,int(NO_OF_DAY_TO_PLOT/10))
    sense_dc_ax.set_xticks(np.arange(start=0,
                                      stop=NO_OF_TIMESLOTS_PER_DAY*(NO_OF_DAY_TO_PLOT+1),
                                      step=NO_OF_TIMESLOTS_PER_DAY*xtick_resolution))
    sense_dc_ax.set_xticklabels(np.arange(start=START_DAY,
                                           stop=END_DAY+1,
                                           step=xtick_resolution))
    sense_dc_ax.legend(loc="lower left",
                       ncol=7,
                       bbox_to_anchor=(0,1.0,1,1))
    
    sense_dc_ax.set_xlabel("Days")
    plt.show()
# End of sorl_plot
########################################################
def task3_plot(run_log, timeslots_per_day, START_DAY=0, NO_OF_DAYS_TO_PLOT = 500):
    
    # Get Environment Log
    ################################################################    

    sense_dc_log = run_log['sense_dc_log']
    tx_dc_log = run_log['tx_dc_log']

    env_log = run_log['env_log']
    henergy_obs_rec=env_log[:,1]
    penergy_obs_rec=env_log[:,2]
    benergy_obs_rec=env_log[:,3]
    menergy_obs_rec=env_log[:,4]
    req_obs_rec=env_log[:,5]
    req_obs2_rec=env_log[:,6]

    sense_reward_log = run_log['sense_reward_log']
    tx_reward_log = run_log['tx_reward_log']
    enp_reward_log = run_log['enp_reward_log']


    NO_OF_TIMESLOTS_PER_DAY = timeslots_per_day
    NO_OF_DATA_DAYS = int(len(henergy_obs_rec)/NO_OF_TIMESLOTS_PER_DAY)

    END_DAY = min(START_DAY + NO_OF_DAYS_TO_PLOT, NO_OF_DATA_DAYS)

    start_index = START_DAY*NO_OF_TIMESLOTS_PER_DAY
    end_index = END_DAY*NO_OF_TIMESLOTS_PER_DAY

    fig_width = 10
    fig_height = fig_width / 1.618

    # Draw figure
    ##############
    fig, axs = plt.subplots(nrows=3,
                            ncols=1,
                            figsize=[fig_width,fig_height],
                            sharex=True)


    enp_dc_ax    = axs[0]
    sense_dc_ax  = axs[1]
    tx_dc_ax     = axs[2]

    x_axis_values = np.arange(start_index,end_index)
    xtick_resolution = max(1,int((END_DAY-START_DAY)/10))

    ###########################################################################################################
    enp_dc_ax.plot(x_axis_values, 
                   menergy_obs_rec[start_index:end_index], 
                   color='tab:red', alpha=1.0,linewidth=1.0, label="batt")

    enp_dc_ax.plot(x_axis_values,
                   enp_reward_log[start_index:end_index], 
                   color='tab:green', alpha=1.0,linewidth=1.0, label="enp_reward")

    enp_dc_ax.set_ylim(0,1)
    enp_dc_ax.grid(which='major', axis='x', linestyle='--')

    enp_dc_ax.legend()
    ###########################################################################################################
    ###########################################################################################################
    sense_dc_ax.plot(x_axis_values,
                     sense_dc_log[start_index:end_index], 
                     color='tab:blue', alpha=0.7,linewidth=2.0, label="sense_dc")

    sense_dc_ax.fill_between(x_axis_values,
                             req_obs_rec[start_index:end_index],0, 
                             color='tab:orange',linestyle='--',linewidth=1.0, alpha=0.2, label="req_dc")

    sense_reward_ax = sense_dc_ax.twinx()
    sense_reward_ax.plot(x_axis_values,
                         sense_reward_log[start_index:end_index],
                         color='tab:purple', alpha=0.7,linewidth=1.0, label="sense_dc_reward")
    sense_reward_ax.set_ylim(0.1,1.1)
    sense_reward_ax.set_ylabel("sense_reward", color='tab:purple')

    sense_dc_ax.set_ylim(0,0.5)
    sense_dc_ax.grid(which='major', axis='x', linestyle='--')

    sense_dc_ax.set_ylabel("sense_dc", color='tab:blue')
    ###########################################################################################################
    ###########################################################################################################
    tx_dc_ax.plot(x_axis_values,
                  tx_dc_log[start_index:end_index], 
                  color='tab:green', alpha=0.7,linewidth=2.0, label="tx_dc")

    tx_dc_ax.fill_between(x_axis_values,
                          req_obs2_rec[start_index:end_index],0,
                          color='tab:pink',linestyle='--',linewidth=1.0, alpha=0.2, label="req_dc2")

    tx_reward_ax = tx_dc_ax.twinx()
    tx_reward_ax.plot(x_axis_values,
                      tx_reward_log[start_index:end_index],
                      color='tab:olive', alpha=0.7,linewidth=1.0, label="tx_dc_reward")
    tx_reward_ax.set_ylim(0.1,1.1)
    tx_reward_ax.set_ylabel("tx_reward", color='tab:olive')

    tx_dc_ax.set_ylim(0,0.5)
    tx_dc_ax.grid(which='major', axis='x', linestyle='--')
    tx_dc_ax.set_xlim([start_index,end_index])
    tx_dc_ax.set_xticks(np.arange(start=start_index,
                                    stop=end_index+1,
                                    step=NO_OF_TIMESLOTS_PER_DAY*xtick_resolution))
    tx_dc_ax.set_xticklabels(np.arange(start=START_DAY,
                                           stop=END_DAY+1,
                                           step=xtick_resolution))

    tx_dc_ax.set_ylabel("tx_dc", color='tab:green')
    plt.show()
# End of task3_plot
########################################################

# def cnfrm_morl_plot(run_log, timeslots_per_day, START_DAY=0, NO_OF_DAY_TO_PLOT = 500):
    
#    # Get Environment Log
#     ################################################################    

#     reward_rec = run_log['reward_rec']
    
#     sense_reward_log = run_log['sense_reward_log'] * run_log['preference']
#     enp_reward_log = run_log['enp_reward_log'] * (1-run_log['preference'])
    
#     sense_dc_log = run_log['sense_dc_log']
#     env_log = run_log['env_log']

#     henergy_obs_rec=env_log[:,1]
#     penergy_obs_rec=env_log[:,2]
#     benergy_obs_rec=env_log[:,3]
#     menergy_obs_rec=env_log[:,4]
#     req_obs_rec=env_log[:,5]
    


#     NO_OF_TIMESLOTS_PER_DAY = timeslots_per_day
#     NO_OF_DAY_TO_PLOT = int(min(NO_OF_DAY_TO_PLOT, len(henergy_obs_rec)/NO_OF_TIMESLOTS_PER_DAY))
#     END_DAY = START_DAY + NO_OF_DAY_TO_PLOT

#     start_index = START_DAY*NO_OF_TIMESLOTS_PER_DAY
#     end_index = END_DAY*NO_OF_TIMESLOTS_PER_DAY
    
    
#     # Draw figure
#     ##############
#     fig, axs = plt.subplots(nrows=2,
#                             ncols=1,
#                             figsize=[20,8],
#                             sharex=True)

#     sense_dc_ax  = axs[0]
#     rwd_ax = axs[1]

#     sense_dc_ax.grid(which='major', axis='x', linestyle='--')
#     sense_dc_ax.set_ylim(-0.1,1.1)

    
#     sense_dc_ax.plot(sense_dc_log[start_index:end_index], 
#                      color='tab:blue', alpha=0.7,linewidth=1.0, label="sense_dc")
#     sense_dc_ax.plot(req_obs_rec[start_index:end_index], 
#                      color='tab:orange',linestyle='--',linewidth=1.0, label="req_dc")
#     sense_dc_ax.plot(benergy_obs_rec[start_index:end_index], 
#                      color='tab:red',linewidth=1.0, label="battery")
#     sense_dc_ax.plot(menergy_obs_rec[start_index:end_index], 
#                      color='tab:red',linestyle='--',linewidth=1.0, alpha=0.5,label="batt10d")
#     sense_dc_ax.plot(reward_rec[start_index:end_index],
#                      color='tab:purple', alpha=0.7,linewidth=1.0, label="reward")
# #     sense_dc_ax.plot(henergy_obs_rec[start_index:end_index],
# #                      color='k',linewidth=0.25,alpha=0.5, label="henergy")
# #     sense_dc_ax.plot(penergy_obs_rec[start_index:end_index],
# #                      color='k',linestyle='--',linewidth=0.25, label="penergy")

    
#     sense_dc_ax.set_xlim([0,NO_OF_TIMESLOTS_PER_DAY*NO_OF_DAY_TO_PLOT])
#     xtick_resolution = max(1,int(NO_OF_DAY_TO_PLOT/10))
#     sense_dc_ax.set_xticks(np.arange(start=0,
#                                       stop=NO_OF_TIMESLOTS_PER_DAY*(NO_OF_DAY_TO_PLOT+1),
#                                       step=NO_OF_TIMESLOTS_PER_DAY*xtick_resolution))
#     sense_dc_ax.set_xticklabels(np.arange(start=START_DAY,
#                                            stop=END_DAY+1,
#                                            step=xtick_resolution))
#     sense_dc_ax.legend(loc="lower left",
#                        ncol=7,
#                        bbox_to_anchor=(0,1.0,1,1))
    
    
    
#     rwd_ax.stackplot(range(0,end_index-start_index),
#                     enp_reward_log[start_index:end_index],
#                     sense_reward_log[start_index:end_index],
#                     labels=['enp_reward','sense_reward'],
#                     colors=['tab:red','tab:blue'],
#                      alpha=0.8,
#                      baseline='zero')
#     rwd_ax.plot(reward_rec[start_index:end_index],
#                      color='tab:purple', alpha=0.7,linewidth=3.0, label="reward")
    
#     rwd_ax.legend(loc="lower left",
#                        ncol=7,
#                        bbox_to_anchor=(0,1.0,1,1))
#     rwd_ax.set_xlabel("Days")
#     plt.show()
# # End of cnfrm_morl_plot
# ########################################################
# def tx_plot(run_log, timeslots_per_day, START_DAY=0, NO_OF_DAY_TO_PLOT = 500):
#     # Get Environment Log
#     ################################################################ 
#     reward_rec = run_log['reward_rec']
#     tx_log = run_log['tx_log']
#     env_log = run_log['env_log']

#     henergy_obs_rec=env_log[:,1]
#     penergy_obs_rec=env_log[:,2]
#     benergy_obs_rec=env_log[:,3]
#     menergy_obs_rec=env_log[:,4]
#     chnoise_obs_rec=env_log[:,5]

#     tx_dc   = tx_log[:,0]
#     rx_dBm  = tx_log[:,1]
#     snr     = tx_log[:,2] #dB
#     tx_prob = tx_log[:,3]


#     NO_OF_TIMESLOTS_PER_DAY = timeslots_per_day
#     NO_OF_DAY_TO_PLOT = int(min(NO_OF_DAY_TO_PLOT, len(henergy_obs_rec)/NO_OF_TIMESLOTS_PER_DAY))
#     END_DAY = START_DAY + NO_OF_DAY_TO_PLOT

#     start_index = START_DAY*NO_OF_TIMESLOTS_PER_DAY
#     end_index = END_DAY*NO_OF_TIMESLOTS_PER_DAY


#     fig, axs = plt.subplots(nrows=2,
#                             ncols=1,
#                             figsize=[20,6],
#                             sharex=True)

#     batt_ax = axs[0]
#     snr_ax = axs[1]
#     snr_twin = snr_ax.twinx()

#     batt_ax.grid(which='major', axis='x', linestyle='--')
#     snr_ax.grid(which='major', axis='x', linestyle='--')

#     snr_ax.set_xlim([0,NO_OF_TIMESLOTS_PER_DAY*NO_OF_DAY_TO_PLOT])
#     xtick_resolution = max(1,int(NO_OF_DAY_TO_PLOT/10))
#     snr_ax.set_xticks(np.arange(start=0,
#                                 stop=NO_OF_TIMESLOTS_PER_DAY*(NO_OF_DAY_TO_PLOT+1),
#                                 step=NO_OF_TIMESLOTS_PER_DAY*xtick_resolution))
#     snr_ax.set_xticklabels(np.arange(start=START_DAY,
#                                      stop=END_DAY+1,
#                                      step=xtick_resolution))

#     snr_ax.set_xlabel("Days")



#     batt_ax.scatter(range(end_index-start_index),
#                     reward_rec[start_index:end_index],
#                     color='tab:olive', alpha=1.0, s=3, label="tx_success")
#     batt_ax.plot(benergy_obs_rec[start_index:end_index],
#                  color='tab:red',linewidth=1.0, label="battery")
#     batt_ax.plot(henergy_obs_rec[start_index:end_index],
#                  color='k',linewidth=0.25,alpha=0.5, label="henergy")
#     batt_ax.plot(tx_dc[start_index:end_index],
#                  color='tab:blue', alpha=0.7,linewidth=1.0, label="tx_dc")
#     batt_ax.plot(tx_prob[start_index:end_index],
#                  color='tab:purple', alpha=0.7,linewidth=1.0, label="tx_prob")
#     batt_ax.set_ylim(-0.1,1.1)
#     batt_ax.legend(loc="lower left",
#                            ncol=7,
#                            bbox_to_anchor=(0,1.0,1,1))


#     snr_ax.plot(tx_prob[start_index:end_index],
#                 color='tab:purple', alpha=0.5,linewidth=1.0, label="tx_prob")
#     snr_ax.set_ylim(0.5,1.1)

#     snr_ax.set_ylabel("tx_prob")

#     snr_twin.plot(rx_dBm[start_index:end_index], 
#                   color='tab:blue', alpha=0.7,linewidth=1.0, label="rx_dBm")
#     snr_twin.plot(chnoise_obs_rec[start_index:end_index], 
#                   color='tab:orange', alpha=0.7,linewidth=1.0, label="ch_noise")
#     snr_twin.set_ylabel("dBm")
#     snr_twin.set_ylim([-100,5])
#     snr_twin.legend(loc="lower left",
#                            ncol=7,
#                            bbox_to_anchor=(0,1.0,1,1))
#     plt.show()
# # End of tx_plot
# ########################################################
