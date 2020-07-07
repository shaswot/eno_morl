import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pathlib import Path

########################################################
# Read GSR values from CSV and convert to henergy
########################################################
def csv2gsr(location,year,SMAX):
    # Get data from CSV
    #####################
    # solar_data/CSV files contain the values of GSR (Global Solar Radiation in MegaJoules per meters squared per hour)
    # weather_data/CSV files contain the weather summary from 06:00 to 18:00 and 18:00 to 06:00+1

    sfile = Path.cwd() / 'solar_data' / location / (str(year) + '.csv')

    # skiprows=4 to remove unnecessary title texts
    # usecols=4 to read only the Global Solar Radiation (GSR) values
    solar_radiation = pd.read_csv(sfile, skiprows=4, encoding='shift_jisx0213', usecols=[4])

    # convert dataframe to numpy array
    solar_radiation = solar_radiation.values

    # convert missing data in CSV files to zero
    solar_radiation[np.isnan(solar_radiation)] = 0

    # reshape solar_radiation into no_of_daysxREADINGS_PER_DAY array
    solar_radiation = solar_radiation.reshape(1,-1).flatten() # 1-D array
    henergy = solar_radiation/SMAX # normalize
    return henergy
########################################################

########################################################
# Class for Energy Prediction
########################################################
# rolling average predictor
# the beginning and ends are padded by rolling over the array
class rolling_predictor(object):
        def __init__(self, PREDICTION_HORIZON, PREDICTION_NOISE):
            self.PREDICTION_HORIZON = PREDICTION_HORIZON
            self.PREDICTION_NOISE = PREDICTION_NOISE
            
        def get_prediction(self, stream):
            PREDICTION_HORIZON = self.PREDICTION_HORIZON
            PREDICTION_NOISE = self.PREDICTION_NOISE
            
            # extend stream by PREDICTION_HORIZON
            stream_length = len(stream)
            padded_stream = np.concatenate((stream[-PREDICTION_HORIZON:],
                                stream,
                                stream[:PREDICTION_HORIZON]))
            
            # this uses a simple rolling average over a given prediction horizon
            weights = np.ones(PREDICTION_HORIZON) / PREDICTION_HORIZON
            prediction = np.convolve(padded_stream, weights, mode='valid')
            prediction = prediction[:stream_length]
            prediction_noise = np.random.normal(0,PREDICTION_NOISE,
                                                size=prediction.shape) # add normal noise
            prediction += prediction_noise
            return prediction
# End of rolling_predictor
########################################################
    
########################################################
# Class for solar energy harvester
########################################################
class csv_solar_harvester(object):
    def __init__(self, 
                 location='tokyo',
                 year=1995,
                 READINGS_PER_DAY = 24,
                 SMAX=4.0, # Max GSR
                 HENERGY_NOISE=0.0, # henergy artifical noise
                 NORMALIZED_HMIN_THRES=1E-5, # henergy cutoff
                 REQ_TIMESLOTS_PER_DAY=240, # no. of timeslots per day
                 PREDICTION_HORIZON=240*10, # lookahead horizon to predict energy
                 PREDICTION_NOISE=0.0): # preidction noise
        
      
        # Initialize variables
        self.day = 0
        self.global_time = 0
        self.henergy_stream = []
        self.penergy_stream = []
        
        henergy = csv2gsr(location,year,SMAX)
        henergy = henergy.reshape(-1, READINGS_PER_DAY)
        # Fix time resolution
        ######################
        self.no_of_days = int(henergy.shape[0]/READINGS_PER_DAY) # the number of days worth of data in the csv file

        # Interpolate the harvested energy data to new time resolution
        ###############################################################
        itp_henergy = interp1d(np.arange(READINGS_PER_DAY),
                               henergy, 
                               kind='quadratic') # quadratic interpolation
        
        new_time_slots =  np.linspace(0, READINGS_PER_DAY-1, REQ_TIMESLOTS_PER_DAY) # new time index
        high_res_henergy = itp_henergy(new_time_slots) # interpolate and fill in the intermediate values
        self.time_slots = new_time_slots # to access from object instance
    
        # Add noise to henergy data
        ###########################
        henergy_noise = np.random.normal(1,HENERGY_NOISE,size=high_res_henergy.shape) # add some noise to it
        high_res_henergy *= henergy_noise # we multiply so that zero energy times slots remain with zero energy
        high_res_henergy[high_res_henergy<NORMALIZED_HMIN_THRES]=0 # clipping threshold
        high_res_henergy[high_res_henergy>1]=1
        self.henergy_stream = high_res_henergy.reshape(1,-1).flatten().tolist() # flattened list of henergy

        # Get energy prediction
        #######################
        self.predictor = rolling_predictor(PREDICTION_HORIZON, PREDICTION_NOISE)
        self.penergy_stream = self.predictor.get_prediction(self.henergy_stream)
        self.penergy_stream[self.penergy_stream<NORMALIZED_HMIN_THRES]=0 # clipping threshold
        self.penergy_stream = self.penergy_stream.tolist()
        
    
    # step through each time step and output time, henergy, penergy
    def step(self):
        if self.global_time < len(self.henergy_stream):
            HARVESTER_END = False
            DAY_END = False
            time = self.time_slots[self.global_time%len(self.time_slots)]
            henergy = self.henergy_stream[self.global_time]
            penergy = self.penergy_stream[self.global_time]
            
            self.global_time += 1 # new time
            if self.global_time%len(self.time_slots)==0: #new day
#                 print("NEW DAY STARTS:", self.day)
                self.day += 1
                DAY_END = True # this means the next time step is for a new day
            if self.global_time >= len(self.henergy_stream):
#                 print("END OF YEAR. DAY:",self.day)
                HARVESTER_END = True
            return time, henergy, penergy, DAY_END, HARVESTER_END
        else:
            print("YEAR ALREADY ENDED")
            DAY_END = True
            HARVESTER_END = True
            return -1,-1,-1,NEW_DAY, HARVESTER_END
# End of csv_solar_harvester
########################################################