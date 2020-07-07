import numpy as np


def best_rq_gen(htrace, hmean, mindc):
# Let R = a*H + mindc --- <1>    such that E[R] = E[H]

# Given hstream -> H, 
#       hmean -> E[H] and,
#       mindc

# IMPORTANT!!!
# mean(H) is NOT NECESSARILY EQUAL to hmean=E[H]
# For e.g., H may correspond to a stream of one year
# while hmean=E[H] may correspond to an average of ten years

# Return rq_stream -> R

# We need to find 'a' such that E[R] = E[H]

# Taking expecation of eq <1>,
#     E[R] = E[a*H + mindc]
# or, E[H] = a*E[H] + mindc  (because, E[R] = E[H])
# or, hmean = a*hmean + mindc
# or, hmean - mindc = a*hmean
# or, (hmean - mindc)/hmean = a
    # Request has the same profile as the harvested energy
    best_rq_trace = mindc + htrace * (hmean-mindc)/(hmean)
    return best_rq_trace
##############################################################

def const_rq_gen(htrace, hmean, mindc):
    # Requests are constant
    const_rq_trace = np.ones_like(htrace)*hmean
    return const_rq_trace

##############################################################

def random_rq_gen(htrace, hmean, mindc):
    # Request is a random timeslot
    random_rq_trace = best_rq_gen(htrace, hmean, mindc)
    np.random.shuffle(random_rq_trace)
    return random_rq_trace

##############################################################

def normal_rq_gen(htrace, hmean, mindc):
    # Request is a normally distributed with mean=hmean
    n_samples = len(htrace)
    amin, amax = 0, 1
    samples = np.zeros((0,))    # empty for now
    while samples.shape[0] < n_samples: 
        s = np.random.normal(loc=hmean,scale=0.1, size=(n_samples,))
        accepted = s[(s >= amin) & (s <= amax)]
        samples = np.concatenate((samples, accepted), axis=0)
    normal_rq_trace = samples[:n_samples]    # we probably got more than needed, so discard extra ones
    normal_rq_trace = mindc + normal_rq_trace * (hmean-mindc)/(hmean)
    return normal_rq_trace

##############################################################

def uniform_rq_gen(htrace, hmean, mindc):
    # Request is a uniformly distributed with mean=hmean
    dev = 0.05
    uniform_rq_trace = hmean + np.random.uniform(low=-1, high=1, size=len(htrace))*dev
    uniform_rq_trace[uniform_rq_trace<mindc]=mindc
    return uniform_rq_trace

##############################################################

def random_day_rq_gen(htrace, hmean, mindc, timesteps_per_day, offset):
    # Request corresponds to a random day profile shifted by an offset
    random_day_rq_trace = best_rq_gen(htrace, hmean, mindc)
    random_day_rq_trace = random_day_rq_trace.reshape(-1,timesteps_per_day)
    np.random.shuffle(random_day_rq_trace)
    random_day_rq_trace = np.roll(random_day_rq_trace, shift=offset)
    random_day_rq_trace = random_day_rq_trace.flatten()
    return random_day_rq_trace
##############################################################

def worst_rq_gen(htrace, hmean, mindc, timesteps_per_day):
    htrace = htrace.reshape(-1,timesteps_per_day)
    
    daily_henergy = htrace.sum(axis=1)
    asc_days = daily_henergy.argsort() # sort by ascending days

    asc_trace = []
    for day in asc_days:
        asc_trace.append(htrace[day])
    asc_trace = np.array(asc_trace) # new htrace where the days are ascending

    asc_trace = asc_trace.flatten()
    dsc_trace = asc_trace[::-1] # reverse the ascending order


    req_trace = dsc_trace * (hmean-mindc)/(hmean)
    req_trace = req_trace + mindc

    req_trace =  np.roll(req_trace, shift=int(timesteps_per_day/2)) # 12 hour offset
    req_trace = req_trace.flatten()
    
    return req_trace
##############################################################

def request_gen(rq_gen, htrace, hmean, mindc, timesteps_per_day, offset):
    if rq_gen == "best":
        rq_stream = best_rq_gen(htrace, hmean, mindc)
    elif rq_gen == "const":
        rq_stream = const_rq_gen(htrace, hmean, mindc)
    elif rq_gen == "random":
        rq_stream = random_rq_gen(htrace, hmean, mindc)
    elif rq_gen == "normal":
        rq_stream = normal_rq_gen(htrace, hmean, mindc)
    elif rq_gen == "uniform":
        rq_stream = uniform_rq_gen(htrace, hmean, mindc)
    elif rq_gen == "random_day":
        rq_stream = random_day_rq_gen(htrace, hmean, mindc, timesteps_per_day, offset)
    elif rq_gen == "worst":
        rq_stream = worst_rq_gen(htrace, hmean, mindc, timesteps_per_day)
    else:
        print("Bad Request")
        return 0
    return rq_stream
