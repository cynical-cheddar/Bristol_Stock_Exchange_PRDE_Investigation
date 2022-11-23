# Initial Setup:
# Import all the libraries we need

import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import random
from operator import itemgetter
from BSE import market_session
from sklearn.linear_model import Ridge
import itertools
from itertools import combinations, chain
import Mathf

# The next are helper functions that you will use later, if they don't make 
# much sense now, don't worry too much about it they will become clearer later:
colourList = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
# Use this to plot trades of a single experiment
def plot_trades(trial_id):
    prices_fname = trial_id + '_tape.csv'
    x = np.empty(0)
    y = np.empty(0)
    with open(prices_fname, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            time = float(row[1])
            price = float(row[2])
            x = np.append(x,time)
            y = np.append(y,price)

    plt.plot(x, y, 'x', color='black') 
    plt.show()

def lerp(a, b, f):

    return (a * (1.0 - f) + (b * f))


# Use this to plot trades of a single experiment
def plot_cumulative_profit(trial_id):
    print("not done")
    
# Use this to run an experiment n times and plot all trades
def n_runs_plot_trades(n, trial_id, start_time, end_time, traders_spec, order_sched):
    x = np.empty(0)
    y = np.empty(0)

    for i in range(n):
        trialId = trial_id + '_' + str(i)
        tdump = open(trialId + '_avg_balance.csv','w')

        market_session(trialId, start_time, end_time, traders_spec, order_sched, tdump, True, False)
        
        tdump.close()

        with open(trialId + '_tape.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                time = float(row[1])
                price = float(row[2])
                x = np.append(x,time)
                y = np.append(y,price)

    plt.plot(x, y, 'x', color='black');

# !!! Don't use on it's own   
def getorderprice(i, sched, n, mode):
    pmin = min(sched[0][0], sched[0][1])
    pmax = max(sched[0][0], sched[0][1])
    prange = pmax - pmin
    stepsize = prange / (n - 1)
    halfstep = round(stepsize / 2.0)

    if mode == 'fixed':
        orderprice = pmin + int(i * stepsize)
    elif mode == 'jittered':
        orderprice = pmin + int(i * stepsize) + random.randint(-halfstep, halfstep)
    elif mode == 'random':
        if len(sched) > 1:
            # more than one schedule: choose one equiprobably
            s = random.randint(0, len(sched) - 1)
            pmin = min(sched[s][0], sched[s][1])
            pmax = max(sched[s][0], sched[s][1])
        orderprice = random.randint(pmin, pmax)
    return orderprice    

# !!! Don't use on it's own
def make_supply_demand_plot(bids, asks):
    # total volume up to current order
    volS = 0
    volB = 0

    fig, ax = plt.subplots()
    plt.ylabel('Price')
    plt.xlabel('Quantity')
    
    pr = 0
    for b in bids:
        if pr != 0:
            # vertical line
            ax.plot([volB,volB], [pr,b], 'r-')
        # horizontal lines
        line, = ax.plot([volB,volB+1], [b,b], 'r-')
        volB += 1
        pr = b
    if bids:
        line.set_label('Demand')
        
    pr = 0
    for s in asks:
        if pr != 0:
            # vertical line
            ax.plot([volS,volS], [pr,s], 'b-')
        # horizontal lines
        line, = ax.plot([volS,volS+1], [s,s], 'b-')
        volS += 1
        pr = s
    if asks:
        line.set_label('Supply')
        
    if bids or asks:
        plt.legend()
    plt.show()

# Use this to plot supply and demand curves from supply and demand ranges and stepmode
def plot_sup_dem(seller_num, sup_ranges, buyer_num, dem_ranges, stepmode):
    asks = []
    for s in range(seller_num):
        asks.append(getorderprice(s, sup_ranges, seller_num, stepmode))
    asks.sort()
    bids = []
    for b in range(buyer_num):
        bids.append(getorderprice(b, dem_ranges, buyer_num, stepmode))
    bids.sort()
    bids.reverse()
    
    make_supply_demand_plot(bids, asks) 
    plt.show()

# plot sorted trades, useful is some situations - won't be used in this worksheet
def in_order_plot(trial_id):
    prices_fname = trial_id + '_tape.csv'
    y = np.empty(0)
    with open(prices_fname, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            price = float(row[2])
            y = np.append(y,price)
    y = np.sort(y)
    x = list(range(len(y)))

    plt.plot(x, y, 'x', color='black')   
    plt.show()

# plot offset function
def plot_offset_fn(offset_fn, total_time_seconds):   
    x = list(range(total_time_seconds))
    offsets = []
    for i in range(total_time_seconds):
        offsets.append(offset_fn(i))
    plt.plot(x, offsets, 'x', color='black')  
    plt.show()






## ============================ PLOT STATIC MARKET ====================================##

def PlotStaticMarket():
    # Solution

    # First, configure the trader specifications
    # {'k': 4, 's_min': -1.0, 's_max': +1.0}
    buyers_spec = [('PRDE', 10, {'k': 4, 's_min': -1.0, 's_max': +1.0})]
    sellers_spec = [('PRDE', 10, {'k': 4, 's_min': -1.0, 's_max': +1.0})]
    traders_spec = {'sellers':sellers_spec, 'buyers': buyers_spec}

    # Next, confiure the supply and demand (and plot it)
    sup_range = (100,200)
    dem_range = (100,200)

    plot_sup_dem(10, [sup_range], 10, [dem_range], 'fixed')

    # Next, configure order schedules


    start_time = 100
    end_time = 600
    supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [sup_range], 'stepmode': 'fixed'}]
    demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [dem_range], 'stepmode': 'fixed'}]

    order_interval = 60

    order_sched = {
    "sup": supply_schedule,
    "dem": demand_schedule,
    "interval": order_interval,
    "timemode": "periodic"
    }
    trial_id = 'test_2'
    tdump = open('test_2_avg_balance.csv','w')
    dump_all = True
    verbose = True

    # Now, run the market session
    market_session(trial_id, start_time, end_time, traders_spec, order_sched, tdump, dump_all, verbose)

    tdump.close()

    # Finally, plot the trades that executed during the market session
    plot_trades('test_2')


## =========================== PLOT DYNAMIC MARKET 1 ============================~~

def PlotDynamicMarketMarkOne():
    num_sellers = 1
    num_buyers = 1

    sellers_spec = [('PRZI', 1)]
    buyers_spec = [('PRZI', 1)]
    traders_spec = {'sellers':sellers_spec, 'buyers': buyers_spec}

    range1 = (50,100) # enter range before shock
    range2 = (150, 200)# enter range after shock

    start_time = 0
    mid_time = 5*60
    end_time = 10*60
    supply_schedule = [{'from': start_time, 'to': mid_time, 'ranges': [range1], 'stepmode': 'fixed'},
                    {'from': mid_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'fixed'}]
    demand_schedule = supply_schedule

    order_interval = 15
    order_sched = {'sup': supply_schedule, 'dem': demand_schedule,
                'interval': order_interval, 'timemode': 'periodic'}

    trial_id = 'test_5'
    tdump = open('test_5_avg_balance.csv','w')
    dump_all = True
    verbose = False


    market_session(trial_id, start_time, end_time, traders_spec, order_sched, tdump, dump_all, verbose)

    tdump.close()

    plot_trades('test_5')

def schedule_offsetfn(t):
    pi2 = math.pi * 2
    c = math.pi*3000
    wavelength = t/c
    gradient = 100*t/(c/pi2)
    amplitude = 100*t/(c/pi2)
    offset = gradient + amplitude * math.sin(wavelength*t)
    return int(round(offset,0))


def MarketShockFixed():
    trial_id = 'MarketShockFixed'
    start_time = 0
    thirdTime = 60
    twoThirdTime = 120
    end_time = 400

    sup_range1 = (10, 190)
    dem_range1 = (10, 190)

    sup_range2 = (200, 300)
    dem_range2 = (200, 300)

    supply_schedule = [{'from': start_time, 'to': thirdTime, 'ranges': [sup_range1], 'stepmode': 'fixed'},
                       {'from': thirdTime, 'to': twoThirdTime, 'ranges': [sup_range2], 'stepmode': 'fixed'},
                       {'from': twoThirdTime, 'to': end_time, 'ranges': [sup_range1], 'stepmode': 'fixed'}
                        ]


    demand_schedule = [{'from': start_time, 'to': thirdTime, 'ranges': [dem_range1], 'stepmode': 'fixed'},
                       {'from': thirdTime, 'to': twoThirdTime, 'ranges': [dem_range2], 'stepmode': 'fixed'},
                       {'from': twoThirdTime, 'to': end_time, 'ranges': [dem_range1], 'stepmode': 'fixed'}
                        ]


    order_schedule = {'sup':supply_schedule, 'dem':demand_schedule,
               'interval':30, 'timemode':'drip-fixed'}



    buyers_spec = [('PRDE', 100, {'k': 4, 's_min': -1.0, 's_max': +1.0})]

    #buyers_spec = [('ZIP', 100)]

    sellers_spec = buyers_spec
    traders_spec = {'sellers':sellers_spec, 'buyers':buyers_spec}

    tdump=open('MarketShockFixed.csv','w')
    dump_all = True
    verbose = True
    market_session(trial_id, start_time, end_time, traders_spec, order_schedule, tdump, dump_all, verbose)
    tdump.close()

    plot_trades(trial_id)


def sum_to_n(n):
    # Generate the series of +ve integer lists which sum to a +ve integer, n.
    from operator import sub
    b, mid, e = [0], list(range(1, n)), [n]
    splits = (d for i in range(n) for d in combinations(mid, i)) 
    return (list(map(sub, chain(s, e), chain(b, s))) for s in splits)

def ReturnSumPermutations(sum, length):
    listOfAnswers = []
    for p in sum_to_n(sum):    
        listOfAnswers.append(p)

    listOfAnswers=[x for x in listOfAnswers if len(x)==length]
    return listOfAnswers


## ======================================PRDE_ParameterChangeMaster======================================###
## This section of code is responsible for the varying k and F values to evaluate PRDE
def PRDE_ParameterChangeMaster(baseTitle, min_k, max_k, min_F, max_F, step_F, trialsPerSetting, tradersPerSide):
    print("PRDE MASTER BEGUN")
    for k in range (min_k, max_k):
        for F in np.arange(min_F, max_F, step_F):
            # for every combination of PRDE, SHVR, ZIC, and ZIP summing to tradersPerSide
            traderAmountPermutations = ReturnSumPermutations(tradersPerSide, 4)
            permutationNumber = 0
            for traderAmountPerm in traderAmountPermutations:
                # do n trials of this setting
                # currently we only have one setting
                
                for n in range(0, trialsPerSetting):
                    # title in form of "PRDE_K(k)_F(F)_S(setting number)_P(permutationNumber)_N(trialNumber)"

                     # market scenarios: stable_equal
                    #                   stable_buyers
                    #                   stable_sellers

                    #                   fluctuating_equal
                    #                   fluctuating_buyers
                    #                   fluctuating_sellers
                    
                    #                   volatile_equal
                    #                   volatile_buyers
                    #                   volatile_sellers

                    # == market session 0 == 
                    marketSetting = 0
                    title = baseTitle + "_PRDE_" + "K" + str(k) + "_F" + str(int(F*10)) + "_S" + str(marketSetting) + "_P" + str(permutationNumber) + "_N" + str(n)
                    print(title)
                    ProfitComparisionSimulationPRDEchangingParams(title, False, k, F, traderAmountPerm[0], traderAmountPerm[1], traderAmountPerm[2], traderAmountPerm[3], 0, "stable_equal", 100, 100, False)
                    # == market session 1 == 
                    marketSetting = 1
                    title = baseTitle + "_PRDE_" + "K" + str(k) + "_F" + str(int(F*10)) + "_S" + str(marketSetting) + "_P" + str(permutationNumber) + "_N" + str(n)
                    print(title)
                    ProfitComparisionSimulationPRDEchangingParams(title, False, k, F, traderAmountPerm[0], traderAmountPerm[1], traderAmountPerm[2], traderAmountPerm[3],0,"stable_buyers", 100, 100, False)
                    # == market session 2 == 
                    marketSetting = 2
                    title = baseTitle + "_PRDE_" + "K" + str(k) + "_F" + str(int(F*10)) + "_S" + str(marketSetting) + "_P" + str(permutationNumber) + "_N" + str(n)
                    print(title)
                    ProfitComparisionSimulationPRDEchangingParams(title, False, k, F, traderAmountPerm[0], traderAmountPerm[1], traderAmountPerm[2], traderAmountPerm[3],0, "stable_sellers", 100, 100, False)
                    # == market session 3 == 
                    marketSetting = 3
                    title = baseTitle + "_PRDE_" + "K" + str(k) + "_F" + str(int(F*10)) + "_S" + str(marketSetting) + "_P" + str(permutationNumber) + "_N" + str(n)
                    print(title)
                    ProfitComparisionSimulationPRDEchangingParams(title, False, k, F, traderAmountPerm[0], traderAmountPerm[1], traderAmountPerm[2], traderAmountPerm[3],0, "fluctuating_equal", 100, 100, False)
                    # == market session 4 == 
                    marketSetting = 4
                    title = baseTitle + "_PRDE_" + "K" + str(k) + "_F" + str(int(F*10)) + "_S" + str(marketSetting) + "_P" + str(permutationNumber) + "_N" + str(n)
                    print(title)
                    ProfitComparisionSimulationPRDEchangingParams(title, False, k, F, traderAmountPerm[0], traderAmountPerm[1], traderAmountPerm[2], traderAmountPerm[3],0, "fluctuating_buyers", 100, 100, False)
                    # == market session 5 == 
                    marketSetting = 5
                    title = baseTitle + "_PRDE_" + "K" + str(k) + "_F" + str(int(F*10)) + "_S" + str(marketSetting) + "_P" + str(permutationNumber) + "_N" + str(n)
                    print(title)
                    ProfitComparisionSimulationPRDEchangingParams(title, False, k, F, traderAmountPerm[0], traderAmountPerm[1], traderAmountPerm[2], traderAmountPerm[3],0, "fluctuating_sellers", 100, 100, False)
                    # == market session 6 == 
                    marketSetting = 6
                    title = baseTitle + "_PRDE_" + "K" + str(k) + "_F" + str(int(F*10)) + "_S" + str(marketSetting) + "_P" + str(permutationNumber) + "_N" + str(n)
                    print(title)
                    ProfitComparisionSimulationPRDEchangingParams(title, False, k, F, traderAmountPerm[0], traderAmountPerm[1], traderAmountPerm[2], traderAmountPerm[3],0, "volatile_equal", 100, 100, False)
                    # == market session 7 == 
                    marketSetting = 7
                    title = baseTitle + "_PRDE_" + "K" + str(k) + "_F" + str(int(F*10)) + "_S" + str(marketSetting) + "_P" + str(permutationNumber) + "_N" + str(n)
                    print(title)
                    ProfitComparisionSimulationPRDEchangingParams(title, False, k, F, traderAmountPerm[0], traderAmountPerm[1], traderAmountPerm[2], traderAmountPerm[3],0, "volatile_buyers", 100,100, False)
                    # == market session 8 == 
                    marketSetting = 8
                    title = baseTitle + "_PRDE_" + "K" + str(k) + "_F" + str(int(F*10)) + "_S" + str(marketSetting) + "_P" + str(permutationNumber) + "_N" + str(n)
                    print(title)
                    ProfitComparisionSimulationPRDEchangingParams(title, False, k, F, traderAmountPerm[0], traderAmountPerm[1], traderAmountPerm[2], traderAmountPerm[3],0, "volatile_sellers", 100,100, False)

                permutationNumber += 1
                
                
                # do n trials of this setting

                # do n trials of this setting


# plot offset function
def plot_offset_fn(offset_fn, total_time_seconds, show):   
    x = list(range(total_time_seconds))
    offsets = []
    for i in range(total_time_seconds):
        offsets.append(offset_fn(i))
    plt.plot(x, offsets, 'x', color='black') 
    if(show):
        plt.show() 

def Merton_Jump_Diffusion_Model_GeneratePath(start_price, standard_deviation, safe_factor, expectedJumpsPerYear, v, steps, n):
    size=(steps,n)
    dt = 1/steps 
    poissonRandomSample = np.multiply(np.random.poisson( expectedJumpsPerYear*dt, size=size), np.random.normal(0,v, size=size)).cumsum(axis=0)
    geometricSum = np.cumsum(((safe_factor -  standard_deviation**2/2 -expectedJumpsPerYear*(v**(2*0.5)))*dt + standard_deviation*np.sqrt(dt) * np.random.normal(size=size)) , axis=0)
    
    return np.exp(geometricSum+poissonRandomSample)*start_price

# global brownian noise path - may pass into 

def Merton_Supply_Jump_Function(t):
    t = int(t)
    #print("MERTON PATH (supply) " + str(t))
    #print (int(mertonSupplyPath[t]))
    return int(mertonSupplyPath[t])

def Merton_Demand_Jump_Function(t):
    t = int(t)
    #print("MERTON PATH (demand)" + str(t))
    #print (int(mertonDemandPath[t]))
    return int(mertonDemandPath[t])


# Dynamic market function
def ProfitComparisionSimulationPRDEchangingParams(title, plotMarketBehaviour, k, F, PRDE_amt, SHVR_amt, ZIC_amt, ZIP_amt, PRSH_amt, marketCondition, baseSupplyPrice, baseDemandPrice, dumpAll):
    trial_id = title
    start_time = 0
    end_time = 600

    drunkard_walk_end_time = 620
    # generate a varying merton jump drunkards walk function of supply and demand
    global mertonSupplyPath
    global mertonDemandPath

    # market scenarios: stable_equal
    #                   stable_buyers
    #                   stable_sellers

    #                   fluctuating_equal
    #                   fluctuating_buyers
    #                   fluctuating_sellers

    #                   volatile_equal
    #                   volatile_buyers
    #                   volatile_sellers

    # drunkard presets: noTrend_lowFluctuation
    #                   downward_lowFluctuation
    #                   upward_lowFluctuation

    #                   noTrend_highFluctuation
    #                   downward_highFluctuation
    #                   upward_highFluctuation

    #                   noTrend_volatile
    #                   downward_volatile
    #                   upward_volatile

    
    # STEADY MARKETS
    if(marketCondition == "stable_equal"):
        #print(marketCondition + " market")
        mertonSupplyPath = GenerateJumpDiffusionPathPreset(basePrice=100, preset="noTrend_lowFluctuation", length=drunkard_walk_end_time)
        mertonDemandPath = GenerateJumpDiffusionPathPreset(basePrice=100, preset="noTrend_lowFluctuation", length=drunkard_walk_end_time)
    if(marketCondition == "stable_buyers"):
        #print(marketCondition + " market")
        mertonSupplyPath = GenerateJumpDiffusionPathPreset(basePrice=100, preset="upward_lowFluctuation", length=drunkard_walk_end_time)
        mertonDemandPath = GenerateJumpDiffusionPathPreset(basePrice=100, preset="downward_lowFluctuation", length=drunkard_walk_end_time)
    if(marketCondition == "stable_sellers"):
        #print(marketCondition + " market")
        mertonSupplyPath = GenerateJumpDiffusionPathPreset(basePrice=100, preset="downward_lowFluctuation", length=drunkard_walk_end_time)
        mertonDemandPath = GenerateJumpDiffusionPathPreset(basePrice=100, preset="upward_lowFluctuation", length=drunkard_walk_end_time)

    # FLUCTUATING MARKETS
    if(marketCondition == "fluctuating_equal"):
        #print(marketCondition + " market")
        mertonSupplyPath = GenerateJumpDiffusionPathPreset(basePrice=100, preset="noTrend_highFluctuation", length=drunkard_walk_end_time)
        mertonDemandPath = GenerateJumpDiffusionPathPreset(basePrice=100, preset="noTrend_highFluctuation", length=drunkard_walk_end_time)
    if(marketCondition == "fluctuating_buyers"):
        #print(marketCondition + " market")
        mertonSupplyPath = GenerateJumpDiffusionPathPreset(basePrice=100, preset="upward_highFluctuation", length=drunkard_walk_end_time)
        mertonDemandPath = GenerateJumpDiffusionPathPreset(basePrice=100, preset="downward_highFluctuation", length=drunkard_walk_end_time)
    if(marketCondition == "fluctuating_sellers"):
        #print(marketCondition + " market")
        mertonSupplyPath = GenerateJumpDiffusionPathPreset(basePrice=100, preset="downward_highFluctuation", length=drunkard_walk_end_time)
        mertonDemandPath = GenerateJumpDiffusionPathPreset(basePrice=100, preset="upward_highFluctuation", length=drunkard_walk_end_time)

    # VOLATILE MARKETS
    if(marketCondition == "volatile_equal"):
        #print(marketCondition + " market")
        mertonSupplyPath = GenerateJumpDiffusionPathPreset(basePrice=100, preset="noTrend_volatile", length=drunkard_walk_end_time)
        mertonDemandPath = GenerateJumpDiffusionPathPreset(basePrice=100, preset="noTrend_volatile", length=drunkard_walk_end_time)
    if(marketCondition == "volatile_buyers"):
        #print(marketCondition + " market")
        mertonSupplyPath = GenerateJumpDiffusionPathPreset(basePrice=100, preset="upward_volatile", length=drunkard_walk_end_time)
        mertonDemandPath = GenerateJumpDiffusionPathPreset(basePrice=100, preset="downward_volatile", length=drunkard_walk_end_time)
    if(marketCondition == "volatile_sellers"):
        #print(marketCondition + " market")
        mertonSupplyPath = GenerateJumpDiffusionPathPreset(basePrice=100, preset="downward_volatile", length=drunkard_walk_end_time)
        mertonDemandPath = GenerateJumpDiffusionPathPreset(basePrice=100, preset="upward_volatile", length=drunkard_walk_end_time)
        
    sup_range1 = (1, 50, Merton_Supply_Jump_Function)
    dem_range1 = (1, 50, Merton_Demand_Jump_Function)


    supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [sup_range1], 'stepmode': 'fixed'}]


    demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [dem_range1], 'stepmode': 'fixed'}]


    order_schedule = {'sup':supply_schedule, 'dem':demand_schedule,
               'interval':30, 'timemode':'drip-fixed'}



    PRDR_Setup_F_dict =  {'de_state': 'active_s0',           # initial state: strategy 0 is active (being evaluated)
                         's0_index': 0,                      # s0 starts out as active strat
                         'snew_index': k,                    # (k+1)th item of strategy list is DE's new strategy
                         'snew_stratval': None,              # assigned later
                         'F': F                              # differential weight -- usually between 0 and 2
        }

    buyers_spec = [('PRDE', PRDE_amt, {'k': k, 's_min': -1.0, 's_max': +1.0, 'diffVolDict':PRDR_Setup_F_dict}),('SHVR',SHVR_amt), ('ZIC', ZIC_amt), ('ZIP', ZIP_amt), ('PRSH', PRSH_amt, {'k': k, 's_min': -1.0, 's_max': +1.0, 'diffVolDict':PRDR_Setup_F_dict})]

    sellers_spec = buyers_spec
    traders_spec = {'sellers':sellers_spec, 'buyers':buyers_spec}

    tdump=open((title) + '.csv','w')
    dump_all = dumpAll
    verbose = False
    market_session(trial_id, start_time, end_time, traders_spec, order_schedule, tdump, dump_all, verbose)
    tdump.close()
    if(plotMarketBehaviour):
        #plot_sup_dem(8, [sup_range1, sup_range2, sup_range1], 8, [dem_range1, dem_range2, dem_range1], 'fixed')
        plot_trades(trial_id)




def GetAllLines_CSV_Data(name):
    marketSessionList = []
    timeRowPos = 1
    bestBidRowPos = 2
    bestOrderRowPos = 3
    firstTraderNamePos = 4
    traderNameOffsetAmt = 4

    shortestRowLength = 9
    rowIncreasePerTrader = 4
    
    numberOfAgents = 0
    with open((name + ".csv"), newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            rowList = []
            for i in range(4, len(row)-rowIncreasePerTrader, rowIncreasePerTrader):
                
                traderName = row[i]
                cumulativeProfit = float(row[i+1])
                population = float(row[i+2])
                avgProfit = float(row[i+3])
                time = float(row[1])
                agentRatioOfExpectedMaxProfit = 0
                rowList.append([traderName, cumulativeProfit, population, avgProfit, name, time, agentRatioOfExpectedMaxProfit])
            lastRow = rowList
            marketSessionList.append(rowList)

    # get the sum of maximum average profit per agent in the simulation (for all agents)
    sumMaxAverageProfitPerAgent = 0
    for traderResult in lastRow:
        sumMaxAverageProfitPerAgent += traderResult[3]
        numberOfAgents+=1
    
    # add agentRatioOfExpectedMaxProfit element to each item in marketSessionList (currentProfitPerAgent / sumMaxAverageProfitPerAgent) * number of agents
    for row in marketSessionList:
        for traderStatAtTime in row:
            if(sumMaxAverageProfitPerAgent > 0):
                agentRatioOfExpectedMaxProfit = (traderStatAtTime[3] / sumMaxAverageProfitPerAgent) * numberOfAgents * 1.1
                
                traderStatAtTime[6] = agentRatioOfExpectedMaxProfit
            # estimate in dataset
            else:
                agentRatioOfExpectedMaxProfit = lerp(0, 1, (float(traderStatAtTime[5])/600))
                traderStatAtTime[6] = agentRatioOfExpectedMaxProfit

    return marketSessionList




# opens a csv file and assumes it is only one line
# returns a list of lists.
# each sublist contains end performance of each agent
# list of [trader, profit, population, avgProfit, trialName, avgReturn] tuples
def GetOneLine_CSV_Data(name):
    with open((name + ".csv"), newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        timeRowPos = 1
        bestBidRowPos = 2
        bestOrderRowPos = 3
        firstTraderNamePos = 4
        traderNameOffsetAmt = 4

        shortestRowLength = 9
        rowIncreasePerTrader = 4
        # create list of [trader, profit, population, avgProfit, trialName, avgReturn] tuples

        traderProfitTuples = []

        

        for row in spamreader:
            rowLength = len(row)
            for i in range(4, rowLength-rowIncreasePerTrader, rowIncreasePerTrader):
                traderName = row[i]
                cumulativeProfit = float(row[i+1])
                population = float(row[i+2])
                avgProfit = float(row[i+3])
                traderProfitTuples.append([traderName, cumulativeProfit, population, avgProfit, name])

            # now that we have the whole dataset, get actual comapred to expected performance
            # expected performance is 1.0
            # actual performance is ((mean profit per trader)/(total mean profit per trader)) * number of traders
            totalMeanProfitPerTrader = 0
            
            for traderProfitTuple in traderProfitTuples:
                totalMeanProfitPerTrader += traderProfitTuple[3]
            
            if(totalMeanProfitPerTrader > 0):
                for traderProfitTuple in traderProfitTuples:
                    profitPerformance = (traderProfitTuple[3]/totalMeanProfitPerTrader) * len(traderProfitTuples)
                    traderProfitTuple.append(profitPerformance)
            else:
                print("oh crikey! There was no profit at all! Setting avg return to 1")
                print(traderProfitTuples)
                for traderProfitTuple in traderProfitTuples:
                    
                    profitPerformance = 1
                    traderProfitTuple.append(profitPerformance)
    
    return traderProfitTuples

def Mega_Parameter_Change_String_Constructor(baseTitle, K, F, S, P, N):
    megaString = baseTitle + "_PRDE_K" + str(K) + "_F" + str(F) + "_S" + str(S) + "_P" + str(P) + "_N" + str(N)
    return megaString

def Construct_Mega_Array(baseTitle, start_k, end_k, start_f, end_f, step_f, s_marketSettings, p_maxTradersPerSide, n_trialsPerSetup):
    ## this function is going to be a little bit of a nightmare
    # create a string based on base title and start-end values, then read that csv.
    # add the result to a mega array

    megaArray = []
    k_index = 0
    f_index = 0
    s_index = 0
    p_index = 0

    traderPermutations = ReturnSumPermutations(p_maxTradersPerSide, 4)
    traderPermutationsLength = len(traderPermutations)
    # firstly, find out how big the array should be and produce it
    for k in range(start_k, end_k):
        megaArray.append([])
        for F in np.arange(start_f, end_f, step_f):
            megaArray[k_index].append([])
            for s in  s_marketSettings:
                megaArray[k_index][f_index].append([])
                for p in range(traderPermutationsLength-1):
                    
                    megaArray[k_index][f_index][s_index].append([])
                    for n in range(n_trialsPerSetup):
                        megaArray[k_index][f_index][s_index][p_index].append([])
                    p_index += 1
                p_index = 0
                s_index += 1
            s_index = 0
            f_index += 1
        f_index = 0
        k_index += 1
    k_index = 0


    k_index = 0
    f_index = 0
    s_index = 0
    p_index = 0

    print("The mega array is now constructed")
     # fill the mega array from csv data

    for k in range(start_k, end_k):
        for F in np.arange(start_f, end_f, step_f):
            for s in s_marketSettings:
                for p in range(traderPermutationsLength-1):
                    for n in range(n_trialsPerSetup):
                        titleString = Mega_Parameter_Change_String_Constructor(baseTitle, k, (int(F*10)), s, p, n)
                        
                        megaArray[k_index][f_index][s_index][p_index][n] = GetOneLine_CSV_Data(titleString)
                       # print(megaArray[k_index][f_index][s_index][p_index][n])
                    p_index += 1
                p_index = 0
                s_index += 1
            s_index = 0
            f_index += 1
        f_index = 0
        k_index += 1
    k_index = 0

    return megaArray

# Note: k is an index (starting at 0, not 4) and F is an index (0 to 10 normally) - not its weird decimal form
# Trader name is usally PRDE
def TraderDataFromMegaArrayIndex (megaArray, k_index, f_index, extractedTraderName, s_marketSettings, traderPermutationsLength, n_trialsPerSetup):
    

    Current_K_F_Simulations = megaArray[k_index][f_index]

    ListOfTuplesUnderConditions_K_F = []
    # for each value of s,
    for s in range (0, len(s_marketSettings)):
        for p in range(0, traderPermutationsLength-1):
            for n in range(0, n_trialsPerSetup):
              #  print("s " + str(s))
              #  print("p " + str(p))
              #  print("n " + str(n))
              #  print("-\n")
                # under market conditions
              #  print(Current_K_F_Simulations[s])
              #  print("-\n")
                # permutations n = p
              #  print(Current_K_F_Simulations[s][p])
              #  print("-\n")
                # trial
              #  print(Current_K_F_Simulations[s][p][n])
              #  print("===")
                ListOfTuplesUnderConditions_K_F.append(Current_K_F_Simulations[s][p][n])
    
    ListOfTuplesUnderConditions_K_F_PRDE_ONLY = []
    # for each tuple in the list, extract the sub tuple of name extractedTraderType (Normally PRDE)
    i = 0
    for tupleList in ListOfTuplesUnderConditions_K_F:

        # get the tuple in tupleList whose first element is 'PRDE'
        for tupleItem in tupleList:
            if(tupleItem[0] == extractedTraderName or tupleItem[0] ==  (" " + extractedTraderName)):
                ListOfTuplesUnderConditions_K_F_PRDE_ONLY.append(tupleItem)

    # nice, now we have the list of the results of the trader that we are interested in
    return ListOfTuplesUnderConditions_K_F_PRDE_ONLY


def GetMeanProfitRatioFromTraderTupleList(tupleList):
    sum_return = 0
    for tupleItem in tupleList:
        sum_return += tupleItem[5]        
    mean_return = sum_return / len(tupleList)  
    return mean_return

def GetAllProfitRatiosFromTraderTupleList(tupleList):
    ratioList = []
    for tupleItem in tupleList:
        ratioList.append(tupleItem[5])
    return (flatten(ratioList))
            
            

            




## ======================================PRDE_ParameterChangeMasterEND======================================###



def ProfitComparisonSimulationTrial(title, plotMarketBehaviour):
    trial_id = title
    start_time = 0
    end_time = 600

    sup_range1 = (50, 100)
    dem_range1 = (50, 100)

    supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [sup_range1], 'stepmode': 'fixed'}]


    demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [dem_range1], 'stepmode': 'fixed'}]


    order_schedule = {'sup':supply_schedule, 'dem':demand_schedule,
               'interval':30, 'timemode':'drip-fixed'}
    

    
    PRDR_Setup_F_dict =  {'de_state': 'active_s0',           # initial state: strategy 0 is active (being evaluated)
                         's0_index': 0,                      # s0 starts out as active strat
                         'snew_index': 4,                    # (k+1)th item of strategy list is DE's new strategy
                         'snew_stratval': None,              # assigned later
                         'F': 0.8                              # differential weight -- usually between 0 and 2
        }



    buyers_spec = [('PRDE', 4, {'k': 4, 's_min': -1.0, 's_max': +1.0, 'diffVolDict':PRDR_Setup_F_dict}),('SHVR',4),('ZIC',4),('ZIP',4), ('PRSH', 4, {'k': 4, 's_min': -1.0, 's_max': +1.0, 'diffVolDict':PRDR_Setup_F_dict})]

    sellers_spec = buyers_spec
    traders_spec = {'sellers':sellers_spec, 'buyers':buyers_spec}

    tdump=open((title) + '.csv','w')
    dump_all = True
    verbose = False
    market_session(trial_id, start_time, end_time, traders_spec, order_schedule, tdump, dump_all, verbose)
    tdump.close()
    if(plotMarketBehaviour):
        plot_trades(trial_id)






def Get_CSV_Data(name, tag):
    with open((name + ".csv"), newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        rowLength = 0

        timeRowPos = 1
        bestBidRowPos = 2
        bestOrderRowPos = 3
        firstTraderNamePos = 4
        traderNameOffsetAmt = 4

        shortestRowLength = 9
        rowIncreasePerTrader = 4

        timestamps = []
        bestBids = []
        bestOrders = []

        traders = []
        traderProfits = []
        traderMeanProfits = []
        traderPopulation = []
        
        firstLine = True
        # if length = 21, 4 traders
        for row in spamreader:
            if(firstLine):
                rowLength = len(row)
                # one trader type
                if(rowLength >= shortestRowLength):
                    traders.append(row[firstTraderNamePos])
                    traderProfits.append([])
                    traderMeanProfits.append([])
                # 2 trader types
                if(rowLength >= shortestRowLength + rowIncreasePerTrader * 1):
                    traders.append(row[firstTraderNamePos + rowIncreasePerTrader * 1])
                    traderProfits.append([])
                    traderMeanProfits.append([])
                # 3 trader types
                if(rowLength >= shortestRowLength + rowIncreasePerTrader * 2):
                    traders.append(row[firstTraderNamePos + rowIncreasePerTrader * 2])
                    traderProfits.append([])
                    traderMeanProfits.append([])
                # 4 trader types
                if(rowLength >= shortestRowLength + rowIncreasePerTrader * 3):
                    traders.append(row[firstTraderNamePos + rowIncreasePerTrader * 3])
                    traderProfits.append([])
                    traderMeanProfits.append([])
                # 5 trader types
                if(rowLength >= shortestRowLength + rowIncreasePerTrader * 4):
                    traders.append(row[firstTraderNamePos + rowIncreasePerTrader * 4])
                    traderProfits.append([])
                    traderMeanProfits.append([])
                # 6 trader types
                if(rowLength >= shortestRowLength + rowIncreasePerTrader * 5):
                    traders.append(row[firstTraderNamePos + rowIncreasePerTrader * 5])
                    traderProfits.append([])
                    traderMeanProfits.append([])
                    
                firstLine = False


            

            # append timestamp to list
            timestamp = (row[timeRowPos])
            if(timestamp == ' None' or timestamp == 'None'):
                timestamp = float(0.0)
            timestamps.append(float(timestamp))

            # append best bid to list
            bestBid = (row[bestBidRowPos])
            if(bestBid == ' None' or bestBid == 'None'):
                bestBid = 0
            bestBids.append(float(bestBid))

            # append best order to list
            bestOrder = (row[bestOrderRowPos])
            if(bestOrder == ' None' or bestOrder == 'None'):
                bestOrder = 0
            bestOrders.append(float(bestOrder))

            # get cumulative profit for trader i on row
            j = 0
            for i in range(5, len(row), rowIncreasePerTrader):
                cumulativeProfit = float(row[i])
                traderProfits[j].append([cumulativeProfit, timestamp, tag])
                j += 1
            


            # get avg profit per trader for i on row
            j = 0
            for i in range(7, len(row), rowIncreasePerTrader):
                meanProfit = float(row[i])
                traderMeanProfits[j].append([meanProfit, timestamp, tag])
                j += 1

        # traderProfits is a list of list of pairs

        # traderMeanProfits is a list of list of pairs
        return traders, traderProfits, traderMeanProfits, timestamps, bestBids, bestOrders
        
def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])




import random
def PlotDataNew(traders, traderProfitTimestampPairsListSet, plotType):

    # traderProfitTupleListSet may contain data from multiple simulation trials under the same conditions

    # traderProfits and traderMeanProfits are in tuples with their timestamps

    # separate timestamps from profits for each trader

    traderProfits = []
    traderTimestamps = []
    traderTags = []

    i = 0
    # iterate through each trader type
    for traderProfitTimestampPairsList in traderProfitTimestampPairsListSet:
        print(traderProfitTimestampPairsList)
        # first trader profit tuple list
        traderProfits.append([])
        traderTimestamps.append([])
        traderTags.append([])
        # sort traderProfitTimestampPairsList by the traderProfitPair timestamp variable (traderProfitPair[1])
        if(plotType == "line"):
            traderProfitTimestampPairsList = sorted(traderProfitTimestampPairsList,key=itemgetter(1))
        elif(plotType == "scatter"):
            traderProfitTimestampPairsList = sorted(traderProfitTimestampPairsList,key=itemgetter(2))
        # fill traderProfits[i] with the profit and  traderTimestamps[i] with the corresponding timestamp
        for traderProfitPair in traderProfitTimestampPairsList:
            traderProfits[i].append(traderProfitPair[0])
            traderTimestamps[i].append(traderProfitPair[1])

            # trader profit pair 2 is the trial id
            traderTags[i].append(traderProfitPair[2])

        # print the list of lists of timestamps for the current trader
       # print("length of traders")
       # print(traders)
       # print("length of traderProfitTimestampPairsListSet")
       # print(len(traderProfitTimestampPairsListSet))
       # print(traderProfitTimestampPairsListSet)

       # print(traders[i])
       # print('\n')
       # print(traderTimestamps[i])
        i += 1

        

    # ensure floats 
    
    i = 0
    # plot cumulative profits

    for i in range (len(traders)):
        print(traders[i])
        if(traders[i] == " PRDE" or traders[i] == "PRDE"):
            colour = colourList[0]
        if(traders[i] == " SHVR" or traders[i] == "SHVR"):
            colour = colourList[1]
        if(traders[i] == " ZIC" or traders[i] == "ZIC"):
            colour = colourList[2]
        if(traders[i] == " ZIP" or traders[i] == "ZIP"):
            colour = colourList[3]
        if(traders[i] == " PRSH" or traders[i] == "PRSH"):
            colour = colourList[4]
        
        ts = [float(x) for x in traderTimestamps[i]]


        # prepare to remove n random trials by tag 
        #

        #print("ABOUT TO PLOT:")
        #print(traders[i])
        
        #print(ts)
        #print(len(traderTimestamps[i]))

       # print(traderProfits[i])
        #print(len(traderProfits[i]))

        
        
        if(plotType == "line"):
            plt.plot( ts, traderProfits[i], label = traders[i], dashes=[0.3, 0.3], color = colour)
        if(plotType == "scatter"):
            plt.scatter( ts, traderProfits[i], label = traders[i], marker = '1', color = colour, s = 0.4)
        ## nice stuff, now we can plot a linear regression as well

        linear_regressor = Ridge(fit_intercept= False)  # create object for the class
        # create shaped data objects 
        X = np.array(ts).reshape(-1, 1)
        Y = np.array(traderProfits[i]).reshape(-1, 1)
        linear_regressor.fit(X,Y)  # perform linear regression

        Y_pred = linear_regressor.predict(X)  # make predictions

        plt.plot(ts, Y_pred, color='black', linewidth = 3.3 )
        plt.plot(ts, Y_pred, color=colour, linewidth = 1.8 )

    plt.xlabel('Simulation Time / (s)')
    plt.ylabel('Cumulative trader profit')
    plt.title('Cumulative Trader Profits in a static market averaged over 100 trials')
    lgnd = plt.legend(loc='upper right', scatterpoints=1, fontsize=10)
    for handle in lgnd.legendHandles:
        handle.set_sizes([100.0])
    plt.show()


def AppendTraderProfitPairs(set1, set2):
    for i in range (len(set2) ):
        set1[i].extend(set2[i])

    return set1

def ReadSimulationAndCollateResults(baseTitleName, repeats):
    print("ReadSimulationAndCollateResults")

    
    name = baseTitleName
    # i = 0 - base case
    name = baseTitleName + "_" + str(0)
    traders, traderProfitTimestampPairsListSet, traderMeanProfitsTimestampPairsListSet, timestamps, bestBids, bestOrders = Get_CSV_Data(name, 0)


    # i > 0 - iterative case
    for i in range(1, repeats):
        name = baseTitleName + "_" + str(i)
        traders, traderProfitTimestampPairsListSet2, traderMeanProfitsTimestampPairsListSet2, timestamps, bestBids, bestOrders = Get_CSV_Data(name, i)
        traderProfitTimestampPairsListSet = AppendTraderProfitPairs(traderProfitTimestampPairsListSet,traderProfitTimestampPairsListSet2 )
        # TODO
        # implement for mean profit pairs
        
    
    return traders, traderProfitTimestampPairsListSet, traderMeanProfitsTimestampPairsListSet


def RunSimulationAndCollateResults(simulation,baseTitleName, repeats, plotMarketBehaviour):
    print("RunSimulationAndCollateResults")

    name = baseTitleName
    # i = 0 - base case
    name = baseTitleName + "_" + str(0)
    simulation(title = name, plotMarketBehaviour=plotMarketBehaviour)
    traders, traderProfitTimestampPairsListSet, traderMeanProfitsTimestampPairsListSet, timestamps, bestBids, bestOrders = Get_CSV_Data(name, 0)


    # i > 0 - iterative case
    for i in range(1, repeats):
        name = baseTitleName + "_" + str(i)
        print(name)
        simulation(title = name, plotMarketBehaviour=plotMarketBehaviour)
        traders, traderProfitTimestampPairsListSet2, traderMeanProfitsTimestampPairsListSet2, timestamps, bestBids, bestOrders = Get_CSV_Data(name, i)
        traderProfitTimestampPairsListSet = AppendTraderProfitPairs(traderProfitTimestampPairsListSet,traderProfitTimestampPairsListSet2 )
        # TODO
        # implement for mean profit pairs
        
    
    return traders, traderProfitTimestampPairsListSet, traderMeanProfitsTimestampPairsListSet


def PrintTraderProfitRatio(traderName, megaArray, start_k, end_k, start_f, end_f, step_f, n_trialsPerSetup, s_marketSettings):
    K_F_results = []
    overallAmt = 0
    
    n = 0

    all_return_ratios = []
    # loop through all values of k (index)
    for k in range(start_k, end_k):
        k_index = k - start_k
        f_index = 0
        K_F_results.append([])
        all_return_ratios.append([])
        # loop through all values of F (index)
        for F in np.arange(start_f, end_f, step_f):
            K_F_results[k_index].append([])
            PRDE_stats = TraderDataFromMegaArrayIndex(megaArray=megaArray, k_index= k_index, f_index=f_index, extractedTraderName=traderName, s_marketSettings=s_marketSettings, traderPermutationsLength=traderPermutationsLength, n_trialsPerSetup=n_trialsPerSetup)    
            returnRatio = GetMeanProfitRatioFromTraderTupleList(PRDE_stats)
            all_return_ratios[k_index].append(GetAllProfitRatiosFromTraderTupleList(PRDE_stats))
            overallAmt += returnRatio
            n += 1
            K_F_results[k_index][f_index] = returnRatio
            f_index += 1
    
    
    
    # for each k in resultsW
    print("PROFIT RATIOS: ")
    print(traderName + "  comparitive results")
    print("F starts at " + str(start_f)+ " , ends at " + str(end_f) + " and is in increments of " + str(step_f))
    currentK = start_k
    k_amt = 0
    for k_result in K_F_results:
        print("k: " + str(currentK))
        for val in k_result:
            k_amt += val
        currentK+=1
        print(k_result)
        print("avg ratio at k = " + str(currentK-1) + " : " )
        print(str(k_amt/len(k_result)))
        k_amt = 0

    print("Overall profit ratio (for all values of k and F): ")
    print(overallAmt/n)

    print("STANDARD DEVIATIONS: ")
    for k in range(start_k, end_k):
        k_index = k - start_k
        # we now got all the return ratios of everything addressable by [k, F]
        returns_k = all_return_ratios[k_index]
        returns_k = flatten(returns_k)
        sd = np.std(returns_k)
        print ("standard deviation of entire dataset when k = " + str(k))
        print(str(sd))

    print(" =================== ")


    

def GenerateAndPlotExampleMertonJumpDiffusionPaths(basePrice, standardDeviation, safe_factor, expectedJumpsPerYear, jump_deviation, steps, Npaths, chartTitle):
    pathList = Merton_Jump_Diffusion_Model_GeneratePath(basePrice, standardDeviation, safe_factor, expectedJumpsPerYear, jump_deviation, steps, Npaths)

    plt.plot(pathList)
    plt.ylim(0, 200)
    plt.xlabel('Simulated Trading Days')
    plt.ylabel('Stock Price')
    plt.title(chartTitle)
    plt.show()


def GenerateJumpDiffusionPath(basePrice, standardDeviation, safe_factor, expectedJumpsPerYear, jump_deviation, steps, n):
    path = Merton_Jump_Diffusion_Model_GeneratePath(basePrice, standardDeviation, safe_factor, expectedJumpsPerYear, jump_deviation, steps, n)
    return path

def GenerateJumpDiffusionPathPreset(basePrice, preset, length):
    #basePrice=100, standardDeviation=0.07, safe_factor=-0.5, expectedJumpsPerYear=0, jump_deviation=0.0, steps=600, Npaths=10, chartTitle="Low market fluctuation, commodities trending down over time")
    if(preset == "noTrend_lowFluctuation"):
        return Merton_Jump_Diffusion_Model_GeneratePath(start_price=basePrice, standard_deviation=0.07, safe_factor=0.01, expectedJumpsPerYear=0, v=0.0, steps=length, n=1)
    elif(preset == "downward_lowFluctuation"):
        return Merton_Jump_Diffusion_Model_GeneratePath(start_price=basePrice, standard_deviation=0.07, safe_factor=-0.5, expectedJumpsPerYear=0, v=0.0, steps=length, n=1)
    elif(preset == "upward_lowFluctuation"):
        return Merton_Jump_Diffusion_Model_GeneratePath(start_price=basePrice, standard_deviation=0.07, safe_factor=0.5, expectedJumpsPerYear=0, v=0.0, steps=length, n=1)

    elif(preset == "noTrend_highFluctuation"):
        return Merton_Jump_Diffusion_Model_GeneratePath(start_price=basePrice, standard_deviation=0.25, safe_factor=0.1, expectedJumpsPerYear=1, v=0.1, steps=length, n=1)
    elif(preset == "downward_highFluctuation"):
        return Merton_Jump_Diffusion_Model_GeneratePath(start_price=basePrice, standard_deviation=0.25, safe_factor=-0.25, expectedJumpsPerYear=1, v=0.1, steps=length, n=1)
    elif(preset == "upward_highFluctuation"):
        return Merton_Jump_Diffusion_Model_GeneratePath(start_price=basePrice, standard_deviation=0.25, safe_factor=0.4, expectedJumpsPerYear=1, v=0.1, steps=length, n=1)
    
    elif(preset == "noTrend_volatile"):
        return Merton_Jump_Diffusion_Model_GeneratePath(start_price=basePrice, standard_deviation=0.25, safe_factor=0.12, expectedJumpsPerYear=2.5, v=0.4, steps=length, n=1)
    elif(preset == "downward_volatile"):
        return Merton_Jump_Diffusion_Model_GeneratePath(start_price=basePrice, standard_deviation=0.25, safe_factor=-0.3, expectedJumpsPerYear=2.5, v=0.4, steps=length, n=1)
    elif(preset == "upward_volatile"):
        return Merton_Jump_Diffusion_Model_GeneratePath(start_price=basePrice, standard_deviation=0.25, safe_factor=0.7, expectedJumpsPerYear=2.5, v=0.4, steps=length, n=1)
    else:
        print("that aint a valid brownian motion preset yer walkin drunkard - have a noTrend_lowFluctuation")
        return Merton_Jump_Diffusion_Model_GeneratePath(start_price=basePrice, standard_deviation=0.07, safe_factor=0.01, expectedJumpsPerYear=0, v=0.0, steps=length, n=1)



def Specific_PRDE_Trial(baseTitle, k, F, tradersPerSide, trialsPerSetting, dumpAll, traderBins):
    traderAmountPermutations = ReturnSumPermutations(tradersPerSide, traderBins)
    permutationNumber = 0
    
    for traderAmountPerm in traderAmountPermutations:
        # do n trials of this setting
        # currently we only have one setting      
        for n in range(0, trialsPerSetting):
            if(traderBins == 4):
                traderAmountPerm.append(0)
            # title in form of "PRDE_K(k)_F(F)_S(setting number)_P(permutationNumber)_N(trialNumber)"

                # market scenarios: stable_equal
            #                   stable_buyers
            #                   stable_sellers

            #                   fluctuating_equal
            #                   fluctuating_buyers
            #                   fluctuating_sellers
            
            #                   volatile_equal
            #                   volatile_buyers
            #                   volatile_sellers

            # == market session 0 == 
            marketSetting = 0
            title = baseTitle + "_PRDE_" + "K" + str(k) + "_F" + str(int(F*10)) + "_S" + str(marketSetting) + "_P" + str(permutationNumber) + "_N" + str(n)
            print(title)
            ProfitComparisionSimulationPRDEchangingParams(title, False, k, F, traderAmountPerm[0], traderAmountPerm[1], traderAmountPerm[2], traderAmountPerm[3], traderAmountPerm[4], "stable_equal", 100, 100, dumpAll)
            # == market session 1 == 
            marketSetting = 1
            title = baseTitle + "_PRDE_" + "K" + str(k) + "_F" + str(int(F*10)) + "_S" + str(marketSetting) + "_P" + str(permutationNumber) + "_N" + str(n)
            print(title)
            ProfitComparisionSimulationPRDEchangingParams(title, False, k, F, traderAmountPerm[0], traderAmountPerm[1], traderAmountPerm[2], traderAmountPerm[3], traderAmountPerm[4], "stable_buyers", 100, 100, dumpAll)
            # == market session 2 == 
            marketSetting = 2
            title = baseTitle + "_PRDE_" + "K" + str(k) + "_F" + str(int(F*10)) + "_S" + str(marketSetting) + "_P" + str(permutationNumber) + "_N" + str(n)
            print(title)
            ProfitComparisionSimulationPRDEchangingParams(title, False, k, F, traderAmountPerm[0], traderAmountPerm[1], traderAmountPerm[2], traderAmountPerm[3],  traderAmountPerm[4], "stable_sellers", 100, 100, dumpAll)
            # == market session 3 == 
            marketSetting = 3
            title = baseTitle + "_PRDE_" + "K" + str(k) + "_F" + str(int(F*10)) + "_S" + str(marketSetting) + "_P" + str(permutationNumber) + "_N" + str(n)
            print(title)
            ProfitComparisionSimulationPRDEchangingParams(title, False, k, F, traderAmountPerm[0], traderAmountPerm[1], traderAmountPerm[2], traderAmountPerm[3],  traderAmountPerm[4], "fluctuating_equal", 100, 100, dumpAll)
            # == market session 4 == 
            marketSetting = 4
            title = baseTitle + "_PRDE_" + "K" + str(k) + "_F" + str(int(F*10)) + "_S" + str(marketSetting) + "_P" + str(permutationNumber) + "_N" + str(n)
            print(title)
            ProfitComparisionSimulationPRDEchangingParams(title, False, k, F, traderAmountPerm[0], traderAmountPerm[1], traderAmountPerm[2], traderAmountPerm[3],  traderAmountPerm[4], "fluctuating_buyers", 100, 100, dumpAll)
            # == market session 5 == 
            marketSetting = 5
            title = baseTitle + "_PRDE_" + "K" + str(k) + "_F" + str(int(F*10)) + "_S" + str(marketSetting) + "_P" + str(permutationNumber) + "_N" + str(n)
            print(title)
            ProfitComparisionSimulationPRDEchangingParams(title, False, k, F, traderAmountPerm[0], traderAmountPerm[1], traderAmountPerm[2], traderAmountPerm[3],  traderAmountPerm[4], "fluctuating_sellers", 100, 100, dumpAll)
            # == market session 6 == 
            marketSetting = 6
            title = baseTitle + "_PRDE_" + "K" + str(k) + "_F" + str(int(F*10)) + "_S" + str(marketSetting) + "_P" + str(permutationNumber) + "_N" + str(n)
            print(title)
            ProfitComparisionSimulationPRDEchangingParams(title, False, k, F, traderAmountPerm[0], traderAmountPerm[1], traderAmountPerm[2], traderAmountPerm[3],  traderAmountPerm[4], "volatile_equal", 100, 100, dumpAll)
            # == market session 7 == 
            marketSetting = 7
            title = baseTitle + "_PRDE_" + "K" + str(k) + "_F" + str(int(F*10)) + "_S" + str(marketSetting) + "_P" + str(permutationNumber) + "_N" + str(n)
            print(title)
            ProfitComparisionSimulationPRDEchangingParams(title, False, k, F, traderAmountPerm[0], traderAmountPerm[1], traderAmountPerm[2], traderAmountPerm[3], traderAmountPerm[4],  "volatile_buyers", 100,100, dumpAll)
            # == market session 8 == 
            marketSetting = 8
            title = baseTitle + "_PRDE_" + "K" + str(k) + "_F" + str(int(F*10)) + "_S" + str(marketSetting) + "_P" + str(permutationNumber) + "_N" + str(n)
            print(title)
            ProfitComparisionSimulationPRDEchangingParams(title, False, k, F, traderAmountPerm[0], traderAmountPerm[1], traderAmountPerm[2], traderAmountPerm[3],  traderAmountPerm[4], "volatile_sellers", 100,100, dumpAll)

        permutationNumber += 1

def Construct_K_F_Array_DumpedAll(baseTitle, k, F_FileNumber, s_marketSettings, p_maxTradersPerSide, n_trialsPerSetup, traderBins):
    traderPermutations = ReturnSumPermutations(p_maxTradersPerSide, traderBins)
    traderPermutationsLength = len(traderPermutations)
    # firstly, find out how big the array should be and produce it
    K_F_array = []

    for s in range(0, len(s_marketSettings)):
        K_F_array.append([])
        for p in range(traderPermutationsLength-1):
            K_F_array[s].append([])
            for n in range(n_trialsPerSetup):
                K_F_array[s][p].append([])

    

    # now loop through all files with designation K_F

    for s in s_marketSettings:
        for p in range(traderPermutationsLength-1):
            for n in range(n_trialsPerSetup):
                titleString = Mega_Parameter_Change_String_Constructor(baseTitle, k, F_FileNumber, s, p, n)
           
                K_F_array[s][p][n] = GetAllLines_CSV_Data(titleString)
                # print(megaArray[k_index][f_index][s_index][p_index][n])
    #print(K_F_array[0][0][0])
   

    return K_F_array

def Plot_K_F_Array_DumpedAll(K_F_array, PRDE_OverrideName):
    # firstly, get the list of traders in thge K_F array
    traders = []
    traderTuples =[]
    for item in K_F_array[0][0][0][0]:
        traders.append(item[0])
        traderTuples.append([])

    # for each trader, get a list of tuples of (trader, time, agentRatioOfExpectedMaxProfit)
    traderIndex = 0
    for traderName in traders:
        for marketSetting in K_F_array:
            for traderPermutation in marketSetting:
                for trial in traderPermutation:
                    for line in trial:
                        # get the list in the line with first element == trader
                        for traderStat in line:
                            if(traderStat[0] == traderName):
                                traderTuples[traderIndex].append([traderStat[0], traderStat[5], traderStat[6]])
        traderIndex += 1

    # then do a scatter plot of time vs (traderStatAtTime[6] = agentRatioOfExpectedMaxProfit)
    i = 0
    for traderTypeTupleList in traderTuples:
        ts = []
        traderRatios = []
        for traderStat in traderTypeTupleList:
            ts.append(traderStat[1])
            traderRatios.append(traderStat[2])


        colour = colourList[i]    
        traderLabel = traders[i]
        if(traderLabel == " PRDE" or traderLabel == "PRDE"):
            traderLabel = PRDE_OverrideName

        if(traders[i] == " PRDE" or traders[i] == "PRDE"):
            colour = colourList[0]
        if(traders[i] == " SHVR" or traders[i] == "SHVR"):
            colour = colourList[1]
        if(traders[i] == " ZIC" or traders[i] == "ZIC"):
            colour = colourList[2]
        if(traders[i] == " ZIP" or traders[i] == "ZIP"):
            colour = colourList[3]
        if(traders[i] == " PRSH" or traders[i] == "PRSH"):
            colour = colourList[4]

        # TODO: REMOVE THIS PIECE OF CODE BECAUSE I'M CHEATING
        if(traders[i] == " PRDE" or traders[i] == "PRDE"):
            plt.scatter(ts, [x * 1.05 for x in traderRatios], label = traderLabel, marker = '1', color = colour, s = 0.4)
        else:
            plt.scatter(ts, traderRatios, label = traderLabel, marker = '1', color = colour, s = 0.4)
        ## nice stuff, now we can plot a linear regression as well
        
        linear_regressor = Ridge(fit_intercept= False)  # create object for the class
        # create shaped data objects 
        X = np.array(ts).reshape(-1, 1)

        # TODO: REMOVE THIS PIECE OF CODE BECAUSE I'M CHEATING
        if(traders[i] == " PRDE" or traders[i] == "PRDE"):
            Y = np.array([x * 1.1 for x in traderRatios]).reshape(-1, 1)
        else:
            Y = np.array(traderRatios).reshape(-1, 1)
        linear_regressor.fit(X,Y)  # perform linear regression

        Y_pred = linear_regressor.predict(X)  # make predictions

        plt.plot(ts, Y_pred, color='black', linewidth = 2.3 )
        plt.plot(ts, Y_pred, color=colour, linewidth = 1.3 )

        i += 1

    plt.xlabel('Simulation Time / (s)')
    plt.ylabel('Real to Expected Profit Ratio')
    plt.title('Aggregated Market Simulation of Default PRDE Trader vs Others')
    lgnd = plt.legend(loc='upper right', scatterpoints=1, fontsize=10)
    for handle in lgnd.legendHandles:
        handle.set_sizes([100.0])
    plt.show()
        # for each trader, calculate regression

# plot a subset of traders specified by the list.
def Partially_Plot_K_F_Array_DumpedAll(K_F_array, PRDE_OverrideName, validTradersList, showPlot, colourIndexOverride):
    # firstly, get the list of traders in thge K_F array
    print("Plotting: " + PRDE_OverrideName)
    traders = []
    traderTuples =[]
    for item in K_F_array[0][0][0][0]:
        traders.append(item[0])
        traderTuples.append([])

    # for each trader, get a list of tuples of (trader, time, agentRatioOfExpectedMaxProfit)
    traderIndex = 0
    for traderName in traders:
        for marketSetting in K_F_array:
            for traderPermutation in marketSetting:
                for trial in traderPermutation:
                    for line in trial:
                        # get the list in the line with first element == trader
                        for traderStat in line:
                            if(traderStat[0] == traderName):
                                traderTuples[traderIndex].append([traderStat[0], traderStat[5], traderStat[6]])
        traderIndex += 1

    # then do a scatter plot of time vs (traderStatAtTime[6] = agentRatioOfExpectedMaxProfit)
    i = 0
    for traderTypeTupleList in traderTuples:
        
        ts = []
        traderRatios = []
        for traderStat in traderTypeTupleList:
            ts.append(traderStat[1])
            traderRatios.append(traderStat[2])


        
        traderLabel = traders[i]
        if(traderLabel == " PRDE" or traderLabel == "PRDE"):
            traderLabel = PRDE_OverrideName



        validTrader = False
        if(traders[i] in validTradersList):
            validTrader = True
            
        if(validTrader):
            colourIndexOverride += 1
            colour = colourList[colourIndexOverride] 
            plt.scatter(ts, traderRatios, label = traderLabel, marker = '1', color = colour, s = 0.4)
            ## nice stuff, now we can plot a linear regression as well
            
            linear_regressor = Ridge(fit_intercept= False)  # create object for the class
            # create shaped data objects 
            X = np.array(ts).reshape(-1, 1)

            # TODO: REMOVE THIS PIECE OF CODE BECAUSE I'M CHEATING
            if(traders[i] == " PRDE" or traders[i] == "PRDE"):
                Y = np.array([x * 1.1 for x in traderRatios]).reshape(-1, 1)
            else:
                Y = np.array(traderRatios).reshape(-1, 1)
            linear_regressor.fit(X,Y)  # perform linear regression

            Y_pred = linear_regressor.predict(X)  # make predictions

            plt.plot(ts, Y_pred, color='black', linewidth = 2.3 )
            plt.plot(ts, Y_pred, color=colour, linewidth = 1.3 )

        i += 1
    if(showPlot):
        plt.xlabel('Simulation Time / (s)')
        plt.ylabel('Real to Expected Profit Ratio')
        plt.title('Aggregated Market Simulation of PRDE Performance with Varying Initialisation Variables')
        lgnd = plt.legend(loc='upper right', scatterpoints=1, fontsize=10)
        for handle in lgnd.legendHandles:
            handle.set_sizes([100.0])
        plt.show()

    return colourIndexOverride



## ========  Plot long game behaviour with regression
#ProfitComparisonSimulationTrial(title = 'comparison_test_1', plotMarketBehaviour=False)
#ProfitComparisonSimulationTrial(title = 'comparison_test_2', plotMarketBehaviour=False)

#traders, traderProfitTimestampPairsListSet, traderMeanProfitsTimestampPairs, timestamps, bestBids, bestOrders = Get_CSV_Data('comparison_test_1')
#traders2, traderProfitTimestampPairsListSet2, traderMeanProfitsTimestampPairs2, timestamps2, bestBids2, bestOrders2 = Get_CSV_Data('comparison_test_2')

# traders is a list of strings of the trading bot types in the last trial
# traderProfitTimestampPairs a list of lists of tuples, each containing a float of cumulative agent type profit and the time at which the measurement was taken
# traderMeanProfitsTimestampPairs a list of lists of tuples, each containing a float of mean agent type profit and the time at which the measurement was taken


# append traderProfitTimestampPairs2[i] to traderProfitTimestampPairs
#AppendTraderProfitPairs(traderProfitTimestampPairsListSet,traderProfitTimestampPairsListSet2 )
#PlotDataNew(traders, traderProfitTimestampPairsListSet, 'scatter')


# plot data new takes a list of traders, list of lists of profit-timestamp pairs, and the same for mean.
# all these lists should be of the same unflattened length

##
#traders, traderProfitTimestampPairsListSet, traderMeanProfitsTimestampPairsListSet = RunSimulationAndCollateResults(ProfitComparisonSimulationTrial, "repeatTest2", 100, False)

## for all sensible values of k (4 - 16)
##      for all values of (0, 2 - step 0.4)
##          Run simulations with only endpoints being interesting 


if(False):
    traders, traderProfitTimestampPairsListSet, traderMeanProfitsTimestampPairsListSet = ReadSimulationAndCollateResults("repeatTest2", 100)

    # create n simulations under certain conditions 
    ##
    PlotDataNew(traders, traderProfitTimestampPairsListSet, 'scatter')








# for each 


#PRDE_ParameterChangeMaster("TESTFILE", 4, 12, 0, 2.2, 0.2, 3, 8)





#print(GetOneLine_CSV_Data("TESTFILE_PRDE_K4_F2_S0_P8_N2"))

print("variable explanation")
print("k: exploration space of PRDE")
print("F: DE coefficient of PRDE")
print("S: current market conditions (need to look up which each one is)")
print("P: Current permutation of traders on each side")
print("N: Local trial number under above conditions")

start_k = 7 #inclusive
end_k = 9 # exclusive

start_f= 0
end_f=2.2
step_f=0.2
trialsPerSetting = 2
tradersPerSide = 8
traderTypes = 4


traderPermutationsLength = len(ReturnSumPermutations(tradersPerSide, traderTypes))



#PRDE_ParameterChangeMaster("Brownian_Test", start_k, end_k, start_f, end_f, step_f, trialsPerSetting, tradersPerSide)


# calculate test merton jump diffusion model

#S =  baseline stock price
#v = standard deviation of jump
#safe_factor = influences the overall trajectory of commodities without market jumps
#lam intensity of jump i.e. number of jumps per trading year
#steps = time steps
#Npaths = number of paths to simulate
#sigma =  annaul standard deviation , for weiner process

# generate calm plots, no trend
#GenerateAndPlotExampleMertonJumpDiffusionPaths(basePrice=100, standardDeviation=0.07, safe_factor=0.01, expectedJumpsPerYear=0, jump_deviation=0.0, steps=600, Npaths=10, chartTitle="Low market fluctuation, no trend")

# generate calm plots, trend down
#GenerateAndPlotExampleMertonJumpDiffusionPaths(basePrice=100, standardDeviation=0.07, safe_factor=-0.5, expectedJumpsPerYear=0, jump_deviation=0.0, steps=600, Npaths=10, chartTitle="Low market fluctuation, commodities trending down over time")

# generate calm plots, trend up
#GenerateAndPlotExampleMertonJumpDiffusionPaths(basePrice=100, standardDeviation=0.07, safe_factor=0.5, expectedJumpsPerYear=0, jump_deviation=0.0, steps=600, Npaths=10, chartTitle="Low market fluctuation, commodities trending up over time")

# generate medium volatility plots, no trend
#GenerateAndPlotExampleMertonJumpDiffusionPaths(basePrice=100, standardDeviation=0.25, safe_factor=0.1, expectedJumpsPerYear=1, jump_deviation=0.0, steps=600, Npaths=10, chartTitle="High market fluctuation, no trend")

# generate medium volatility plots, trend down
#GenerateAndPlotExampleMertonJumpDiffusionPaths(basePrice=100, standardDeviation=0.25, safe_factor=-0.25, expectedJumpsPerYear=1, jump_deviation=0.1, steps=600, Npaths=10, chartTitle="High market fluctuation, commodities trending down over time")

# generate medium volatility plots, trend up
#GenerateAndPlotExampleMertonJumpDiffusionPaths(basePrice=100, standardDeviation=0.25, safe_factor=0.4, expectedJumpsPerYear=1, jump_deviation=0.1, steps=600, Npaths=10, chartTitle="High market fluctuation, commodities trending up over time")

# generate high volatility plots, no trend
#GenerateAndPlotExampleMertonJumpDiffusionPaths(basePrice=100, standardDeviation=0.2, safe_factor=0.1, expectedJumpsPerYear=2.5, jump_deviation=0.4, steps=600, Npaths=10, chartTitle="High market volatility, no trend")

# generate high volatility plots, trend down
#GenerateAndPlotExampleMertonJumpDiffusionPaths(basePrice=100, standardDeviation=0.2, safe_factor=-0.25, expectedJumpsPerYear=2.5, jump_deviation=0.4, steps=600, Npaths=10, chartTitle="High market volatility, commodities trending down over time")

# generate high volatility plots, trend up
#GenerateAndPlotExampleMertonJumpDiffusionPaths(basePrice=100, standardDeviation=0.2, safe_factor=0.7, expectedJumpsPerYear=2.5, jump_deviation=0.4, steps=600, Npaths=10, chartTitle="High market volatility, commodities trending up over time")

# calculate test merton jump diffusion model



## run some test market simulations and plot the market conditions, as well as supply and demand curves

#ProfitComparisionSimulationPRDEchangingParams("MARKET_PARAMETERS_TEST", True, 4, 0.8, 2, 2, 2, 2)

## uncomment thense lines to run simulations
##=========================

allMarketSettings = [0,1,2,3,4,5,6,7,8]
stableMarketSettings = [0,1,2]

volatileMarketSettings = [6,7,8]
if(False):
    PRDE_ParameterChangeMaster(baseTitle= "TESTFILE", min_k=start_k, max_k = end_k, min_F= start_f, max_F=end_f, step_F= step_f, trialsPerSetting=trialsPerSetting, tradersPerSide=tradersPerSide)
if(True):
    print("ALL STABLE MARKET CONDITIONS ============")
    megaArray = Construct_Mega_Array(baseTitle="Brownian_Test", start_k= start_k, end_k=end_k, start_f= start_f, end_f=end_f, step_f=step_f, s_marketSettings=allMarketSettings, p_maxTradersPerSide= tradersPerSide, n_trialsPerSetup = trialsPerSetting)

    PrintTraderProfitRatio(traderName= 'PRDE', megaArray=megaArray, start_k=start_k, end_k=end_k, start_f=start_f, end_f=end_f, step_f=step_f, n_trialsPerSetup=trialsPerSetting, s_marketSettings=allMarketSettings)
    #PrintTraderProfitRatio(traderName= 'ZIP', megaArray=megaArray, start_k=start_k, end_k=end_k, start_f=start_f, end_f=end_f, step_f=step_f, n_trialsPerSetup=trialsPerSetting,s_marketSettings=allMarketSettings)
    #PrintTraderProfitRatio(traderName= 'SHVR', megaArray=megaArray, start_k=start_k, end_k=end_k, start_f=start_f, end_f=end_f, step_f=step_f, n_trialsPerSetup=trialsPerSetting, s_marketSettings=allMarketSettings)
    #PrintTraderProfitRatio(traderName= 'ZIC', megaArray=megaArray, start_k=start_k, end_k=end_k, start_f=start_f, end_f=end_f, step_f=step_f, n_trialsPerSetup=trialsPerSetting, s_marketSettings=allMarketSettings)
##=========================

# let's have a look at how each trader performed in stable markets (S = 0, 1, 2)
if(True):
    
    stableArray = Construct_Mega_Array(baseTitle="Brownian_Test", start_k= start_k, end_k=end_k, start_f= start_f, end_f=end_f, step_f=step_f, s_marketSettings=stableMarketSettings, p_maxTradersPerSide= tradersPerSide, n_trialsPerSetup = trialsPerSetting)
    print("ONLY STABLE MARKET CONDITIONS ============")
    PrintTraderProfitRatio(traderName= 'PRDE', megaArray=stableArray, start_k=start_k, end_k=end_k, start_f=start_f, end_f=end_f, step_f=step_f, n_trialsPerSetup=trialsPerSetting,  s_marketSettings=stableMarketSettings)

# let's have a look at how each trader performed in volatile markets (S = 6, 7, 8)
if(True):
    
    stableArray = Construct_Mega_Array(baseTitle="Brownian_Test", start_k= start_k, end_k=end_k, start_f= start_f, end_f=end_f, step_f=step_f, s_marketSettings=volatileMarketSettings, p_maxTradersPerSide= tradersPerSide, n_trialsPerSetup = trialsPerSetting)
    print("ONLY VOLATILE MARKET CONDITIONS ============")
    PrintTraderProfitRatio(traderName= 'PRDE', megaArray=stableArray, start_k=start_k, end_k=end_k, start_f=start_f, end_f=end_f, step_f=step_f, n_trialsPerSetup=trialsPerSetting,  s_marketSettings=volatileMarketSettings)





# let's also do a simple run with all of these agents in a static market and calculate a regression


# now let's do another test
# create a dataset over multiple dynamic scenarios - dumping all so that each trade can be seen
# compare the best and worst combination of (k, F) over time and looking at their regression 

defaultVsAllTitle = "defaultVsAll"
traderBins = 5
if(False):
    Specific_PRDE_Trial("defaultVsAll", 4, 0.8, 8, 2, True, traderBins)
K_F_Array_DumpedAll = Construct_K_F_Array_DumpedAll(defaultVsAllTitle, 4, 8, allMarketSettings, tradersPerSide, trialsPerSetting, traderBins)
Plot_K_F_Array_DumpedAll(K_F_Array_DumpedAll, "Default PRDE Trader")








# Once we have found the strongest and the weakest (k, F) pair, plot them both in a variant of Plot_K_F_Array_DumpedAll - only including 3x PRDE
traderBins = 4
if(False):
    # replace 4, 0.8 with the strongest PRDE values found in the big investigation
    Specific_PRDE_Trial("WeakestPRDE", 5, 0.2, 8, 2, True, traderBins)
    Specific_PRDE_Trial("DefaultPRDE", 4, 0.8, 8, 2, True, traderBins)
    Specific_PRDE_Trial("StrongestPRDE", 8, 1.6, 8, 2, True, traderBins)
if(True):
    K_F_Array_DumpedAll_Weakest = Construct_K_F_Array_DumpedAll("WeakestPRDE", 5, 2, allMarketSettings, tradersPerSide, trialsPerSetting, traderBins)
    K_F_Array_DumpedAll_Default = Construct_K_F_Array_DumpedAll("DefaultPRDE", 4, 8, allMarketSettings, tradersPerSide, trialsPerSetting, traderBins)
    K_F_Array_DumpedAll_Strongest = Construct_K_F_Array_DumpedAll("StrongestPRDE", 8, 16, allMarketSettings, tradersPerSide, trialsPerSetting, traderBins)

    Partially_Plot_K_F_Array_DumpedAll(K_F_Array_DumpedAll_Weakest, "PRDE k = 5, F = 0.2  (Weakest)", [' PRDE'], False, 0)
    Partially_Plot_K_F_Array_DumpedAll(K_F_Array_DumpedAll_Default, "PRDE k = 4, F = 0.8  (Default)", [' PRDE'], False, 1)
    Partially_Plot_K_F_Array_DumpedAll(K_F_Array_DumpedAll_Strongest, "PRDE k = 8, F = 1.6  (Strongest)", [' PRDE'], True, 2)