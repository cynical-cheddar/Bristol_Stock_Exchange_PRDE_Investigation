

# default schedule_offsetfn - currently unused due to being replaced with merton jump dynamic markets
def schedule_offsetfn(t):
    pi2 = math.pi * 2
    c = math.pi*3000
    wavelength = t/c
    gradient = 100*t/(c/pi2)
    amplitude = 100*t/(c/pi2)
    offset = gradient + amplitude * math.sin(wavelength*t)
    return int(round(offset,0))

# ProfitComparisionSimulationPRDEchangingParams is called for each iteration of (k, F)
# called for permutation of traders under each market condition for n local trials

def ProfitComparisionSimulationPRDEchangingParams(title, plotMarketBehaviour, k, F, PRDE_amt, SHVR_amt, ZIC_amt, ZIP_amt, marketCondition, baseSupplyPrice, baseDemandPrice):
    trial_id = title
    start_time = 0
    end_time = 600
    print("ProfitComparisionSimulationPRDEchangingParams")
    # generate a varying brownian function of supply

    # ==========TODO============ - finish implementation of Merton-Jump dynamic price schedule
    global mertonSupplyPath
    global mertonDemandPath
    if(marketCondition == "Low_Steady"):
        print("low steady market")
        mertonSupplyPath = GenerateJumpDiffusionPath(basePrice=baseSupplyPrice, standardDeviation=0.07, safe_factor=0.01, expectedJumpsPerYear=0, jump_deviation=0.0, steps=650, n = 1)
        mertonDemandPath = GenerateJumpDiffusionPath(basePrice=baseDemandPrice, standardDeviation=0.07, safe_factor=0.01, expectedJumpsPerYear=0, jump_deviation=0.0, steps=650, n = 1)
    # ======================================== ##
    sup_range1 = (10, 190, schedule_offsetfn)
    dem_range1 = (10, 190, schedule_offsetfn)

    sup_range2 = (200, 300, schedule_offsetfn)
    dem_range2 = (200, 300, schedule_offsetfn)

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



    PRDR_Setup_F_dict =  {'de_state': 'active_s0',          # initial state: strategy 0 is active (being evaluated)
                         's0_index': 0,    # s0 starts out as active strat
                         'snew_index': k,             # (k+1)th item of strategy list is DE's new strategy
                         'snew_stratval': None,            # assigned later
                         'F': F                          # differential weight -- usually between 0 and 2
        }

    buyers_spec = [('PRDE', PRDE_amt, {'k': k, 's_min': -1.0, 's_max': +1.0, 'diffVolDict':PRDR_Setup_F_dict}),('SHVR',SHVR_amt), ('ZIC', ZIC_amt), ('ZIP', ZIP_amt)]

    sellers_spec = buyers_spec
    traders_spec = {'sellers':sellers_spec, 'buyers':buyers_spec}

    tdump=open((title) + '.csv','w')
    dump_all = False
    verbose = False
    market_session(trial_id, start_time, end_time, traders_spec, order_schedule, tdump, dump_all, verbose)
    tdump.close()
    if(plotMarketBehaviour):
        plot_sup_dem(8, [sup_range1, sup_range2, sup_range1], 8, [dem_range1, dem_range2, dem_range1], 'fixed')
        plot_trades(trial_id)
