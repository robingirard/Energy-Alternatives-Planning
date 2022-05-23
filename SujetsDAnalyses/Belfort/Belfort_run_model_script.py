from Belfort_model import *

for year in [2030,2040,2050,2060]:
    for bati_hyp in ['ref','SNBC']:
        for reindus in [True,False]:
            for mix in ['nuclear_plus','nuclear_minus','100_enr']:
                if year!=2030 or mix=='nuclear_plus':
                    print("\nRun scenario year={} bati_hyp={} reindus={} and mix={}".format(year,bati_hyp,reindus,mix))
                    run_model_multinode(year, bati_hyp, reindus, mix)
