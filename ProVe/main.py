import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import time
from netver.utils.colors import bcolors
from netver.verifier import NetVer

# safety properties

# safety property for toy DNN model_2_68
property={'type':'positive', 'P': [[0.0, 1.0], [0.0, 1.0]]}

# safety property for realistic DRL mapless navigation Fixed_obs_NT_PPO_1
#property={'type':'decision', 'P': [[0.98, 1.0], [0.98, 1.0], [0.98, 1.0], [0.98, 1.0], [0.04, 0.05], [0.04, 0.05], [0.04, 0.05], [0.95, 1.0], [0.95, 1.0], [0.95, 1.0], [0.95, 1.0], [-1, 1.], [0, 1.]], 'A': 4}

#safety property for realistic DRL mapless navigation REI_id73, REI_id114 and PPO_id165
# property = {
#     "type" : "decision",
#     "P" :  [[ 0.95,  1.0],
# 			[0.95,  1.0],
# 			[0.95,  1.0],
#             [0.135,  0.185],
#             [0.95,  1.0],
#             [0.95,  1.0],
#             [0.95,  1.0],
#             [0.0,  1.0],
#             [0.2,  1.0]],
#     "A": 0
# }


# safety property ACAS XU FV benchmark
# property = {
#     "type" : "decision",
#     "name": "property2",
#     "P" :  [[ 0.600000,  0.679858],
# 			[-0.500000,  0.500000],
# 			[-0.500000,  0.500000],
# 			[ 0.450000,  0.500000],
# 			[-0.500000, -0.450000]],
#     "A": 0
# }




def run_prover(propagation, model_str, property):
    # load your model
    model = tf.keras.models.load_model(model_str, compile=False)

    # ProVe hyperparameters
    method = "ProVe"            # select the method: for the formal analysis select "ProVe" otherwise "estimated"
    discretization = 3          # how many digitals consider for the computation
    CPU = False                 # select the hardware for the formal analysis
    verbose = 1                 # whether to print info abount the computation of ProVe
    cloud = 3000000             # number of random state to sample for the approximate verification i.e., the "estimated" analysis

    memory_limit = 0            # maximum threshold for virtual memory usage (indicate 0 to use the available free memory)
    disk_limit = 0              # maximum threshold for disk usage (indicate 0 to use the available free disk space)

    if method == 'ProVe':
        netver = NetVer(method, model, property, memory_limit=memory_limit, disk_limit=disk_limit,
                        rounding=discretization, cpu_only=CPU, interval_propagation=propagation, reversed=False)
    else:
        netver = NetVer("estimated", model, property, cloud_size=cloud)

    print(bcolors.OKCYAN + '\n\t#################################################################################' + bcolors.ENDC)
    print(bcolors.OKCYAN + '\t\t\t\t\tProVe hyperparameters:\t\t\t\t' + bcolors.ENDC)
    if method == 'ProVe':  
        print(bcolors.OKCYAN + f'\t method=formal, GPU={not CPU}, interval_propagation={propagation}, rounding={discretization}, verbose={verbose}\t' + bcolors.ENDC)
    else:
        print(bcolors.OKCYAN + f'\t\t\t\tmethod=estimated, cloud_size={cloud}' + bcolors.ENDC)
    print(bcolors.OKCYAN + '\t#################################################################################'+ bcolors.ENDC)

    start = time.time()
    approx = NetVer("estimated", model, property, cloud_size=cloud).run_verifier(start, verbose=0)[1]['violation_rate']
    print(bcolors.BOLD + bcolors.WARNING + f"\n\tEstimated VR is {round(approx, 4)}%\n" + bcolors.ENDC)
    
   
    start = time.time()
    sat, info = netver.run_verifier(start, verbose, approx)
    end = time.time()

    time_execution = round((end-start)/60,5)
    violation_rate = round(info['violation_rate'], 3)

  
    if sat:
        print(bcolors.OKGREEN + "\nThe property is SAT!")
        print(f"\tTime execution: {time_execution} min\n"+ bcolors.ENDC)
    else:
        print(bcolors.BOLD + bcolors.FAIL + "\nThe property is UNSAT!"+ bcolors.ENDC)
        print(bcolors.BOLD + bcolors.FAIL + "\t"f"Violation rate: {violation_rate}%"+ bcolors.ENDC)
        print(f"\tTime execution: {time_execution} min\n" )

    

if __name__ == "__main__":

    # define model to run #DNN-Verification
    model_name = "model_2_68.h5"
    model_path = "./models"

    #run_prover("naive", f"{model_path}/{model_name}", property)
    #run_prover("symbolic", f"{model_path}/{model_name}", property)
    run_prover("relaxation", f"{model_path}/{model_name}", property)
