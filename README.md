# Neural-Network-Verifier
A set of methods for the Formal and Approximate Verification and analysis of Neural Networks, implemented in Python


![Alt text](https://github.com/GabrieleRoncolato/SymbolicPropagationCUDA/blob/master/images/overview.png)
****


## Dependencies: 
    - Tensorflow 2.16.1
    - numpy 1.26.4
    - cupy 13.1.0
    - psutil 5.9.8
    - cmake

## Definition of the properties
Properties can be defined with 3 different formulations:

### PQ
Following the definition of Marabou [Katz et al.], given an input property P and an output property Q, the property is verified if for each *x* in P, it follows that N(x) is in Q *(i.e., if the input belongs to the interval P, the output must belong to the interval Q)*.
```python
property = {
	"type" : "PQ",
	"P" : [[0.1, 0.34531], [0.7, 1.1]],
	"Q" : [[0.0, 0.2], [0.0, 0.2]]
}
```

### Decision
Following the definition of ProVe [Corsi et al.], given an input property P and an output node A corresponding to an action, the property is verified if, for each *x* in P, it follows that the action A will never be selected *(i.e., if the input belongs to the interval P, the output of node A is never the one with the highest value)*.
```python
property = {
	"type" : "decision",
	"P" : [[0.1, 0.3], [0.7, 1.1]],
	"A" : 1
}
```

### Positive
Following the definition of α,β-CROWN [Wang et al.], given an input property P, the output of the network is non-negative *(i.e., if the input belongs to the interval P, the output of each node is greater or equals zero)*
```python
property = {
	"type": "positive",
	"P" : [[0.1, 0.3], [0.7, 1.1]]
}
```


## Parameters
ProVe will use the default parameters for the formal analysis. You can change all the parameters when creating the NetVer object as follows. 
Follow a list of the available parameters (with the default value):

```python
# Common to all the algorithms
time_out_cycle = 35         # timeout on the number of cycles
time_out_checked = 0        # timeout on the checked area. If the unproved area is less than this value, the algorithm stops returning the residual as a violation
rounding = None             # rounding value for the input domain (P)

method = "ProVe"            # select the method: for the formal analysis, select "ProVe" otherwise, "estimated"

propagation= ["naive", "symbolic", "relaxation"] # Select the way to perform the interval propagation: "naive" with simple Moore algebra,
						 #  "symbolic" for the method proposed in (Formal Security Analysis of Neural Networks using Symbolic Intervals, Wang et al. 2018)
						 #    and finally, "relaxation" for the method proposed in (Efficient Formal Safety Analysis of Neural Networks, Wang et al. 2018)  

discretization = 3          # how many digitals are considered for the computation
CPU = False                 # Select the hardware for the formal analysis
verbose = 1                 # whether to print info about the computation of ProVe
cloud = 3000000             # number of random states to sample for the approximate verification i.e., the "estimated" analysis
memory_limit = 0            # maximum threshold for virtual memory usage (indicate 0 to use the available free memory)
disk_limit = 0              # maximum threshold for disk usage (indicate 0 to use the available free disk space)
```
Please note that, due to the #P-hardness of the problem, performing the exact count/enumeration of all the portions of the property's domain that violate a safety property could require some time and, in general, is subject to strong scalability issues. To address such an issue, in the provided code, we furnish the possibility of using both the RAM memory and the disk memory in order to complete the verification process without incurring memory errors. We refer the interested reader to (C) for further details.

## Results of the analysis
The analysis returns two values, SAT and info. *SAT* is true if the property is respected, UNSAT otherwise; *value* is a dictionary that contains different values based on the used algorithm:

- counter_example: a counter-example that falsifies the property 
- violation_rate: the violation rate, i.e., the percentage of the unsafe volume of the property's domain 
- exit_reason: reason for an anticipate exit *(usually timeout)*



## Example Code
To run the example code, use *example.py* from the main folder. For the time being, we only support Keras models.
```python
import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import time
from netver.verifier import NetVer

def get_actor_model( ):
	inputs = tf.keras.layers.Input(shape=(2,))
	hidden_0 = tf.keras.layers.Dense(32, activation='relu', bias_initializer='random_normal')(inputs)
	hidden_1 = tf.keras.layers.Dense(32, activation='relu', bias_initializer='random_normal')(hidden_0)
	outputs = tf.keras.layers.Dense(5, activation='linear')(hidden_1)

	return tf.keras.Model(inputs, outputs)

if __name__ == "__main__":
	model = get_actor_model()

	property = {
		"type": "decision",
		"P": [[0.1, 0.3], [0.7, 1.1]],
		"A": 1
	}


	approx = NetVer("estimated", model, property, cloud_size=1000000).run_verifier(time.time(), verbose=0)[1]['violation_rate']
	print(f"\n\tEstimated VR is {round(approx, 4)}%\n")

	netver = NetVer("ProVe", model, property, memory_limit=0, disk_limit=0,rounding=3, cpu_only=False, interval_propagation="relaxation")
	sat, info = netver.run_verifier(start_time=time.time(), verbose=1, estimation=approx)
	print( f"\nThe property is SAT? {sat}" )
	print( f"\tViolation rate: {info['violation_rate']}\n" )
```


## Contributors
*  **Gabriele Roncolato** - gabriele.roncolato@studenti.univr.it
*  **Luca Marzari** - luca.marzari@univr.it
*  **Davide Corsi** - davide.corsi@univr.it

## Reference
If you use our verifier in your work, please kindly cite our papers:

**(A)** [Formal verification of neural networks for safety-critical tasks in deep reinforcement learning](https://proceedings.mlr.press/v161/corsi21a.html) Corsi D., Marchesini E., and Farinelli A. UAI, 2021
```
@inproceedings{corsi2021formal,
  title={Formal verification of neural networks for safety-critical tasks in deep reinforcement learning},
  author={Corsi, Davide and Marchesini, Enrico and Farinelli, Alessandro},
  booktitle={Uncertainty in Artificial Intelligence},
  pages={333--343},
  year={2021},
  organization={PMLR}
}
```
    
**(B)** [The \#DNN-Verification Problem: Counting Unsafe Inputs for Deep Neural Networks](https://dl.acm.org/doi/abs/10.24963/ijcai.2023/25).  Marzari* L., Corsi* D., Cicalese F and Farinelli A. IJCAI, 2023
```
@inproceedings{marzari2023dnn,
  title={The \#DNN-Verification Problem: Counting Unsafe Inputs for Deep Neural Networks},
  author={Marzari, Luca and Corsi, Davide and Cicalese, Ferdinando and Farinelli, Alessandro},
  booktitle={Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence},
  pages={217--224},
  year={2023}
}
```

**(C)** [Scaling #DNN-Verification Tools with Efficient Bound Propagation and Parallel Computing](https://arxiv.org/pdf/2312.05890).  Marzari L., Roncolato G., and Farinelli A. AIRO, 2023
```
@incollection{marzari2023scaling,
  title={Scaling \#DNN-Verification Tools with Efficient Bound Propagation and Parallel Computing},
  author={Marzari, Luca and Roncolato, Gabriele and Farinelli, Alessandro},
  booktitle={AIRO 2023 Artificial Intelligence and Robotics 2023},
  year={2023}
}
```
