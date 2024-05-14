from netver.backend.ProVe import ProVe
from netver.backend.Estimated import Estimated
import tensorflow as tf; import numpy as np



class NetVer:

	"""
	Main class of the NetVer project, this project implements different methods for the formal verification of neural networks. 
	
	This class is the hub of the project, translates the properties expressed in different format in the correct format for the tools (same for the network
	models). This class also provides some check for the structure and the given parameters and returns errors and warning message if some parameters is not
	correctly setted.

	All the network/property are translated to solve the following two types of query:
		- positive: all the outputs of the network must be greater than 0
		- reverse positive: at least one output of the network must be greater than 0

	Attributes
	----------
		verifier : Object
			the requested verification tool correctly setted for the verification
		algorithms_dictionary: dict
			a dictionary that translate the key_string of the methods inside the object for the tool
		
	Methods
	-------
		run_verifier( verbose )
			method that formally verify the property P on the ginve network, running the given verification tool

	"""

	# Dictionary for the translation of the key string to the requested alogirhtm class object
	algorithms_dictionary = {
		"ProVe" : ProVe,
		"estimated" : Estimated
	}


	def __init__( self, algo, network, property, memory_limit=0, disk_limit=0, **kwargs ):

		"""
		Constructor of the class. This method builds the object verifier, setting all the parameters and parsing the proeprty 
		and the network to the correct format for the tool. 

		Parameters
		----------
			algo : string
				a string that indicates the algorith/tool to use
			property : dict
				a dictionary that describe the property to analyze
			network : tf.keras.Model
				neural network model to analyze
			memory_limit : int
				maximum amount of virtual memory to use during the verification process
			disk_limit : int
				maximum amount of disk space to use for intermediate results during the verification process
			kwargs : **kwargs
				dictionary to overide all the non-necessary paramters (if not specified the algorithm will use the default values)	
		"""

		if property["type"] == "positive":
			self.primal_network = network
			self.dual_network = self.create_dual_net_positive( network )
			kwargs["reversed"] = False

		elif property["type"] == "decision":
			self.primal_network = self._create_net_decision( network, property )
			self.dual_network = self.create_dual_net_positive( self.primal_network )
			kwargs["reversed"] = True

		else:
			raise ValueError("Invalid property type, valid values: [positive, decision]")

		kwargs["memory_limit"] = memory_limit
		kwargs["disk_limit"] = disk_limit

		# Check mismatch between size of the input layer and domain P of the property
		assert( self.primal_network.input.shape[1] == len(property["P"]) )

		# Creation of the object verifier, calling the selected algorithm class with the required parameters

		self.verifier = self.algorithms_dictionary[algo]( self.primal_network, np.array(property["P"]), self.dual_network,
														  **kwargs )

		
	def run_verifier(self, start_time, verbose=0, estimation=None):

		"""
		Method that perform the formal analysis, launching the object verifier setted in the constructor.

		Parameters
		----------
			verbose : int
				when verbose > 0 the software print some log informations

		Returns:
		--------
			sat : bool
				true if the proeprty P is verified on the given network, false otherwise
			info : dict
				a dictionary that contains different information on the process, the 
				key 'counter_example' returns the input configuration that cause a violation
				key 'exit_code' returns the termination reason (timeout or completed)
		"""
		
		#
		return self.verifier.verify(start_time, verbose, estimation)


	def _create_net_decision( self, network, property ):

		"""
		This method modify the network using the given network and the decision property (i.e., the pivot node can not be the one with the highest value), 
		to create a network ehich is verifiable with a 'reverse positive' query (i.e., at least one output of the network must be greater than 0). 
		To this end, the method adds n-1 nodes to the netwrok, each of which is the results of itself - the pivot node.
		If one of the other output is greater than the pivot node the 'reverse positive' query is succesfully proved.

		Parameters
		----------
			network : tf.keras.Model
				tensorflow model to analyze, the model must be formatted in the 'tf.keras.Model(inputs, outputs)' format
			property : dict
				a dictionary that describe the 'decision' property to analyze 

		Returns:
		--------
			network_custom : tf.keras.Model
				the netowrk model modified for the 'reverse positive' query
		"""

		# Get the size of the output layer and the pivot node
		output_size = network.output.shape[1]
		prp_node = property["A"]

		# Create the custom last layer (linear activation) of n-1 nodes, 
		# and create the fully connected new network with this last layer attached
		output_custom = tf.keras.layers.Dense(output_size-1, activation='linear', name='output_custom')(network.output)
		network_custom = tf.keras.Model( network.input, output_custom )

		# Create the array for the biases and weights of the new fully connected layer to zero
		custom_biases = np.zeros(output_size-1)
		custom_weights = np.zeros((output_size, output_size-1))

		# Set to -1 the weights exiting from the pivot node for the formula node_i - pivot
		for i in range(output_size-1): custom_weights[prp_node][i] = -1

		# To complete the formula node_i - pivot set to 1 the exit weights of the i-th node
		c = 0
		for i in range(output_size):
			if i == prp_node: continue
			custom_weights[i][c] = 1
			c += 1

		# Set the weights and biases of the last fully connectd layer to the new generated values
		network_custom.layers[-1].set_weights([custom_weights, custom_biases])

		#
		return network_custom


	def create_dual_net_positive( self, network ):

		"""
		This method generate the dual netowrk using the given network and the decision property (i.e., the pivot node can not be the one with the highest value),
		
		Parameters
		----------
			network : tf.keras.Model
				tensorflow model to analyze, the model must be formatted in the 'tf.keras.Model(inputs, outputs)' format

		Returns:
		--------
			dual_network : tf.keras.Model
				the netowrk model modified for the 'reverse positive' query	
		"""

		# Get the size of the output layer
		output_size = network.output.shape[1]

		# Create the custom last layer (linear activation) of n nodes, 
		# and create the fully connected new network with this last layer attached
		output_custom = tf.keras.layers.Dense(output_size, activation='linear', name='new')(network.output)
		dual_network = tf.keras.Model( network.input, output_custom )

		# Create the array for the biases and weights of the new fully connected layer to zero
		custom_biases = np.zeros(output_size)
		custom_weights = np.zeros((output_size, output_size))

		# Set to -1 the weights exiting from each output node to create the negation of the output layer
		for i in range(output_size): custom_weights[i][i] = -1
		
		# To complete the formula node_i - pivot set to 1 the exit weights of the i-th node
		dual_network.layers[-1].set_weights([custom_weights, custom_biases])

		#
		return dual_network

