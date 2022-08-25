import numpy as np

from ga.decay_types import DecayType


class DecayManager:
	def __init__(self , decay_type : DecayType, max_days, **decay_args):

		self.decay_type = decay_type
		self.max_days = max_days
		self.under_lockdown = 0

		self.b_0 = decay_args["b_0"]
		self.b_on = decay_args["b_on"]
		self.b_l_min = decay_args["b_l_min"]
		self.b_min = decay_args["b_min"]
		self.c_0 = decay_args["c_0"]
		self.c_l = decay_args["c_l"]

		self.lockdown_t_0 = False
		self.lockdown_t_1 = False
		self.lockdown_days = 0
		self.last_b = 0

		self.beta_generator = self.beta_decay_generator()

		if decay_type == DecayType.On_Off:
			self.decay_function = self.on_off

		elif decay_type == DecayType.Decay_Off:
			self.decay_function = self.decay_off

		elif decay_type == DecayType.On_Decay:
			self.decay_function = self.on_decay

		elif decay_type == DecayType.Decay_Decay:
			self.decay_function = self.decay_decay

	def step(self, action):
		return self.decay_function(action)

	def beta_decay_generator(self):
		arr = [ (self.b_0 - self.b_min)*np.exp(-1*self.c_0*i) + self.b_min for i in range(self.max_days)]
		for i in range(self.max_days):
			yield arr[i]

	def decay_decay(self, action):
		if action ==0:
			self.last_b = next(self.beta_generator)
			self.lockdown_days = 0
			return self.last_b
		else:
			self.lockdown_days+=1
			return (self.last_b - self.b_l_min)*np.exp(-1 * self.c_l * (self.lockdown_days)) + self.b_l_min


	def decay_off(self, action):
		if not action :
			self.last_b = next(self.beta_generator)
			self.lockdown_days = 0
			return self.last_b
		else:
			self.lockdown_days+=1
			return self.b_l_min

	def on_decay(self, action):
		if not action :
			self.last_b = self.b_on
			self.lockdown_days = 0
			return self.last_b
		else:
			self.lockdown_days+=1
			return (self.last_b - self.b_l_min)*np.exp(-1 * self.c_l * (self.lockdown_days)) + self.b_l_min

	def on_off(self, action):
		if not action :
			self.last_b = self.b_on
			self.lockdown_days = 0
			return self.last_b
		else:
			self.lockdown_days+=1
			return self.b_l_min