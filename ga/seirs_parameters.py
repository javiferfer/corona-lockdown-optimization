import configparser


class SEIRSParams:
	def __init__(self, config_file_path):
		config = configparser.ConfigParser()
		config.read(config_file_path)
		self.no_quarantine_period = config.getint('Shared_parameters', 'start_date') # start date of the lockdowns
		self.max_time = config.getint('Shared_parameters', 'max_days') 
		self.progression_rate = config.getfloat('Shared_parameters', 'sigma')  # Sigma
		self.progression_rate_detected = self.progression_rate  # Sigma_d
		self.recovery_rate = config.getfloat('Shared_parameters', 'gamma')  # Gamma
		self.recovery_rate_detected = self.recovery_rate
		self.resusceptibility_rate = config.getfloat('Shared_parameters', 'xi')  # Epsilon
		self.mortality_rate_without_hospital = config.getfloat('Shared_parameters', 'mu_i')  # Mu_i
		self.test_positive_rate = config.getfloat('Shared_parameters', 'theta_i')  # theta_i
		self.test_infectious_rate = config.getfloat('Shared_parameters', 'theta_e')  # theta_e
		self.mortality_rate_detected_cases = self.mortality_rate_without_hospital
		self.contact_tracing_exposed = 0
		self.contact_tracing_infected = 0
		self.psi_i = float(config.get('Shared_parameters', 'psi_i'))
		self.psi_e = float(config.get('Shared_parameters', 'psi_e'))
		self.phi_i = float(config.get('Shared_parameters', 'phi_i'))
		self.phi_e = float(config.get('Shared_parameters', 'phi_e'))
		self.initial_population = config.getint('Shared_parameters', 'population')
		self.initial_infected_population = config.getint('Shared_parameters', 'initi')
		self.b_min = config.getfloat('Case_parameters','b_min')
		self.b_l_min = config.getfloat('Case_parameters','b_l_min')
		self.cur_country = config.get('Case_parameters',"cur_country")
		self.b_0 = config.getfloat('Case_parameters','b_0_'+self.cur_country)
		self.b_on = config.getfloat('Case_parameters','b_on_'+self.cur_country)
		self.c_0 = config.getfloat('Case_parameters','c_0_'+self.cur_country)
		self.c_l = config.getfloat('Case_parameters','c_l_'+self.cur_country)
		self.decay_type = config.get('Case_parameters',"decay_type")
