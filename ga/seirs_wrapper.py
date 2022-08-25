import numpy as np

from seirsplus.seirsplus.models import SEIRSModel
from ga.decay_functions import DecayManager
from ga.decay_types import DecayType


class SEIRSWrapper():
    def __init__(self, params):

        self.no_quarantine_period = params.no_quarantine_period  # start date of the lockdowns
        self.max_time = params.max_time
        self.qurantine_interval = self.no_quarantine_period - self.max_time

        self.progression_rate = params.progression_rate
        self.progression_rate_detected = params.progression_rate_detected  # Sigma_d
        self.recovery_rate = params.recovery_rate
        self.recovery_rate_detected = params.recovery_rate_detected
        self.resusceptibility_rate = params.resusceptibility_rate
        self.mortality_rate_without_hospital = params.mortality_rate_without_hospital
        self.test_positive_rate = params.test_positive_rate
        self.test_infectious_rate = params.test_infectious_rate
        self.mortality_rate_detected_cases = params.mortality_rate_detected_cases
        self.contact_tracing_exposed = params.contact_tracing_exposed
        self.contact_tracing_infected = params.contact_tracing_infected

        self.psi_i = params.psi_i
        self.psi_e = params.psi_e
        self.phi_i = params.phi_i
        self.phi_e = params.phi_e

        self.initial_population = params.initial_population
        self.initial_infected_population = params.initial_infected_population

        b_min = params.b_min
        b_l_min = params.b_l_min

        b_0 = params.b_0
        b_on = params.b_on
        c_0 = params.c_0
        c_l = params.c_l
        decay_type = params.decay_type

        self.num_of_quarantines = []
        self.quarantine_time = []
        self.cur_quarantine_length = 0
        self.lockdown_counter = 0
        self.quarantine_counter = 0
        self.non_quarantine_counter = 0

        self.dm = DecayManager(DecayType[decay_type], self.max_time, b_min=b_min, b_l_min=b_l_min, 
            b_0 = b_0, b_on = b_on, c_0=c_0, c_l=c_l)

        self.transmission_rate = self.dm.b_on
        self.current_beta = self.transmission_rate
        self.detected_infection_chance = self.transmission_rate
        self.R0 = self.transmission_rate/self.recovery_rate  # Calculate R0 for check
        

        self.recorded_betas = []

    def step(self, action):
        t = int(round(self.ref_model.t))
        
        if action:
            # self.quarantine_counter += 1
            # self.cur_quarantine_length += 1
            # self.transmission_rate -= self.ramp_down
            # if self.transmission_rate < 0.0001:
            #     self.transmission_rate=0.0001
            cur_beta = self.dm.step(action)
                
            checkpoints = {'t': [t],  # time of itervention
                           'beta':    [cur_beta],
                           'beta_D':  [self.detected_infection_chance],
                           'mu_I':    [self.mortality_rate_without_hospital],
                           'theta_E': [self.test_infectious_rate],
                           'theta_I': [self.test_positive_rate]
                           }
        else:
            # self.cur_quarantine_length = 0
            # self.lockdown_counter = 0
            # self.transmission_rate += self.ramp_up
            # if self.transmission_rate > 0.13:
            #     self.transmission_rate=0.13
            cur_beta = self.dm.step(action)

            checkpoints = {'t': [t],  # time of itervention
                           'beta':    [cur_beta],
                           'beta_D':  [self.detected_infection_chance],
                           'mu_I':    [self.mortality_rate_without_hospital],
                           'theta_E': [self.test_infectious_rate],
                           'theta_I': [self.test_positive_rate]
                           }

        self.recorded_betas.append(cur_beta)
        # if action == 0 :
        #     self.non_quarantine_counter+=1

        self.ref_model.run(T=1, checkpoints=checkpoints, verbose=False)
        self.num_of_quarantines.append(action)
        self.quarantine_time.append(t)
        if self.ref_model.t > self.max_time:
            done = True
            self.set_reduced_parameters()
        else:
            done = False

        # obs, score, done and info (GYM format)
        return [(self.ref_model.numD_E[-1] + self.ref_model.numD_I[-1])/self.initial_population], self.initial_population - self.ref_model.numF[-1], done, ""
        # note: since the agent doesnt have recurrent layers, we feed it with the last action. (less parameters this way)

    def reset(self): # reset the environment
        self.ref_model = SEIRSModel(beta=self.transmission_rate,
                                    sigma=self.progression_rate,
                                    gamma=self.recovery_rate,
                                    mu_I=self.mortality_rate_without_hospital,
                                    xi=self.resusceptibility_rate,
                                    beta_D=self.detected_infection_chance,
                                    sigma_D=self.progression_rate_detected,
                                    gamma_D=self.recovery_rate_detected,
                                    mu_D=self.mortality_rate_detected_cases,
                                    theta_E=self.test_infectious_rate,
                                    theta_I=self.test_positive_rate,
                                    psi_E=self.psi_e,
                                    psi_I=self.psi_i,
                                    initI=self.initial_infected_population,
                                    initN=self.initial_population,
                                    initE=0,
                                    initD_E=0,
                                    initD_I=0,
                                    initR=0,
                                    initF=0)

        self.ref_model.run(T=self.no_quarantine_period, verbose=False)

        self.num_of_quarantines.append(0)
        self.quarantine_time.append(round(self.ref_model.t))
        return [(self.ref_model.numD_E[-1]+self.ref_model.numD_I[-1])/self.initial_population]

    def set_reduced_parameters(self):
        self.t_map = {}
        reduced_inf = []

        for i, t in enumerate(self.ref_model.tseries):
            key = str(int(t))
            if not key in self.t_map.keys():
                self.t_map[key] = i

        self.infected_time_series = []

        for key, value in self.t_map.items():
            self.infected_time_series.append(self.ref_model.numI[value])

        self.death_time_series = []

        for key, value in self.t_map.items():
            self.death_time_series.append(self.ref_model.numF[value])

    def final_scores(self, mode="flatten"):

        if mode == "flatten":
            normalized_pop = (
                np.array(self.infected_time_series)/self.initial_population)
            score = 1-np.max(normalized_pop) + \
                np.mean(np.abs(np.std(normalized_pop)))*0.001

        if mode == "death":
            normalized_pop = (
                self.death_time_series[-1]/self.initial_population)
            score = 1-normalized_pop

        return score 
