import numpy as np
import pickle
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Pool
import argparse
import copy
import os

from ga.seirs_parameters import SEIRSParams
from ga.ga_methods import mutate, es_select
from ga.utils import run, plot_agent
from ga.agents import DummyAgent


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--quarantine-block', type=int, default=15, help='min length of each qurantine')    
	parser.add_argument('--save-name', type=str, default="default_run", help='loggging and saving name')
	parser.add_argument('--decay-type', type=str, default="", help='override decay type')
	parser.add_argument('--folder-name', type=str, default="", help='folder to save the data')
	args = parser.parse_args()
	
	params = SEIRSParams('config/default_config.ini')
	
	if args.decay_type != "":
		params.decay_type = args.decay_type

	start_day = params.no_quarantine_period
	max_days = params.max_time
	qurantine_interval= max_days - start_day

	num_agents = 50
	num_threads = 5
	population = []

	folder_name="ga_runs"
	if args.folder_name != "":
		folder_name = args.folder_name

	for i in range(num_agents):  # init 100 neural networks with random weights
		agent1 = DummyAgent(total_lockdown_days=60, lockdown_length=args.quarantine_block, sim_length=qurantine_interval)
		mutate(agent1)
		population.append(agent1)  # store weights of the agents rather than the object 
	population = np.array(population)

	if not os.path.exists(f"./ga_runs_{params.cur_country}/{folder_name}"):
		os.mkdir(f"./ga_runs_{params.cur_country}/{folder_name}")
	if not os.path.exists(f"./ga_runs_{params.cur_country}/{folder_name}/qurantine_block_{args.quarantine_block}_{args.save_name}"):
		os.mkdir(f"./ga_runs_{params.cur_country}/{folder_name}/qurantine_block_{args.quarantine_block}_{args.save_name}")

	sr = SummaryWriter(f"./ga_runs_{params.cur_country}/{folder_name}/qurantine_block_{args.quarantine_block}_{args.save_name}/")
	p = Pool(num_threads)  # number of processes
	momentum = 80  # momentum of exploration
	resample_num = 1

	total_lockdown_days_arr = np.ones(len(population))*60
	lockdown_length_arr = np.ones(len(population))*args.quarantine_block
	cpy_params = [copy.copy(params) for _ in range(len(total_lockdown_days_arr))]

	for i in range(400):
		scores_arr = []
		quarantines_arr = []
		for _ in range(resample_num):
			return_val = p.map(run, zip(population, total_lockdown_days_arr, lockdown_length_arr, cpy_params),int((num_agents)/num_threads))  # run the model for all agents in paralel
			return_val = np.array(return_val)
			scores, quarantines = return_val[:, 0], return_val[:, 1]
			quarantines = np.vstack(quarantines)
			scores_arr.append(scores)
			quarantines_arr.append(quarantines)
			
		scores_arr = np.array(scores_arr)
		quarantines_arr = np.array(quarantines_arr)
		min_indices = [np.argmin(scores_arr[:, i]) for i in range(len(population))]
		min_scores = scores_arr[min_indices, range(len(population))]
		population = es_select(min_scores, population, scale = 20 + momentum)  # create a new population with 5 elites

		momentum -= momentum/100  # Simulated annealing-like momentum

		# for live plotting
		sr.add_scalar("max_score", np.max(scores), i)
		sr.add_scalar("mean_score", np.mean(scores), i)
		sr.add_scalar("number of quarantines", np.mean(np.sum(quarantines, axis=1)), i)
		sr.add_scalar("momentum", momentum, i)
	
		if i % 10 == 0:
			for elite in range(5):
				img = plot_agent(population[elite],params)
				sr.add_image(f"curve elite {elite}", np.array(img),i)   

		if i % 10 == 0:  # save model and plot seirs graph
			pickle.dump(population, open(f"./ga_runs_{params.cur_country}/{folder_name}/qurantine_block_{args.quarantine_block}_{args.save_name}/epoc_{i}.p","wb"))
	
	p.close()


if __name__ == "__main__":
	main()
