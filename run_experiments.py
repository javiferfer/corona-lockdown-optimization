import os
from multiprocessing import Pool
from tqdm import tqdm


def run(cmd_arg):
	os.system(f"python train_ga.py {cmd_arg}")

def __main__():
	p = Pool(5)
	decay_types = ["On_Off", "Decay_Off", "On_Decay", "Decay_Decay"]
	lockdown_blocks = [5, 10, 15, 30]
	number_of_runs = 5

	for decay_type in tqdm(decay_types):
		for lockdown_block in lockdown_blocks:
			cmd_arg = [ f"--quarantine-block={lockdown_block} --save-name=final_run_{decay_type} --folder-name=run_{i} --decay-type={decay_type}" for i in range(number_of_runs)]
			p.map(run, cmd_arg)
   

if __name__ == "__main__":
	__main__()
