import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from ga.seirs_wrapper import SEIRSWrapper
from ga.agents import DummyAgent


#runs 1 epoc for one agent
def run(inputs, mode="flatten"):
    agent, total_lockdown_days, lockdown_length, params = inputs

    env = SEIRSWrapper(params)

    obs = env.reset()
    for i in range(env.max_time):
        action = agent.forward(i)
        obs, score, done, info = env.step(int(action))
        if done:
            break
    return [env.final_scores(mode=mode), np.array(env.num_of_quarantines)]


def get_curve_stats(agent, params):
    score_list = []
    env = SEIRSWrapper(params)
    start_day = env.no_quarantine_period
    qurantines = []
    infected_list = []

    obs = env.reset()
    for i in range(env.max_time):
        action = agent.forward(i)
        qurantines.append(int(action))
        obs, score, done, info = env.step(int(action))

        score_list.append(score)
        infected_list.append(env.ref_model.numI[-1])

    qurantines = np.insert(
        np.array(env.num_of_quarantines).flatten(), 0, 0)
    mask = qurantines[1:]-qurantines[:-1]
    indices = np.argwhere(np.abs(mask) > 0).flatten()

    recorded_betas = list(np.ones(start_day)*env.recorded_betas[0])+env.recorded_betas

    return [env, recorded_betas, mask]


# Plots the seirs graph for given agent weights
def plot_agent(agent,params):

    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(211)
    bx = fig.add_subplot(212)

    score_list = []

    env = SEIRSWrapper(params)
    start_day = env.no_quarantine_period
    qurantines = []
    infected_list = []

    obs = env.reset()
    for i in range(env.max_time):
        action = agent.forward(i)
        qurantines.append(int(action))
        obs, score, done, info = env.step(int(action))

        score_list.append(score)
        infected_list.append(env.ref_model.numI[-1])

    qurantines = np.insert(
        np.array(env.num_of_quarantines).flatten(), 0, 0)
    mask = qurantines[1:]-qurantines[:-1]
    indices = np.argwhere(np.abs(mask) > 0).flatten()
    colors = ["gray"]
    for i in indices:
        if mask[i] == 1:
            colors.append("green")
        else:
            colors.append("red")

    vlines = [env.no_quarantine_period]
    for t, i in zip(env.quarantine_time[:], mask):
        if not i == 0:
            vlines.append(t)
    env.ref_model.plot(vline_colors=colors, vlines=vlines, ax=ax, plot_S=False, plot_E="line", plot_R=False,
                        plot_F="line", plot_I="line", plot_D_I="line", plot_D_E="line", combine_D=True)
    infected_s = env.ref_model.numI
    max_number_infected = np.max(infected_s)/env.initial_population
    t = ax.text(0.83, 0.6, "Peak of infection curve = {:.4f}".format(max_number_infected), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='large')
    t.set_bbox(dict(facecolor='white', edgecolor='white'))
    t = ax.text(0.87, 0.5, "% of fatalities = {:.4f}".format((env.death_time_series[-1]/env.initial_population)),
            horizontalalignment='center', verticalalignment='center', alpha=1.0, transform=ax.transAxes, fontsize='large')
    t.set_bbox(dict(facecolor='white', edgecolor='white'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.axvline(env.max_time, color='k', linestyle='dashed')
    ax.set_ylim(0.0, 0.20)
    ax.set_xlabel('day')
    ax.legend(loc="upper right", title_fontsize="x-large")
    
    recorded_betas = list(np.ones(start_day)*env.recorded_betas[0])+env.recorded_betas
    bx.plot(recorded_betas, label = "betas")
    bx.set_ylabel('infection rate')
    bx.set_xlabel('day')
    bx.set_xlim(0, 582)

    fig.tight_layout()
    canvas.draw()  # draw the canvas, cache the renderer
    s, (width, height) = canvas.print_to_buffer()
    image = np.fromstring(s, np.uint8).reshape((height, width, 4))
    image = np.transpose(image,[2,0,1])
    return image


# created for the visulation purpouses only
def tmp_run(inpts, mode="flatten"): 
    total_lockdown_days = 60
    weight, lockdown_length = inpts

    env = SEIRSWrapper()
    agent = DummyAgent(total_lockdown_days, lockdown_length)

    for target_param, param in zip(agent.parameters(), weight):
        target_param.data.copy_(param)

    obs = env.reset()
    for i in range(env.max_time):
        action = agent(i).detach().float().flatten().item()
        obs, score, done, info = env.step(np.round(action))

        if done:
            break

    return [env.ref_model.numI, env.final_scores()]

# created for the visulation purpouses only
def get_lockdowns(population, quarantine_block, pooling):
    lockdown_length_arr = np.ones((len(population), 1)) * quarantine_block
    return_vals = []
    return_vals = pooling.map(tmp_run, zip(population, lockdown_length_arr))
    return return_vals
