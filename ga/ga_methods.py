import numpy as np
import copy

from ga.agents import DummyAgent


# default mutation operator. 
def mutate(ind, scale=100):
   
    first_item = ind.params[0]
    random_first =np.random.random(1)*scale - scale/2
    diff = ind.params[1:]
    
    mutation = np.random.random(diff.shape) * scale - scale/2
    ind.params = np.clip(diff+mutation, -ind.lockdown_length, ind.sim_length)
    ind.params = np.concatenate([[random_first[0]+first_item], ind.params], axis=0)
    indices = np.argsort(ind.params)
    ind.params = ind.params[indices]
    ind.params = np.clip(ind.params, 0, ind.sim_length)

    return ind


def crossover(ind1, ind2, scale):
    length = len(ind1.params)
    mask = np.random.random(length).astype(int)
    offspring = ind1.params * mask + ind2.params * (1-mask)
    new_ind = copy.deepcopy(ind1)
    new_ind.params = offspring
    mutate(new_ind, scale)
    return new_ind


def section_crossover(ind1, ind2, scale):
    locations = (np.random.random(2)*len(ind1.params)).astype(int)
    locations = np.sort(locations)
    mask1 = np.ones(locations[0])
    mask2 = np.zeros(locations[1]-locations[0])
    mask3 = np.ones(len(ind1.params)-locations[1])
    mask = np.concatenate([mask1,mask2,mask3],axis=0)
    offspring = ind1.params * mask + ind2.params * (1-mask)
    new_ind = copy.deepcopy(ind1)
    new_ind.params = offspring
    mutate(new_ind, scale)
    return new_ind


def es_select(scores, pop, scale=1.4):
    pop_params = np.array([i.params for i in pop])
    indices = np.argsort(scores)
    elite = pop[indices[-1]]
    scores = scores-scores.mean() 
    scores /= scores.std()
    elite.params = elite.params + 2/(len(pop)*scale) * np.dot(pop_params.T, scores)
    elite.params = np.clip(elite.params, 0, elite.sim_length)
    indices = np.argsort(elite.params)
    elite.params = elite.params[indices]

    new_pop = np.empty(len(pop)).astype(DummyAgent)

    for i in (range(1,len(pop))):
        new_pop[i] = mutate(copy.deepcopy(elite), scale=scale)
    
    new_pop[0] = elite
    return new_pop


def select(scores, pop, n_elites = 5, scale=1.4, strategy = "section_crossover"):

    if strategy == "crossover":
        crossover_strategy = crossover
    elif strategy == "section_crossover":
        crossover_strategy = section_crossover

    indices = np.argsort(scores)
    elites = indices[-n_elites:]
    scores = scores-scores.min() + 1e-8
    scores /= scores.sum()

    new_pop = np.empty(len(pop)).astype(DummyAgent)

    for i in (range(len(pop))):
        inds = np.random.choice( pop, 2, p=scores.astype(float))
        new_ind = crossover_strategy(*inds, scale=scale)
        new_pop[i] = new_ind

    new_pop[:len(elites)] = pop[elites]
    return new_pop


# Not used. Only for visulation of the population.
def cluster_pop(pop,fitted_umap,first=False):
    tensor_pop=np.concat([np.concat([param.view(-1) for param in ind]).view(1,-1) for ind in pop])
    
    if first:
        standard_embedding = fitted_umap.fit_transform(tensor_pop.detach().numpy())
    else:
        standard_embedding = fitted_umap.transform(tensor_pop.detach().numpy())
        
    return standard_embedding