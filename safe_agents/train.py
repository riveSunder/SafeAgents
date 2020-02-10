import numpy as np
import argparse

import time

from safe_agents.policies import MLP
from open_safety_gym.envs.puck_env import PuckEnv 

def get_fitness(agent, env, epds, get_cost=True):

    #epd_rewards = []
    #epd_costs = []
    for epd in range(epds):
        steps = 0
        done = False
        sum_reward = 0
        sum_cost = 0
        obs = env.reset()

        while not done and steps < 1000:
            action = agent.forward(obs)
            if len(action.shape) > 1:
                action = action.squeeze()
            
            obs, reward, done, info = env.step(action)

            sum_reward += reward
            sum_cost += info["cost"]
            steps += 1

    sum_reward /= epds
    sum_cost /= epds

    return sum_reward, sum_cost

def get_elite_mean(population, reward, cost=None,cost_constraint=2.5):


    if cost is not None:
        adjusted_cost = [max([cost_constraint, elem]) for elem in cost]
        cost_fitness_agent = [[cost, fit, agent.parameters]
                for a_cost, cost, fit, agent in \
                sorted(zip(adjusted_cost, cost, reward, population),\
                key = lambda trip: [-trip[0], trip[2]], reverse=True)]
            
        fitness = [elem[1] for elem in cost_fitness_agent]
        cost = [elem[0] for elem in cost_fitness_agent]
        population = [elem[2] for elem in cost_fitness_agent]

    else:
        fitness_agent = [[fit, agent] \
                for fit, agent in zip(reward, population)]

        fitness = [elem[0] for elem in cost_fitness_agent]
        cost = 0.0

    keep = int(0.25 * len(population))

    elite_pop = population[:keep] 
    elite_cost = cost[:keep] 
    elite_fitness = fitness[:keep]

    print("population mean cost, rewards: {:.3e}, {:.3e}".format(\
            np.mean(cost), np.mean(fitness)))

    print("elite mean cost, rewards: {:.3e}, {:.3e}".format(\
            np.mean(elite_cost), np.mean(elite_fitness)))

    param_sum = elite_pop[0]

    for agent_idx in range(1,keep):
        param_sum += elite_pop[agent_idx] 
    

    param_means = param_sum / keep

    return [param_means, np.mean(cost), np.mean(elite_cost), \
            np.mean(fitness), np.mean(elite_fitness)]

def train_es(env, input_dim, output_dim, pop_size=6, max_gen=100, cost_constraint=0.0, cost_penalty=False):

    # hard-coded policy parameters
    hid_dim = [32,32]
    es_lr = 1e-1


    
    # generate a population
    population = [MLP(input_dim, output_dim, hid_dim) \
            for ii in range(pop_size)]
#    param_means = np.load("./params2_goal00.npy")
#
#    for ll in range(pop_size):
#        population[ll].mean = param_means 
#        population[ll].init_params() 
    results = {"costs": [],
            "rewards": [],
            "elite_costs": [],
            "elite_rewards": []}
    try:
        for gen in range(max_gen):
            fitnesses = []
            costs = []
            print("generation {}".format(gen))

            for agent_idx in range(len(population)):

                reward, cost = get_fitness(population[agent_idx], env, epds=2)
                fitnesses.append(reward)
                costs.append(cost)
            
            try:
                param_means *= (1 - es_lr)
                gen_results = get_elite_mean(population, fitnesses, costs, cost_constraint)
                param_means += es_lr * gen_results[0] 
            except:
                gen_results = get_elite_mean(population, fitnesses, costs, cost_constraint)
                param_means = gen_results[0] 

            for ll in range(pop_size):
                population[ll].mean = param_means 
                population[ll].init_params() 
            
            results["costs"].append(gen_results[1])
            results["elite_costs"].append(gen_results[2])
            results["rewards"].append(gen_results[3])
            results["elite_rewards"].append(gen_results[4])

            if gen % 5 == 0:
                np.save("./means_gen{}.npy".format(gen), param_means)
                np.save("./temp_results.npy", results)
            del fitnesses
            del costs

    except KeyboardInterrupt:
        pass


    np.save("./means_gen{}.npy".format(gen), param_means)
    np.save("./temp_results.npy", results)
    import pdb; pdb.set_trace()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args for safe agents")
    parser.add_argument("-n", "--env_name", type=str,\
            help="name of environment", default="PuckEnvGoal0-v0")
    parser.add_argument("-a", "--algo", type=str,\
            help="training algo", default="es")


    args = parser.parse_args()

    if "uck" in args.env_name:
        env = PuckEnv(render=False)
        obs_dim = env.observation_space.sample().shape[0]
        act_dim = env.action_space.sample().shape[0]

        train_es(env, obs_dim, act_dim, cost_constraint=25, pop_size=128, max_gen=1024)
        
        print("all oK")

