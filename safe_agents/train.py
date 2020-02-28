import numpy as np
import argparse

import time

from safe_agents.policies import MLP
from open_safety_gym.envs.puck_env import PuckEnv
from open_safety_gym.envs.balance_bot_env import BalanceBotEnv 

def get_fitness(agent, env, epds, get_cost=True, max_steps=1000):

    #epd_rewards = []
    #epd_costs = []
    total_steps = 0
    sum_reward = 0
    sum_cost = 0
    for epd in range(epds):
        steps = 0
        done = False
        obs = env.reset()

        while not done and steps < max_steps:
            action = agent.forward(obs)
            if len(action.shape) > 1:
                action = action.squeeze()
            
            obs, reward, done, info = env.step(action)

            sum_reward += reward
            sum_cost += info["cost"]
            steps += 1
        total_steps += steps

    sum_reward /= epds
    sum_cost /= epds

    return sum_reward, sum_cost, total_steps

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
#    param_means = np.load("./safe_agents/results/means_gen995.npy")
#
#    for ll in range(pop_size):
#        population[ll].mean = param_means 
#        population[ll].init_params() 
    results = {"costs": [],
            "rewards": [],
            "elite_costs": [],
            "elite_rewards": [], 
            "steps": []}
    try:
        for gen in range(max_gen):
            fitnesses = []
            costs = []
            total_steps = []

            for agent_idx in range(len(population)):
                max_steps = 2000 #np.min([2000, 250 + 10* gen])
                epds = 4 #np.max([1,int(10 - gen/10)])

                reward, cost, steps = get_fitness(population[agent_idx], env, epds=epds, max_steps=max_steps)
                fitnesses.append(reward)
                total_steps.append(steps)
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
            results["steps"].append(np.sum(total_steps))

            if gen % 50 == 0:
                np.save("./means_c{}_gen{}.npy".format(\
                    int(constraint*10),gen), param_means)
                np.save("./temp_c{}_results.npy".format(\
                    int(constraint*10)), results)

            print("generation {} total steps {} steps/epd {}".format(\
                    gen, np.sum(total_steps), np.sum(total_steps)/(epds*pop_size)))

            del fitnesses
            del costs
            del total_steps

    except KeyboardInterrupt:
        pass


    np.save("./means_c{}_gen{}.npy".format(\
        int(constraint*10),gen), param_means)
    np.save("./temp_c{}_results.npy".format(\
        int(constraint*10)), results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args for safe agents")
    parser.add_argument("-n", "--env_name", type=str,\
            help="name of environment", default="PuckEnvGoal0-v0")
    parser.add_argument("-a", "--algo", type=str,\
            help="training algo", default="es")
    parser.add_argument("-c", "--constraint", type=float,\
            help="safety constraint", default=10.0)

    args = parser.parse_args()

    constraint = args.constraint
    if "uck" in args.env_name:
        env = PuckEnv(render=False)
        obs_dim = env.observation_space.sample().shape[0]
        act_dim = env.action_space.sample().shape[0]

        train_es(env, obs_dim, act_dim, cost_constraint=constraint, pop_size=64, max_gen=2048)
        
    elif "alance" in args.env_name:
        env = BalanceBotEnv(render=False)
        obs_dim = env.observation_space.sample().shape[0]
        act_dim = env.action_space.sample().shape[0]

        train_es(env, obs_dim, act_dim, cost_constraint=constraint, pop_size=64,\
                max_gen=2048)
        
    print("all oK")

