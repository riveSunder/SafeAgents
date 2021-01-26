import numpy as np
import argparse

import time

from safe_agents.policies import MLP
from open_safety.envs.puck_env import PuckEnv
from open_safety.envs.balance_bot_env import BalanceBotEnv 
#from open_safety_gym.envs.kart_env import KartEnv
#from open_safety_gym.envs.hoverboard_env import HoverboardEnv

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
            
            import pdb; pdb.set_trace()
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

    keep = int(0.125 * len(population))

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


def enjoy(env, input_dim, output_dim, max_episodes=1, model=None, steps_per_epd=1000):

    # hard-coded params
    hid_dim = [16,16]

    exp_id = "eval_{}".format(str(int(time.time())))

    results = {"costs": [],
            "rewards": [],
            "steps": []}
    
    pop_size = 1
    population = [MLP(input_dim, output_dim, hid_dim) \
            for ii in range(pop_size)]

    if model is not None:
        param_means = np.load(model)

        for ll in range(pop_size):
            population[ll].mean = param_means 
            population[ll].covar *= 1e-3
            population[ll].init_params() 

    for epd in range(max_episodes):

        reward, cost, steps = get_fitness(population[0], env, epds=1, get_cost=True,\
                max_steps=steps_per_epd)
        
        print("episode {} accumulated reward, cost = {:.3e}, {:.3e}".format(epd, reward, cost))

        results["costs"].append(cost)
        results["rewards"].append(reward)
        results["steps"].append(steps)



    mean_epd_cost = np.mean(results["costs"])
    mean_epd_reward = np.mean(results["rewards"])

    print("mean episodic reward and cost over {} episodes".format(max_episodes))
    print("     reward : {:.3e} +/- {:.3e}".format(mean_epd_reward, np.std(results["rewards"])))
    print("     cost   : {:.3e} +/- {:.3e}".format(mean_epd_cost, np.std(results["costs"])))

def train_es(env, input_dim, output_dim, pop_size=6, max_gen=100, \
        cost_constraint=0.0, cost_penalty=False,\
        model=None, reward_hypothesis=False, steps_per_epd=1000):


    # hard-coded policy parameters
    hid_dim = [16,16]
    es_lr = 1e-1
    reward_cost_ratio = 20
    save_results = True
    exp_id = "c{}_rh{}_{}".format(int(cost_constraint*10),\
        bool(reward_hypothesis), str(int(time.time()))[-10:])
    
    # generate a population
    population = [MLP(input_dim, output_dim, hid_dim) \
            for ii in range(pop_size)]
    
    if model is not None:
        param_means = np.load(model)

        for ll in range(pop_size):
            population[ll].mean = param_means 
            population[ll].init_params() 

    results = {"costs": [],
            "combined_rc": [],
            "rewards": [],
            "elite_costs": [],
            "elite_rewards": [], 
            "steps": []}

    t0 = time.time()    
    try:
        for gen in range(max_gen):
            fitnesses = []
            costs = []
            total_steps = []
            rc_fitnesses = []

            for agent_idx in range(len(population)):
                max_steps = steps_per_epd #2000 #np.min([2000, 250 + 10* gen])
                epds = 4 #np.max([1,int(10 - gen/10)])

                reward, cost, steps = get_fitness(population[agent_idx],\
                        env, epds=epds, max_steps=max_steps)
                fitnesses.append(reward)
                total_steps.append(steps)
                costs.append(cost)
                
                rc_fitnesses.append(reward - cost / reward_cost_ratio)
            
            try:
                param_means *= (1 - es_lr)

                if reward_hypothesis:
                    costs = [el*0 for el in costs]
                    cost_constraint = float("Inf")
                    gen_results = get_elite_mean(population, rc_fitnesses, costs, cost_constraint)
                else:
                    gen_results = get_elite_mean(population, fitnesses, costs, cost_constraint)

                param_means += es_lr * gen_results[0] 
            except:

                if reward_hypothesis:
                    gen_results = get_elite_mean(population, rc_fitnesses, costs, cost_constraint)
                else:
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

            if save_results and gen % 50 == 0:
                np.save("./results/means_{}_gen{}.npy".format(\
                    exp_id, gen), param_means)
                np.save("./results/temp_{}_results.npy".format(\
                    exp_id), results)

            elapsed = time.time() - t0
            t1 = time.time()
            epd_elapsed = time.time() - t1
            
            print("generation: {} total steps: {} elapsed total: {:.3f} this generation: {:.3f}"\
                    .format(\
                    gen, np.sum(total_steps), elapsed, epd_elapsed))

            del fitnesses
            del costs
            del total_steps

    except KeyboardInterrupt:
        pass

    if save_results:
        np.save("./results/means_{}_gen{}.npy".format(\
            exp_id, gen), param_means)
        np.save("./results/temp_{}_results.npy".format(\
            exp_id), results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args for safe agents")
    parser.add_argument("-n", "--env_name", type=str,\
            help="name of environment", default="PuckEnvGoal0-v0")
    parser.add_argument("-a", "--algo", type=str,\
            help="training algo", default="es")
    parser.add_argument("-c", "--constraint", type=float,\
            help="safety constraint", default=9e9)
    parser.add_argument("-m", "--model", type=str,\
            help="resume parameters stored at filepath (default None)", default=None)
    parser.add_argument("-r", "--reward_hypothesis", type=bool,\
            help="combine cost and reward (reward hypothesis)", default=False)
    parser.add_argument("-p", "--pop_size",type=int,\
            help="population size", default=64)
    parser.add_argument("-e", "--enjoy", type=bool,\
            help="render episodes", default = False)
    parser.add_argument("-g", "--generations", type=int,\
            help="number of generations", default=1024)
    parser.add_argument("-s", "--steps_per_episode", type=int,\
            help="steps per episode ;)", default=1000)

    parser.add_argument("-hm", "--how_many_epds", type=int,\
            help="number of episodes to enjoy", default=3)


    args = parser.parse_args()

    constraint = args.constraint
    rh = args.reward_hypothesis
    model = args.model
    pop_size = args.pop_size
    render = args.enjoy
    steps_per_episode = args.steps_per_episode
    max_generations = args.generations
    env_name = args.env_name

    if "uck" in args.env_name:
        env = PuckEnv(render=render)
        obs_dim = env.observation_space.sample().shape[0]
        act_dim = env.action_space.sample().shape[0]

        
    elif "alance" in args.env_name:
        env = BalanceBotEnv(render=render)
        obs_dim = env.observation_space.sample().shape[0]
        act_dim = env.action_space.sample().shape[0]

    elif "art" in args.env_name:
        env = KartEnv(render=render)
        obs_dim = env.observation_space.sample().shape[0]
        act_dim = env.action_space.sample().shape[0]

    elif "overboard" in args.env_name:
        env = HoverboardEnv(render=render)

        obs_dim = env.observation_space.sample().shape[0]
        act_dim = env.action_space.sample().shape[0]

    if not(args.enjoy):
        train_es(env, obs_dim, act_dim, cost_constraint=constraint, pop_size=pop_size,\
                max_gen=max_generations, model=model, reward_hypothesis=rh,\
                steps_per_epd=steps_per_episode)
    else:

        enjoy(env, obs_dim, act_dim, max_episodes=args.how_many_epds,\
                model=model, steps_per_epd=1000)


    print("all oK")

