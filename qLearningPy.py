# code from https://gist.github.com/malzantot/9d1d3fa4fdc4a101bc48a135d8f9a289 -- the only thing that I changed were the variable values to experiment with stochastic vs deterministic modeling
# comments nearly all written by me

#import libraries
import numpy as np
import gym
from gym import wrappers

#declare variables:
n_states = 40 # declare number of action-reward states
iter_max = 10000 # number of value iteration updates

initial_lr = 0.7 # alpha, therefore he gives it as fully deterministic at the start
min_lr = 0.003 # given that he changes alpha, he quantifies the minimum learning rate -- denoting that he will not let the model be fully stochastic ever
gamma = 1.0 # discount factor, 1 value means deep reinforcement
t_max = 10000
eps = 0.02 # epsilon acts as a "random check" which arbitrarily determines if exploration continues

def run_episode(env, policy=None, render=False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    for _ in range(t_max):
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            a,b = obs_to_state(env, obs)
            action = policy[a][b]
        obs, reward, done, _ = env.step(action)
        total_reward += gamma ** step_idx * reward # increase alpha by factor of discount factor * current state reward
        step_idx += 1 # increase step
        if done:
            break
    return total_reward

def obs_to_state(env, obs): # comment
    """ Maps an observation to state """
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / n_states
    a = int((obs[0] - env_low[0])/env_dx[0])
    b = int((obs[1] - env_low[1])/env_dx[1])
    return a, b   


# create and initialize the environment
if __name__ == '__main__':
    env_name = 'MountainCar-v0' #using MountainCar
    env = gym.make(env_name) #create the gym
    env.seed(0) #seeding allows the program to be rerun and get the same values again -- help from Peter to understand this
    np.random.seed(0)
    print ('----- using Q Learning -----')
    
    q_table = np.zeros((n_states, n_states, 3))
    for i in range(iter_max):
        obs = env.reset()
        total_reward = 0 # reward initialized to 0
        ## eta: learning rate is decreased at each step
        eta = max(min_lr, initial_lr * (0.85 ** (i//100)))
        for j in range(t_max):
            a, b = obs_to_state(env, obs)
            if np.random.uniform(0, 1) < eps: # compare random to epsilon value for exploration
                action = np.random.choice(env.action_space.n)
            else:
                logits = q_table[a][b] # if random value is not less than epsilon comparison, 
                logits_exp = np.exp(logits) # logits_exp = the exponential of all of the table values (np.exp does this)
                probs = logits_exp / np.sum(logits_exp) # divide exponential by its sum
                action = np.random.choice(env.action_space.n, p=probs) # define action for this state
            obs, reward, done, _ = env.step(action) #assign reward to action for current state
            total_reward += reward # increase reward based on current step
            ## update q table
            a_, b_ = obs_to_state(env, obs)
            q_table[a][b][action] = q_table[a][b][action] + eta * (reward + gamma *  np.max(q_table[a_][b_]) - q_table[a][b][action]) #update q based on observed action/reward -- this is the equation
            if done:
                break
        if i % 100 == 0: #every 100 steps....
            print('Iteration #%d -- Total reward = %d.' %(i+1, total_reward))
            
solution_policy = np.argmax(q_table, axis=2)
solution_policy_scores = [run_episode(env, solution_policy, False) for _ in range(100)]
print("Average score of solution = ", np.mean(solution_policy_scores))
# Animate it
run_episode(env, solution_policy, True)