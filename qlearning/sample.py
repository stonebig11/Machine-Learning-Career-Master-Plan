import numpy as np
import gym

# Create the environment
env = gym.make('FrozenLake-v1', map_name="8x8", render_mode="human", is_slippery=False)  # Example environment: FrozenLake-v1 (is_slippery=False)
env.reset()
print(env.render())

# Q-learning parameters
num_episodes = 10000
learning_rate = 0.1
discount_factor = 0.99
exploration_prob = 0.3

# Initialize the Q-table with zeros
action_space_size = env.action_space.n
state_space_size = env.observation_space.n
q_table = np.zeros((state_space_size, action_space_size))

def map_state(state):
    if isinstance(state, int):
        return state
    return state[0]


# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        # Convert states to indices
        state_index = map_state(state)
        

        if np.random.uniform(0, 1) < exploration_prob:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state_index, :])  # Exploit
        
        next_state, reward, done, truncated, _ = env.step(action)
        next_state_index = map_state(next_state)
        
        # Q-value update using the Bellman equation
        q_table[state_index, action] = (1 - learning_rate) * q_table[state_index, action] + \
            learning_rate * (reward + discount_factor * np.max(q_table[next_state_index, :]))
        
        state = next_state

# After training, use the Q-table to make decisions
state = env.reset()
done = False
while not done:
    print(state)
    state_index = map_state(state)
    action = np.argmax(q_table[state_index, :])
    next_state, reward, done, truncated, _ = env.step(action)
    state = next_state
