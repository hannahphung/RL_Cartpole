"""Test a random policy.

Author: Elie KADOCHE.
"""

import torch

from src.envs.cartpole import CartpoleEnv
from src.models.actor import ActorModel

if __name__ == "__main__":
    # Create policy
    policy = ActorModel()
    policy.eval()
    print(policy)

    # Create environment
    env = CartpoleEnv()

    # Reset it
    total_reward = 0.0
    state, _ = env.reset(seed=None)

    # While the episode is not finished
    terminated = False
    while not terminated:

        # Use the policy to generate the probabilities of each action
        probabilities = policy(state)

        # ---> TODO: how to select an action
        # for output maybe 2? each one is left or right 
        # the probabilities are the magnitude of the force 
        action = 0
        if probabilities[0][1] > probabilities[0][0]:
            action = 1

        # One step forward
        state, reward, terminated, _, _ = env.step(action)

        # Render (or not) the environment
        total_reward += reward
        env.render()

    # Print reward
    print("total_reward = {}".format(total_reward))
