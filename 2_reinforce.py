"""REINFORCE algorithm.

Author: Elie KADOCHE.
"""

import torch
from torch import optim as optim
#from torch import lr_scheduler as lr_scheduler
from torch.distributions import Categorical

from src.envs.cartpole import CartpoleEnv
from src.models.actor import ActorModel

# Policy model path
ACTOR_PATH = "models/actor.pt"

# Maximum environment length
HORIZON = 500

# ---> TODO: change the discount factor to solve the problem
DISCOUNT_FACTOR = 0.99

# ---> TODO: change the learning rate to solve the problem
# very small
LEARNING_RATE = 0.001

# initialize learning rate
# after each iteration -> decrease the learning rate
# because closer to converging

if __name__ == "__main__":
    # Create policy
    actor = ActorModel()
    actor.train()

    # Create the environment
    env = CartpoleEnv()

    # Create optimizer with the policy parameters
    actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)

    # adjust learning rate
    # actor_optimizer = optim.SGD(actor.parameters(), lr=LEARNING_RATE, momentum=0.9)
    # scheduler = optim.lr_scheduler.ExponentialLR(actor_optimizer, gamma=0.9)
    # scheduler = optim.lr_scheduler.LinearLR(actor_optimizer)

    # ---> TODO: when do we stop the training?
    # criteria to stop the training: 
    # when the model performs 20 consecutive successful iteration 
    # success_iteration == 20

    # Run infinitely many episodes
    training_iteration = 0
    success_iteration = 0
    while True:

        # Experience
        # ------------------------------------------

        # Reset the environment
        state, _ = env.reset()

        # During experience, we will save:
        # - the probability of the chosen action at each time step pi(at|st)
        # - the rewards received at each time step ri
        saved_probabilities = list()
        saved_rewards = list()

        # Prevent infinite loop
        for t in range(HORIZON + 1):

            # Use the policy to generate the probabilities of each action
            probabilities = actor(state)

            # Create a categorical distribution over the list of probabilities
            # of actions and sample an action from it
            distribution = Categorical(probabilities)
            action = distribution.sample()

            # Take the action
            state, reward, terminated, _, _ = env.step(action.item())

            # Save the probability of the chosen action and the reward
            saved_probabilities.append(probabilities[0][action])
            saved_rewards.append(reward)

            # End episode
            if terminated:
                break

        # Compute discounted sum of rewards
        # ------------------------------------------

        # Current discounted reward
        discounted_reward = 0.0

        # List of all the discounted rewards, for each time step
        # this discounted_rewards is backward too
        discounted_rewards = list()

        # ---> TODO: compute discounted rewards
        # move backward in the arr
        n = len(saved_rewards)
        cur_sum_rewards = saved_rewards[n-1]
        discounted_rewards.append(cur_sum_rewards)
        discounted_reward += cur_sum_rewards

        for i in range(n - 2,-1,-1):
            cur_sum_rewards = saved_rewards[i] + DISCOUNT_FACTOR * cur_sum_rewards 
            discounted_rewards.insert(0,cur_sum_rewards)
            discounted_reward += cur_sum_rewards
            
        # Eventually normalize for stability purposes
        discounted_rewards = torch.tensor(discounted_rewards)
        mean, std = discounted_rewards.mean(), discounted_rewards.std()
        discounted_rewards = (discounted_rewards - mean) / (std + 1e-7)

        # Update policy parameters
        # ------------------------------------------

        # For each time step
        actor_loss = list()
        for p, g in zip(saved_probabilities, discounted_rewards):

            # ---> TODO: compute policy loss LEARNING_RATE
            time_step_actor_loss = -(g * torch.log(p))

            # Save it
            actor_loss.append(time_step_actor_loss)

        # Sum all the time step losses
        actor_loss = torch.cat(actor_loss).sum()

        # Reset gradients to 0.0
        actor_optimizer.zero_grad()

        # Compute the gradients of the loss (backpropagation)
        actor_loss.backward()

        # Update the policy parameters (gradient ascent)
        actor_optimizer.step()

        # Logging
        # ------------------------------------------

        # Episode total reward
        episode_total_reward = sum(saved_rewards)

        # ---> TODO: when do we stop the training?
        if episode_total_reward == 500:
            success_iteration += 1

        # reset consecutive success count
        if episode_total_reward < 500:
            success_iteration = 0

        # Log results
        log_frequency = 5
        training_iteration += 1
        if training_iteration % log_frequency == 0:

            # Save neural network
            torch.save(actor, ACTOR_PATH)

            # Print results
            print("iteration {} - last reward: {:.2f}".format(
                training_iteration, episode_total_reward))
            if success_iteration > 0:
                # only print after every 5 iterations 
                # so success_iteration is not always printed 
                # might never see the success_iteration count of 20
                print('number of consecutive successful iteration so far:', success_iteration)

            # ---> TODO: when do we stop the training?

        if success_iteration == 20:
            print('Successfully trained the model')
            break

    # change the learning rate
    # scheduler.step()
