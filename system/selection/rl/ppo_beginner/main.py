"""
	This file is the executable for running PPO. It is based on this medium article: 
	https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
"""

import gymnasium as gym
import sys
import torch

from system.selection.rl.ppo_beginner.arguments import get_args
from ppo import PPO
from network import FeedForwardNN
from system.selection.rl.ppo_beginner.eval_policy import eval_policy

def train(env, hyperparameters, actor_model, critic_model):
	"""
		Trains the model.

		Parameters:
			env - the environment to train on
			hyperparameters - a dict of hyperparameters to use, defined in main
			actor_model - the actor model to load in if we want to continue training
			critic_model - the critic model to load in if we want to continue training

		Return:
			None
	"""	
	print("-"*20,f"Now   Training", flush=True)

	# Create a model for PPO.
	# 创建一个PPO模型，使用指定的策略类和环境，以及超参数
	model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)

	# Tries to load in an existing actor/critic model to continue training on
	if actor_model != '' and critic_model != '':
		print("-"*20,f"Loading in {actor_model} and {critic_model}...", flush=True)
		model.actor.load_state_dict(torch.load(actor_model))
		model.critic.load_state_dict(torch.load(critic_model))
		print("-"*20,f"Successfully loaded.", flush=True)
	elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
		print("-"*20,f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
		sys.exit(0)
	else:
		print("-"*20,f"Training from scratch.", flush=True)

	# Train the PPO model with a specified total timesteps
	# NOTE: You can change the total timesteps here, I put a big number just because
	# you can kill the process whenever you feel like PPO is converging
	timesteps=200_000_000
	timesteps = 2000
	model.learn(total_timesteps=timesteps)


def test(env, actor_model):
	"""
		Tests the model.

		Parameters:
			env - the environment to test the policy on
			actor_model - the actor model to load in

		Return:
			None
	"""
	print(f"Testing {actor_model}", flush=True)

	# If the actor model is not specified, then exit
	if actor_model == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)

	# Extract out dimensions of observation and action spaces
	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]

	# Build our policy the same way we build our actor model in PPO
	policy = FeedForwardNN(obs_dim, act_dim)

	# Load in the actor model saved by the PPO algorithm
	policy.load_state_dict(torch.load(actor_model))

	# Evaluate our policy with a separate module, eval_policy, to demonstrate
	# that once we are done training the model/policy with ppo.py, we no longer need
	# ppo.py since it only contains the training algorithm. The model/policy itself exists
	# independently as a binary file that can be loaded in with torch.
	eval_policy(policy=policy, env=env, render=True)

def main(args):
	"""
		The main function to run.

		Parameters:
			args - the arguments parsed from command line

		Return:
			None
	"""
	# NOTE: Here's where you can set hyperparameters for PPO. I don't include them as part of
	# ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
	# To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters
	# 这里可以设置PPO的超参数。为了方便，不将超参数作为命令行参数，因为每次输入都太麻烦。
	# 要查看超参数的列表，可以查看ppo.py中的_init_hyperparameters函数。
	hyperparameters = {
		'timesteps_per_batch': 100,  # 每个batch的时间步数
		'max_timesteps_per_episode': 20,  # 每个episode的最大时间步数
		'gamma': 0.99,  # 折扣因子
		'n_updates_per_iteration': 10,  # 每次迭代更新次数
		'lr': 3e-4,  # 学习率
		'clip': 0.2,  # PPO中的剪裁参数
		'render': True,  # 是否渲染环境
		'render_every_i': 10  # 每隔多少次迭代渲染一次
	}


	# Creates the environment we'll be running. If you want to replace with your own
	# custom environment, note that it must inherit Gym and have both continuous
	# 创建将要运行的环境。如果想替换为自定义环境，需要确保它继承自Gym，并具备连续的观测和动作空间。
	env = gym.make('Pendulum-v1', render_mode='human' if args.mode == 'test' else 'rgb_array')

	# Train or test, depending on the mode specified
	if args.mode == 'train':
		train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
	else:
		test(env=env, actor_model=args.actor_model)

if __name__ == '__main__':
	args = get_args() # Parse arguments from command line
	main(args)
