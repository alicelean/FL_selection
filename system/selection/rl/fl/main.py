import gymnasium as gym
import torch
from arguments import get_args
from ppo import PPO
from network import FeedForwardNN
from fl_env import FederatedLearningEnv

def train(env, hyperparameters, actor_model, critic_model,total_timesteps):
    #初始化PPO
    model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)
    if actor_model != '' and critic_model != '':
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
    #模型训练
    model.learn(total_timesteps=total_timesteps)

def test(env, actor_model):
    policy = FeedForwardNN(env.observation_space.shape[1], env.action_space.n)
    policy.load_state_dict(torch.load(actor_model))
    eval_policy(policy=policy, env=env, render=True)

def rl_main(num_clients,select_num,total_timesteps):
    args = get_args()
    hyperparameters = {
        'timesteps_per_batch': 100,
        'max_timesteps_per_episode': 20,
        'gamma': 0.99,
        'n_updates_per_iteration': 10,
        'lr': 3e-4,
        'clip': 0.2,
        'render': True,
        'render_every_i': 10
    }

    env = FederatedLearningEnv(num_clients=num_clients, k=select_num)

    if args.mode == 'train':
        train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model,total_timesteps=total_timesteps)
    else:
        test(env=env, actor_model=args.actor_model)

if __name__ == '__main__':
    total_timesteps=100
    rl_main(20,5,total_timesteps)



