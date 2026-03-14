import argparse
import os
import pprint

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.exploration import GaussianNoise
from tianshou.policy import reach_avoid_DDPGPolicy as DDPGPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic

# best network structure: control net: 256, 256, 256; disturbance net: 512, 512, 512; critic net: 512, 512, 512
# with batch size 512, exploration_noise 0.1, step_per_epoch = buffer_size 40000, gamma 0.95

# another best network structure: control net: 256, 256, 256; disturbance net: 512, 512, 512; critic net: 512, 512, 512, 512
# with batch size 512, exploration_noise 0.5, step_per_epoch = buffer_size 40000, gamma 0.95
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='ra_droneracing_Game-v5') # v4 or v5
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=40000)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--exploration-noise', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--step-per-epoch', type=int, default=40000)
    parser.add_argument('--step-per-collect', type=int, default=8)
    parser.add_argument('--update-per-step', type=float, default=0.125)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[256,256])
    parser.add_argument('--control-net', type=int, nargs='*', default=[512, 512, 512, 512])
    parser.add_argument('--disturbance-net', type=int, nargs='*', default=[512, 512, 512, 512])
    parser.add_argument('--critic-net', type=int, nargs='*', default=[512,512,512, 512])
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--is-game', type=bool, default=False) # it will be set automatically
    parser.add_argument('--rew-norm', action="store_true", default=False)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--actor-gradient-steps', type=int, default=1)
    parser.add_argument('--actor-activation', type=str, default='ReLU')
    parser.add_argument('--critic-activation', type=str, default='ReLU')
    parser.add_argument('--is-game-baseline', type=bool, default=False) # it will be set automatically
    parser.add_argument('--is-baseline', type=bool, default=False) # it will be set automatically
    parser.add_argument('--is-dqn', type=bool, default=False) # it will be set automatically
    parser.add_argument('--is-dqn-baseline', type=bool, default=False) # it will be set automatically
    parser.add_argument('--tdthree', type=bool, default=False) # it will be set automatically
    parser.add_argument(
        '--device', type=str, default='cpu'
    )
    parser.add_argument('--kwargs', type=str, default='{}')
    args = parser.parse_known_args()[0]
    return args


args=get_args()
if args.task.split('-')[0].split("_")[-1] == 'Game':
    args.is_game = True

env = gym.make(args.task)
args.state_shape = env.observation_space.shape or env.observation_space.n
args.action_shape = env.action_space.shape or env.action_space.n
if not args.is_dqn and not args.is_dqn_baseline:
    args.max_action = env.action_space.high[0]

if args.is_game:
    args.action1_shape = env.action1_space.shape or env.action1_space.n
    args.action2_shape = env.action2_space.shape or env.action2_space.n
    args.max_action1 = env.action1_space.high[0]
    args.max_action2 = env.action2_space.high[0]

if args.reward_threshold is None:
    default_reward_threshold = {"Pendulum-v0": -250, "Pendulum-v1": -250}
    args.reward_threshold = default_reward_threshold.get(
        args.task, env.spec.reward_threshold
    )
# you can also use tianshou.env.SubprocVectorEnv
# train_envs = gym.make(args.task)
train_envs = DummyVectorEnv(
    [lambda: gym.make(args.task) for _ in range(args.training_num)]
)
# test_envs = gym.make(args.task)
test_envs = DummyVectorEnv(
    [lambda: gym.make(args.task) for _ in range(args.test_num)]
)
# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
train_envs.seed(args.seed)
test_envs.seed(args.seed)
# model

if args.actor_activation == 'ReLU':
    actor_activation = torch.nn.ReLU
elif args.actor_activation == 'Tanh':
    actor_activation = torch.nn.Tanh
elif args.actor_activation == 'Sigmoid':
    actor_activation = torch.nn.Sigmoid
elif args.actor_activation == 'SiLU':
    actor_activation = torch.nn.SiLU

if args.critic_activation == 'ReLU':
    critic_activation = torch.nn.ReLU
elif args.critic_activation == 'Tanh':
    critic_activation = torch.nn.Tanh
elif args.critic_activation == 'Sigmoid':
    critic_activation = torch.nn.Sigmoid
elif args.critic_activation == 'SiLU':
    critic_activation = torch.nn.SiLU

if args.critic_net is not None:
    critic_net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.critic_net,
        activation=critic_activation,
        concat=True,
        device=args.device
    )
else:
    critic_net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        activation=critic_activation,
        concat=True,
        device=args.device
    )

critic = Critic(critic_net, device=args.device).to(args.device)
critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

critic1 = Critic(critic_net, device=args.device).to(args.device)
critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
critic2 = Critic(critic_net, device=args.device).to(args.device)
critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)    
if args.control_net is None:
    args.control_net = args.hidden_sizes
if args.disturbance_net is None:
    args.disturbance_net = args.hidden_sizes
if args.critic_net is None:
    args.critic_net = args.hidden_sizes
# import pdb; pdb.set_trace()
if args.tdthree:
    from tianshou.policy import reach_avoid_game_TD3Policy as DDPGPolicy
    actor1_net = Net(args.state_shape, hidden_sizes=args.control_net, activation=actor_activation, device=args.device)
    actor1 = Actor(
        actor1_net, args.action1_shape, max_action=args.max_action1, device=args.device
    ).to(args.device)
    actor1_optim = torch.optim.Adam(actor1.parameters(), lr=args.actor_lr)
    actor2_net = Net(args.state_shape, hidden_sizes=args.disturbance_net, activation=actor_activation, device=args.device)
    actor2 = Actor(
        actor2_net, args.action2_shape, max_action=args.max_action1, device=args.device
    ).to(args.device)
    actor2_optim = torch.optim.Adam(actor2.parameters(), lr=args.actor_lr)
    # import pdb; pdb.set_trace()
    policy = DDPGPolicy(
    critic1=critic1,
    critic1_optim=critic1_optim,
    critic2=critic2,
    critic2_optim=critic2_optim,
    tau=args.tau,
    gamma=args.gamma,
    exploration_noise=GaussianNoise(sigma=args.exploration_noise),
    reward_normalization=args.rew_norm,
    estimation_step=args.n_step,
    action_space=env.action_space,
    actor1=actor1,
    actor1_optim=actor1_optim,
    actor2=actor2,
    actor2_optim=actor2_optim,
    actor_gradient_steps=args.actor_gradient_steps,
    )
    log_path = os.path.join(args.logdir, args.task, 'td3_reach_avoid_actor_activation_{}_critic_activation_{}_game_gd_steps_{}_tau_{}_training_num_{}_buffer_size_{}_gamma_{}'.format(
        args.actor_activation, 
        args.critic_activation, 
        args.actor_gradient_steps,args.tau, 
        args.training_num, 
        args.buffer_size,
        args.gamma)
    )
elif args.is_game:    
    if args.is_game_baseline:
        from tianshou.policy import reach_avoid_game_DDPGPolicy_baseline as DDPGPolicy
    else:
        from tianshou.policy import reach_avoid_game_DDPGPolicy as DDPGPolicy
    actor1_net = Net(args.state_shape, hidden_sizes=args.control_net, activation=actor_activation, device=args.device)
    actor1 = Actor(
        actor1_net, args.action1_shape, max_action=args.max_action1, device=args.device
    ).to(args.device)
    actor1_optim = torch.optim.Adam(actor1.parameters(), lr=args.actor_lr)
    actor2_net = Net(args.state_shape, hidden_sizes=args.disturbance_net, activation=actor_activation, device=args.device)
    actor2 = Actor(
        actor2_net, args.action2_shape, max_action=args.max_action1, device=args.device
    ).to(args.device)
    actor2_optim = torch.optim.Adam(actor2.parameters(), lr=args.actor_lr)
    
    policy = DDPGPolicy(
    critic,
    critic_optim,
    tau=args.tau,
    gamma=args.gamma,
    exploration_noise=GaussianNoise(sigma=args.exploration_noise),
    reward_normalization=args.rew_norm,
    estimation_step=args.n_step,
    action_space=env.action_space,
    actor1=actor1,
    actor1_optim=actor1_optim,
    actor2=actor2,
    actor2_optim=actor2_optim,
    actor_gradient_steps=args.actor_gradient_steps,
    )
    if args.is_game_baseline:
        log_path = os.path.join(args.logdir, args.task, 'baseline_ddpg_reach_avoid_actor_activation_{}_critic_activation_{}_game_gd_steps_{}_tau_{}_training_num_{}_buffer_size_{}'.format(
        args.actor_activation, 
        args.critic_activation, 
        args.actor_gradient_steps,args.tau, 
        args.training_num, 
        args.buffer_size)
    )
    elif args.critic_net is not None:
        log_path = os.path.join(args.logdir, args.task, 'ddpg_reach_avoid_actor_activation_{}_critic_activation_{}_game_gd_steps_{}_tau_{}_training_num_{}_buffer_size_{}_c_net_{}_{}_a1_{}_{}_a2_{}_{}'.format(
        args.actor_activation, 
        args.critic_activation, 
        args.actor_gradient_steps,args.tau, 
        args.training_num, 
        args.buffer_size,
        args.critic_net[0],
        len(args.critic_net),
        args.control_net[0],
        len(args.control_net),
        args.disturbance_net[0],
        len(args.disturbance_net))
    )
    else:
        log_path = os.path.join(args.logdir, args.task, 'ddpg_reach_avoid_actor_activation_{}_critic_activation_{}_game_gd_steps_{}_tau_{}_training_num_{}_buffer_size_{}'.format(
        args.actor_activation, 
        args.critic_activation, 
        args.actor_gradient_steps,args.tau, 
        args.training_num, 
        args.buffer_size)
    )
elif args.is_baseline:
    from tianshou.policy import reach_avoid_DDPGPolicy_baseline as DDPGPolicy
    actor_net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, activation=actor_activation, device=args.device)
    actor = Actor(
        actor_net, args.action_shape, max_action=args.max_action, device=args.device
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    policy = DDPGPolicy(
    actor,
    actor_optim,
    critic,
    critic_optim,
    tau=args.tau,
    gamma=args.gamma,
    exploration_noise=GaussianNoise(sigma=args.exploration_noise),
    reward_normalization=args.rew_norm,
    estimation_step=args.n_step,
    action_space=env.action_space
    )
    log_path = os.path.join(args.logdir, args.task, 'baseline_ddpg_reach_avoid_actor_activation_{}_critic_activation_{}_tau_{}_training_num_{}_buffer_size_{}_gamma_{}'.format(
    args.actor_activation, 
    args.critic_activation, 
    args.tau, 
    args.training_num,
    args.buffer_size,
    args.gamma)
    )
elif args.is_dqn:
    from tianshou.policy import RA_control_no_disturbance_policy as DQNPolicy
    net = Net(
    args.state_shape,
    args.action_shape,
    hidden_sizes=args.hidden_sizes,
    device=args.device,
    ).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.actor_lr)
    policy = DQNPolicy(
        net,
        optim,
        args.gamma,
        1,
        target_update_freq=args.target_update_freq
    )
    log_path = os.path.join(args.logdir, args.task, 'dqn_reach_avoid_actor_activation_{}_critic_activation_{}_tau_{}_training_num_{}_buffer_size_{}_gamma_{}'.format(
        args.actor_activation, 
        args.critic_activation, 
        args.tau, 
        args.training_num,
        args.buffer_size,
        args.gamma)
    )
elif args.is_dqn_baseline:
    from tianshou.policy import RA_control_no_disturbance_policy_baseline as DQNPolicy
    net = Net(
    args.state_shape,
    args.action_shape,
    hidden_sizes=args.hidden_sizes,
    device=args.device,
    ).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.actor_lr)
    policy = DQNPolicy(
        net,
        optim,
        args.gamma,
        1,
        target_update_freq=args.target_update_freq
    )
    log_path = os.path.join(args.logdir, args.task, 'baseline_dqn_reach_avoid_actor_activation_{}_critic_activation_{}_tau_{}_training_num_{}_buffer_size_{}_gamma_{}'.format(
        args.actor_activation, 
        args.critic_activation, 
        args.tau, 
        args.training_num,
        args.buffer_size,
        args.gamma)
    )
else:
    from tianshou.policy import reach_avoid_DDPGPolicy as DDPGPolicy
    actor_net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, activation=actor_activation, device=args.device)
    actor = Actor(
        actor_net, args.action_shape, max_action=args.max_action, device=args.device
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    
    policy = DDPGPolicy(
    actor,
    actor_optim,
    critic,
    critic_optim,
    tau=args.tau,
    gamma=args.gamma,
    exploration_noise=GaussianNoise(sigma=args.exploration_noise),
    reward_normalization=args.rew_norm,
    estimation_step=args.n_step,
    action_space=env.action_space
    )
    log_path = os.path.join(args.logdir, args.task, 'ddpg_reach_avoid_actor_activation_{}_critic_activation_{}_tau_{}_training_num_{}_buffer_size_{}_gamma_{}'.format(
        args.actor_activation, 
        args.critic_activation, 
        args.tau, 
        args.training_num,
        args.buffer_size,
        args.gamma)
    )
    if args.critic_net is not None:
        log_path = os.path.join(args.logdir, args.task, 'ddpg_reach_avoid_actor_activation_{}_critic_activation_{}_game_gd_steps_{}_tau_{}_training_num_{}_buffer_size_{}_critic_net_{}_{}'.format(
        args.actor_activation, 
        args.critic_activation, 
        args.actor_gradient_steps,args.tau, 
        args.training_num, 
        args.buffer_size,
        args.critic_net[0],
        len(args.critic_net))
    )

# collector
train_collector = Collector(
    policy,
    train_envs,
    VectorReplayBuffer(args.buffer_size, len(train_envs)),
    exploration_noise=True
)
test_collector = Collector(policy, test_envs)
# log
log_path = os.path.join(args.logdir, args.task, 'ddpg')
writer = SummaryWriter(log_path)
logger = TensorboardLogger(writer)



def save_best_fn(policy, epoch=0):
    torch.save(policy.state_dict(), os.path.join(log_path, 'policy_{}_{}_{}.pth'.format(args.hidden_sizes[0], 
                                                                                    len(args.hidden_sizes),
                                                                                    epoch)))
#     torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

def stop_fn(mean_rewards):
    return False




# id_epoch = 1300 # v5, safe distance = 0.1
id_epoch = 110 # v5, safe distance = 0.2

envs = gym.make(args.task)
policy.load_state_dict(torch.load('log/{}/ddpg_reach_avoid_actor_activation_{}_critic_activation_{}_game_gd_steps_{}_tau_{}_training_num_{}_buffer_size_{}_c_net_{}_{}_a1_{}_{}_a2_{}_{}_gamma_{}/policy.pth'.format(
        args.task,
        args.actor_activation,
        args.critic_activation, 
        args.actor_gradient_steps,
        args.tau,
        args.training_num,
        args.buffer_size,
        args.critic_net[0],
        len(args.critic_net),
        args.control_net[0],
        len(args.control_net),
        args.disturbance_net[0],
        len(args.disturbance_net),
        args.gamma,
        ),
        map_location=torch.device('cpu')
    )
)

# old policy loading, changed on June 25th, 2024:
# policy.load_state_dict(torch.load('log/{}/ddpg_reach_avoid_actor_activation_{}_critic_activation_{}_game_gd_steps_{}_tau_{}_training_num_{}_buffer_size_{}_c_net_{}_{}_a1_{}_{}_a2_{}_{}_gamma_{}/noise_{}_actor_lr_{}_critic_lr_{}_policy_{}_{}_{}_batch_{}_step_per_epoch_{}_kwargs_{}_seed_{}.pth'.format(
#         args.task,
#         args.actor_activation,
#         args.critic_activation, 
#         args.actor_gradient_steps,
#         args.tau,
#         args.training_num,
#         args.buffer_size,
#         args.critic_net[0],
#         len(args.critic_net),
#         args.control_net[0],
#         len(args.control_net),
#         args.disturbance_net[0],
#         len(args.disturbance_net),
#         args.gamma,
#         args.exploration_noise,
#         args.actor_lr,
#         args.critic_lr,
#         args.hidden_sizes[0],
#         len(args.hidden_sizes),
#         id_epoch,
#         args.batch_size,
#         args.step_per_epoch,
#         args.kwargs,
#         args.seed),
#         map_location=torch.device('cpu')
#     )
# )

from tianshou.data import Batch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import math

def find_a(state):
    tmp_obs = np.array(state).reshape(1,-1)
    tmp_batch = Batch(obs = tmp_obs, info = Batch())
    tmp = policy(tmp_batch, model = "actor_old").act
    act = policy.map_action(tmp).cpu().detach().numpy().flatten()
    envs.reset(initial_state = state)
    tmp, rew, _, _, info  = envs.step(act)
    if info["constraint"]<0:
        print("constraint is violated!")
    if rew > 0:
        print("success!")
    return act, rew, info["constraint"]

# if distance less than 0.2, then separate!
def filtered_target_velocity(state):
    # check who is in front
    if state[1] > state[7]:
        vy_star1 = state[4]
        vy_star2 = 0.0
    else:
        vy_star1 = 0.0
        vy_star2 = state[10]
    return vy_star1, vy_star2

'''
The above is the place where we define the policy and reload policy.

The following is the place where we deploy and fly the policy!
'''





import numpy as np
from pycrazyswarm import *
import uav_trajectory
import time
import os
import csv
import socket
import pickle
vel = None

real_experiment = False
real_position =True

def write_csv(data, file_path='traj_data.csv'):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)
        f.flush()  # Ensure data is written to disk


def executeTrajectory(timeHelper, cf1, cf2, horizon, stopping_horizon, rate=100, offset=np.zeros(3)):
    K1 = np.array([3.1127])
    K2 = np.array([ 9.1704,   16.8205])
    total_horizon = horizon + stopping_horizon
    X_list[:,0] = np.array([
        cf1.initialPosition[0] + offset[0],
        cf1.initialPosition[1] + offset[1],
        cf1.initialPosition[2] + offset[2],
        0.0,
        0.0,
        0.0,
        cf2.initialPosition[0] + offset[0],
        cf2.initialPosition[1] + offset[1],
        cf2.initialPosition[2] + offset[2],
        0.0,
        0.0,
        0.0
    ])
    x_star = 0.0
    vx_star = 0.0
    vy_star1 = 0.0
    vy_star2 = 0.3
    z_star = offset[2]
    vz_star = 0.0
    dt = 0.1
    
    start_time = timeHelper.time() # get current time
    next_X = np.zeros((12))
    velocity1 = [0.0, 0.7, 0.0]# initial velocity, old vy = 0.2
    velocity2 = [0.0, 0.3, 0.0]# initial velocity, old vy = 0.2
    state1 = cf1.initialPosition + offset
    state2 = cf2.initialPosition + offset
    onboard_velocity1 = [0.0, 0.0, 0.0]
    onboard_velocity2 = [0.0, 0.0, 0.0]
    # safe_reached = False
    global safe_reached
    correction_term=np.array([
        0.0, 0.0, 3.0, 0.0, z_star, 0.0, 
        0.0, 0.0, 3.0, 0.0, z_star, 0.0
    ])
    history_constraint = []
    emergency_stop = False
    while not timeHelper.isShutdown(): # while the timeHelper is not shutdown
        t = timeHelper.time() - start_time # get the time
        if t > (total_horizon)*dt: # if the time is greater than the duration of the trajectory
            break 
        if real_position:
            state1 = cf1.position()
            state2 = cf2.position()
        if real_experiment:
            pass
        else:
            state1 = cf1.position()
            state2 = cf2.position()
            onboard_velocity1 = velocity1
            onboard_velocity2 = velocity2
            pass
        
        X = np.array([state1[0], state1[1], state1[2], 
                    velocity1[0], velocity1[1], velocity1[2],
                    state2[0], state2[1], state2[2],
                    velocity2[0], velocity2[1], velocity2[2]
                    ])
        if t > horizon*dt:
            vy_star1 = 0.0
            vy_star2 = 0.0
        NN_state = np.array([
            state1[0], velocity1[0], state1[1]-3.0, velocity1[1], state1[2]-z_star, velocity1[2],
            state2[0], velocity2[0], state2[1]-3.0, velocity2[1], state2[2]-z_star, velocity2[2]
        ])
        
        NN_action, reward, constraint = find_a(NN_state)
        # safety filter:
        if not safe_reached:
            history_constraint.append(constraint)
        
        if constraint < 0 and not safe_reached:
            print("constraint violated!")
            vy_star1, vy_star2 = filtered_target_velocity(X)
            emergency_stop = True
        # import pdb; pdb.set_trace()
        if reward > 0 and min(history_constraint) > 0:
            safe_reached = True
            print("safe reached!")
        if safe_reached:
            action1 = [
                K2@np.array([x_star-state1[0], vx_star-velocity1[0]]),
                K1@np.array([vy_star1-velocity1[1]]),
                K2@np.array([z_star-state1[2], vz_star-velocity1[2]])
            ]
        else:    
            action1 = [
                NN_action[0],
                NN_action[1],
                NN_action[2]
            ]
            vy_star1 = velocity1[1]
        
        action2 = [
            K2@np.array([x_star-state2[0], vx_star-velocity2[0]]),
            K1@np.array([vy_star2-velocity2[1]]),
            K2@np.array([z_star-state2[2], vz_star-velocity2[2]])
        ]
        next_X = X + dt*np.array([
            velocity1[0],
            velocity1[1],
            velocity1[2],
            action1[0],
            action1[1],
            action1[2],
            velocity2[0],
            velocity2[1],
            velocity2[2],
            action2[0],
            action2[1],
            action2[2]
        ])
        velocity1 = [next_X[3], next_X[4], next_X[5]]
        velocity2 = [next_X[9], next_X[10], next_X[11]]
        state1 = [next_X[0], next_X[1], next_X[2]]
        state2 = [next_X[6], next_X[7], next_X[8]]
        print(X)
        if emergency_stop:
            cf1.cmdVelocityWorld(np.array([0.0, vy_star1, 0.0]), yawRate=0.0)
            cf2.cmdVelocityWorld(np.array([0.0, vy_star2, 0.0]), yawRate=0.0)
        else:
            cf1.cmdFullState(
                next_X[0:3],  # position
                next_X[3:6],  # velocity
                np.zeros((3,)),  # acceleration
                0.0,  # yaw
                np.zeros((3,)) # Omega
            ) 
            cf2.cmdFullState(
                next_X[6:9],  # position
                next_X[9:12],  # velocity
                np.zeros((3,)),  # acceleration
                0.0,  # yaw
                np.zeros((3,)) # Omega
            )
        # import pdb; pdb.set_trace()
        X_list[:,int(t/dt)] = np.array([
            state1[0],
            state1[1],
            state1[2],
            onboard_velocity1[0],
            onboard_velocity1[1],
            onboard_velocity1[2],
            state2[0],
            state2[1],
            state2[2],
            onboard_velocity2[0],
            onboard_velocity2[1],
            onboard_velocity2[2]
        ])
        timeHelper.sleepForRate(rate)


if __name__ == "__main__":
    global safe_reached
    safe_reached = False
    horizon = 40 # 30 # 50
    stopping_horizon = 10 # 10 #15
    X_list = np.zeros((12,horizon+stopping_horizon))
    swarm = Crazyswarm() # create a Crazyswarm object
    timeHelper = swarm.timeHelper
    cf1 = swarm.allcfs.crazyflies[0]
    cf2 = swarm.allcfs.crazyflies[1]

    rate = 10.0#30.0
    Z = 0.5

    cf1.takeoff(targetHeight=Z, duration=Z+2.0)
    cf2.takeoff(targetHeight=Z, duration=Z+2.0)
    timeHelper.sleep(Z+2.0)
    
    executeTrajectory(timeHelper, cf1,cf2, horizon, stopping_horizon, rate, offset=np.array([0, 0, Z]))
    
    cf1.notifySetpointsStop()
    cf2.notifySetpointsStop()
    
    cf1.land(targetHeight=0.05, duration=Z+1.5)
    cf2.land(targetHeight=0.05, duration=Z+1.5)
    timeHelper.sleep(Z+2.0)
    # import pdb; pdb.set_trace()
    # name the data file with the current time and date
    # get the current time
    current_time = time.strftime("%Y%m%d-%H%M%S")
    if safe_reached:
        np.savetxt('safe_reached_nn_pid_{}.csv'.format(current_time), X_list, delimiter=',', header='')
    else:
        np.savetxt('failed_nn_pid_{}.csv'.format(current_time), X_list, delimiter=',', header='')