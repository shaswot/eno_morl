import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random
class off_ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done, pref):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, pref)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, pref = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, pref
    
    def __len__(self):
        return len(self.buffer)
################################################################################
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
################################################################################
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.1, max_sigma=0.03, min_sigma=0.03, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = -1#action_space.low
        self.high         = +1#action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)
    
#https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
################################################################################
class ENoise(object):
    def __init__(self, action_space, max_sigma=0.15, min_sigma=0.01, decay_period=None):
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.low          = -1#action_space.low
        self.high         = 1#action_space.high
    
    def get_action(self, action, t=0):
        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        action =  np.random.normal(loc=action, scale=sigma)
        return np.clip(action, self.low, self.high)
################################################################################
class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, device, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
################################################################################
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, device, init_w=3e-3):
        super(PolicyNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
        self.device = device
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x
    
    def get_action(self, state):
        state  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0, 0]
################################################################################
################################################################################
class small_ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, device, init_w=3e-3):
        super(small_ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
################################################################################
class small_PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, device, init_w=3e-3):
        super(small_PolicyNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
        self.device = device
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x
    
    def get_action(self, state):
        state  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0, 0]
################################################################################
def ddpg_update(value_net,
                target_value_net,
                policy_net,
                target_policy_net,
                value_optimizer,
                policy_optimizer,
                value_criterion,
                batch_size,
                replay_buffer,
                device,
                writer,
                frame_idx,
                gamma=0.99,
                min_value=-np.inf,
                max_value=np.inf,
                soft_tau=1e-2,
                policy_clipgrad=None,
                value_clipgrad=None):
    
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    
    state      = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action     = torch.FloatTensor(action).unsqueeze(1).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    policy_loss = value_net(state, policy_net(state))
    policy_loss = -policy_loss.mean()
    if writer is not None:
        writer.add_scalar("Loss/policy_loss", -policy_loss, frame_idx)
   
    next_action    = target_policy_net(next_state)
    target_value   = target_value_net(next_state, next_action.detach())
    expected_value = reward.view(batch_size,-1) + (1.0 - done) * gamma * target_value
    expected_value = torch.clamp(expected_value, min_value, max_value)

    value = value_net(state, action)
    value_loss = value_criterion(value, expected_value.detach())
    if writer is not None:
        writer.add_scalar("Loss/value_loss", value_loss, frame_idx)


    policy_optimizer.zero_grad()
    policy_loss.backward()
    if policy_clipgrad is not None:
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=policy_clipgrad)
    policy_avg_grad=[]
    policy_max_grad=[]
    with torch.no_grad():
        for n, p in policy_net.named_parameters():
            policy_avg_grad.append(p.grad.abs().mean().item())
            policy_max_grad.append(p.grad.abs().max().item())
    if writer is not None:
        writer.add_scalar("Grad/policy_gmean", np.mean(policy_avg_grad),frame_idx)
        writer.add_scalar("Grad/policy_gmax",  np.max(policy_max_grad),frame_idx)
    policy_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    if value_clipgrad is not None:
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=value_clipgrad)
    value_avg_grad=[]
    value_max_grad=[]
    with torch.no_grad():
        for n, p in value_net.named_parameters():
            value_avg_grad.append(p.grad.abs().mean().item())
            value_max_grad.append(p.grad.abs().max().item())
    if writer is not None:
        writer.add_scalar("Grad/value_gmean", np.mean(value_avg_grad),frame_idx)
        writer.add_scalar("Grad/value_gmax",  np.max(value_max_grad),frame_idx)
    value_optimizer.step()

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
################################################################################
def nn_update(netname,
                batch,
                value_net,
                target_value_net,
                policy_net,
                target_policy_net,
                value_optimizer,
                policy_optimizer,
                value_criterion,
                device,
                writer=None,
                frame_idx=None,
                gamma=0.99,
                min_value=-np.inf,
                max_value=np.inf,
                soft_tau=1e-2,
                policy_clipgrad=None,
                value_clipgrad=None):
    
    state, action, reward, next_state, done, pref = batch
    
    batch_size = len(done)
    
    state      = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action     = torch.FloatTensor(action).unsqueeze(1).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
    pref       = torch.FloatTensor(pref).unsqueeze(1).to(device)


    policy_loss = value_net(state, policy_net(state))
    policy_loss = -policy_loss.mean()
    if writer is not None:
        writer.add_scalar("Loss/"+netname+"_policy_loss", -policy_loss, frame_idx)
   
    next_action    = target_policy_net(next_state)
    target_value   = target_value_net(next_state, next_action.detach())
    expected_value = reward.view(batch_size,-1) + (1.0 - done) * gamma * target_value * pref
    expected_value = torch.clamp(expected_value, min_value, max_value)

    value = value_net(state, action)
    value_loss = value_criterion(value, expected_value.detach())
    if writer is not None:
        writer.add_scalar("Loss/"+netname+"_value_loss", value_loss, frame_idx)
    policy_optimizer.zero_grad()
    policy_loss.backward()
    if policy_clipgrad is not None:
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=policy_clipgrad)
    policy_avg_grad=[]
    policy_max_grad=[]
    with torch.no_grad():
        for n, p in policy_net.named_parameters():
            policy_avg_grad.append(p.grad.abs().mean().item())
            policy_max_grad.append(p.grad.abs().max().item())
    if writer is not None:
        writer.add_scalar("Grad/"+netname+"_policy_gmean", np.mean(policy_avg_grad),frame_idx)
        writer.add_scalar("Grad/"+netname+"_policy_gmax",  np.max(policy_max_grad),frame_idx)
    policy_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    if value_clipgrad is not None:
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=value_clipgrad)
    value_avg_grad=[]
    value_max_grad=[]
    with torch.no_grad():
        for n, p in value_net.named_parameters():
            value_avg_grad.append(p.grad.abs().mean().item())
            value_max_grad.append(p.grad.abs().max().item())
    if writer is not None:
        writer.add_scalar("Grad/"+netname+"_value_gmean", np.mean(value_avg_grad),frame_idx)
        writer.add_scalar("Grad/"+netname+"_value_gmax",  np.max(value_max_grad),frame_idx)
    value_optimizer.step()

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
################################################################################
def calc_values_of_states(states, actions, net, device):
    mean_values = []
    eval_batch_size = 64
    split_state = torch.split(states,eval_batch_size)
    split_action = torch.split(actions,eval_batch_size)
    no_of_blocks = len(split_state)
    with torch.no_grad():
        for i in range(no_of_blocks):
            mean_values.append(net(split_state[i],split_action[i]).mean().item())
    return np.mean(mean_values)
################################################################################
def nn_update2(netname,
                batch,
                value_net,
                target_value_net,
                policy_net,
                target_policy_net,
                value_optimizer,
                policy_optimizer,
                value_criterion,
                device,
                writer=None,
                frame_idx=None,
                gamma=0.99,
                min_value=-np.inf,
                max_value=np.inf,
                soft_tau=1e-2,
                policy_clipgrad=None,
                value_clipgrad=None):
    
    state, action, reward, next_state, done, pref = batch
    
    batch_size = len(done)
    
    state      = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action     = torch.FloatTensor(action).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
    pref       = torch.FloatTensor(pref).unsqueeze(1).to(device)


    policy_loss = value_net(state, policy_net(state))
    policy_loss = -policy_loss.mean()
    if writer is not None:
        writer.add_scalar("Loss/"+netname+"_policy_loss", -policy_loss, frame_idx)
   
    next_action    = target_policy_net(next_state)
    target_value   = target_value_net(next_state, next_action.detach())
    expected_value = reward.view(batch_size,-1) + (1.0 - done) * gamma * target_value * pref
    expected_value = torch.clamp(expected_value, min_value, max_value)

    value = value_net(state, action)
    value_loss = value_criterion(value, expected_value.detach())
    if writer is not None:
        writer.add_scalar("Loss/"+netname+"_value_loss", value_loss, frame_idx)
    policy_optimizer.zero_grad()
    policy_loss.backward()
    if policy_clipgrad is not None:
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=policy_clipgrad)
    policy_avg_grad=[]
    policy_max_grad=[]
    with torch.no_grad():
        for n, p in policy_net.named_parameters():
            policy_avg_grad.append(p.grad.abs().mean().item())
            policy_max_grad.append(p.grad.abs().max().item())
    if writer is not None:
        writer.add_scalar("Grad/"+netname+"_policy_gmean", np.mean(policy_avg_grad),frame_idx)
        writer.add_scalar("Grad/"+netname+"_policy_gmax",  np.max(policy_max_grad),frame_idx)
    policy_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    if value_clipgrad is not None:
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=value_clipgrad)
    value_avg_grad=[]
    value_max_grad=[]
    with torch.no_grad():
        for n, p in value_net.named_parameters():
            value_avg_grad.append(p.grad.abs().mean().item())
            value_max_grad.append(p.grad.abs().max().item())
    if writer is not None:
        writer.add_scalar("Grad/"+netname+"_value_gmean", np.mean(value_avg_grad),frame_idx)
        writer.add_scalar("Grad/"+netname+"_value_gmax",  np.max(value_max_grad),frame_idx)
    value_optimizer.step()

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
################################################################################
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
################################################################################
