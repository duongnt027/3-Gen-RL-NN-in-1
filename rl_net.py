import torch
import random
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from sklearn.neighbors import KernelDensity
from torch.distributions import MultivariateNormal, Normal

from utils import tensor_pop

class AnorEnv:
    def __init__(self, dataset, generator, detector, shuffle=True, device="cpu"):
        self.device = device
        
        # Get all data from dataset
        self.X = dataset.X.to(device)
        self.y = dataset.y.to(device)
        
        if shuffle:
            indices = torch.randperm(len(self.X))
            self.X = self.X[indices]
            self.y = self.y[indices]

        self.generator = generator
        self.detector = detector
        
        # Define observation and action space dimensions
        self.observation_space = torch.zeros(generator.in_dim)  # Same as input dimension
        self.action_space = torch.zeros(2)  # [mu, sigma] for sampling
        
        self.current_state = None
        self.current_step = 0
        self.max_steps = len(self.X)
        
        # Keep track of available states and generated buffer
        self.states = self.X.clone()
        self.buffer_X = self.X.clone()

    def reset(self):
        self.current_step = 0
        self.states = self.X.clone()
        self.buffer_X = self.X.clone()
        
        if self.states.shape[0] > 0:
            idx = random.randint(0, self.states.shape[0] - 1)
            self.current_state, self.states = tensor_pop(self.states, idx)
        else:
            self.current_state = torch.randn(self.X.shape[1]).to(self.device)
            
        return self.current_state, {}

    def step(self, action):
        self.current_step += 1
        
        terminated = False
        truncated = False
        info = {}

        action = torch.tensor(action)

        # Action contains [mu, sigma] for normal distribution
        mu, sigma = action[0], torch.abs(action[1]) + 1e-6  # Ensure sigma > 0
        
        # Encode current state to latent space
        current_state_batch = self.current_state.unsqueeze(0)
        y_batch = torch.tensor([[1.0]], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            mean_z, logvar_z = self.generator.encode(current_state_batch, y_batch)
            # print(mean_z, logvar_z)
            z = self.generator.reparameterize(mean_z, logvar_z)
        # print(z)
        # Sample perturbation in latent space
        dist = Normal(mu, sigma)
        delta = dist.sample(z.shape).to(self.device)
        z_new = z + delta
        
        # Decode to get new sample
        with torch.no_grad():
            x_new = self.generator.decode(z_new, y_batch)
        
        # Calculate reward
        reward = self._calculate_reward(x_new.squeeze(0))
        
        # Check termination conditions
        if self.states.shape[0] == 0:
            terminated = True
        
        if self.current_step >= self.max_steps:
            truncated = True
            
        # Check if generated sample is feasible and add to buffer
        # if self._check_feasible(x_new.squeeze(0)):
        if True:
            self.buffer_X = torch.cat([self.buffer_X, x_new], dim=0)
            info['added_to_buffer'] = True
        else:
            info['added_to_buffer'] = False
        
        # Get next state
        if not terminated and self.states.shape[0] > 0:
            idx = random.randint(0, self.states.shape[0] - 1)
            self.current_state, self.states = tensor_pop(self.states, idx)
        
        return x_new.squeeze(0), reward, terminated, truncated, info

    def _calculate_reward(self, x):
        # Calculate reward = diversity - detector_prob
        with torch.no_grad():
            detect_prob = self.detector.infer_fn(x).squeeze()
            diversity_score = self._calculate_diversity(x)
        return diversity_score - detect_prob.item()

    def _calculate_diversity(self, x):
        # Calculate diversity as mean distance to existing samples
        if self.buffer_X.shape[0] == 0:
            return 1.0
            
        dists = torch.norm(self.buffer_X - x, dim=1)
        # Remove distances that are too small (likely the same sample)
        dists = dists[dists > 1e-6]
        
        if len(dists) == 0:
            return 1.0
            
        return dists.mean().item()

    def _check_feasible(self, x, threshold=0.01):
        try:
            # Convert to numpy for sklearn
            X_np = self.X.cpu().numpy()
            x_np = x.cpu().numpy().reshape(1, -1)
            
            # Fit KDE on original data
            kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X_np)
            log_prob = kde.score_samples(x_np)
            prob = np.exp(log_prob[0])
            
            return prob > threshold
        except:
            # Fallback: simple distance-based feasibility
            print("min_dist")
            min_dist = torch.min(torch.norm(self.X - x.unsqueeze(0), dim=1))
            return min_dist < 2.0  # Reasonable threshold
        
class GenPPO:
    def __init__(self, actor_net, critic_net, env, **hyperparameters):
        self._init_hyperparameters(hyperparameters)
        
        self.env = env
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # Set device first
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor_net = actor_net
        self.actor_net = self.actor_net.to(self.device)
        self.critic_net = critic_net
        self.critic_net = self.critic_net.to(self.device)

        self.actor_optim = Adam(self.actor_net.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic_net.parameters(), lr=self.lr)

        # Covariance matrix for action sampling
        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5, device=self.device)
        self.cov_mat = torch.diag(self.cov_var)

    def get_action(self, observation):
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        else:
            observation = observation.to(self.device)

        mean_sigma = self.actor_net(observation)
        dist = MultivariateNormal(mean_sigma, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.cpu().detach().numpy(), log_prob.detach()
        
    def _init_hyperparameters(self, hyperparameters):
        # Default hyperparameters
        self.timesteps_per_batch = 2048        
        self.max_timesteps_per_episode = 200   
        self.n_updates_per_iteration = 10      
        self.lr = 3e-4                         
        self.gamma = 0.99                      
        self.clip = 0.2                        
        self.render = False                    
        self.render_every_i = 10              
        self.save_freq = 10                   
        self.seed = None            
        
        # Override with provided hyperparameters
        for param, val in hyperparameters.items():
            setattr(self, param, val)
        
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        t = 0  # Timesteps run so far this batch
        print(self.timesteps_per_batch)
        print(t)
        
        while t < self.timesteps_per_batch:
            ep_rews = []
            
            obs, _ = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1
                # Convert obs to numpy if it's a tensor
                if isinstance(obs, torch.Tensor):
                    batch_obs.append(obs.cpu().numpy())
                else:
                    batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, rew, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        # Convert to tensors and move to device
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float, device=self.device)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float, device=self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float, device=self.device)
        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def learn(self, total_timesteps):
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode")
        print(f"{self.timesteps_per_batch} timesteps per batch for total of {total_timesteps}")
        
        t_so_far = 0  
        i_so_far = 0  
        
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            t_so_far += np.sum(batch_lens)
            i_so_far += 1

            # Calculate advantage - ensure all tensors are on the same device
            V, _ = self.evaluate(batch_obs, batch_acts)
            batch_rtgs = batch_rtgs.to(self.device)  # Make sure batch_rtgs is on correct device
            A_k = batch_rtgs - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # Update networks
            for _ in range(self.n_updates_per_iteration):
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
            
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor_net.state_dict(), './ppo_actor.pth')
                torch.save(self.critic_net.state_dict(), './ppo_critic.pth')
                print(f"Iteration {i_so_far}: Actor loss: {actor_loss:.4f}, Critic loss: {critic_loss:.4f}")

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Return tensor on the correct device
        return torch.tensor(batch_rtgs, dtype=torch.float, device=self.device)
    
    def evaluate(self, batch_obs, batch_acts):
        # Ensure inputs are on correct device
        batch_obs = batch_obs.to(self.device)
        batch_acts = batch_acts.to(self.device)
        
        V = self.critic_net(batch_obs).squeeze()
        mean = self.actor_net(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs
