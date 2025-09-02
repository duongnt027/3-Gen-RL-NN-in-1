import torch
import random
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from sklearn.neighbors import KernelDensity
from torch.distributions import MultivariateNormal, Normal

from utils import tensor_pop

class AnorEnv:
    # Khởi tạo môi trường học từ dataset
    # Đầu vào là dataset (Dataset), mô hính tạo sinh (nn.Module), mô hình phân biệt (nn.Module), shuffle (True-False)
    def __init__(self, dataset, generator, detector, shuffle=True):
        # Thiết lập device theo phần cứng của máy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Lấy dữ liệu X, y từ dataset, lưu trên device
        self.X = dataset.X.to(self.device)
        self.y = dataset.y.to(self.device)
        
        # Trộn dữ liệu X, y
        if shuffle:
            indices = torch.randperm(len(self.X))
            self.X = self.X[indices]
            self.y = self.y[indices]

        # Định nghĩa mô hình tạo sinh, mô hình phát hiện
        self.generator = generator
        self.detector = detector
        
        # Định nghĩa không gian quan sát và không gian hành động bằng các tensor zero dựa trên các chiều (Thực ra có thể định nghĩa chiều của các không gian luôn)
        # Define observation and action space dimensions
        self.observation_space = torch.zeros(generator.in_dim)  # shape (generator.in_dim, )
        self.action_space = torch.zeros(2)  # [mu, sigma] để thay đổi biến, shape (2, )
        
        # Định nghĩa states, buffer_x giống self.X nhưng không chung địa chỉ
        self.indices = torch.nonzero(self.X == 1, as_tuple=True)[0]
        self.states = self.X.clone()
        self.buffer_X = self.X.clone()

        # Định nghĩa 
        # current state là trạng thái quan sát lúc này (dữ liệu X từ dataset), 
        # current step là số bước hiện tại của quá trình tương tác với môi trường, 
        # max_step là số lượng ban đầu của X
        self.current_state = None
        self.current_step = 0
        self.max_steps = len(self.indices)
        
    # Reset lại các thông số của môi trường về trạng thái ban đầu
    def reset(self):
        self.current_step = 0
        self.indices = torch.nonzero(self.X == 1, as_tuple=True)[0]
        print(self.indices)
        self.states = self.X.clone()
        self.buffer_X = self.X.clone()
        
        # Khởi tạo lại current_state và indices từ đầu, current_state là 1 random có y bằng 1 từ X
        idx = random.randint(0, self.indices.shape[0] - 1)
        current_idx, self.indices = tensor_pop(self.indices, idx)

        self.current_state = self.X[current_idx].to(self.device)

        # Thông tin của môi trường
        self.info = {
            "current_step": 0,
            "current_index": current_idx,
            "added_to_buffer": False,
            "terminated": False,
            "truncated": False
        }
            
        return self.current_state, self.info

    def step(self, action):
        # Tăng bước tương tác với môi trường
        self.current_step += 1
        
        # Định nghĩa tín hiệu dừng
        terminated = False # dừng theo môi trường
        truncated = False # dừng theo điều kiện

        # Đảm bảo action luôn là 1 tensor lưu theo device
        action = torch.tensor(action)
        action = action.to(self.device)

        # action chứa mean, sigma của 1 phân phối chuẩn
        mu, sigma = action[0], torch.abs(action[1]) + 1e-6  # đảm bảo sigma > 0
        
        # Chuẩn bị current_state trước khi encode
        current_state_batch = self.current_state.unsqueeze(0)
        y_batch = torch.tensor([[1.0]], dtype=torch.float32).to(self.device)
        
        # Encode current_state để lấy latent vector
        with torch.no_grad():
            mean_z, logvar_z = self.generator.encode(current_state_batch, y_batch)
            z = self.generator.reparameterize(mean_z, logvar_z)
        
        # Tạo delta từ mean, sigma của action và thay đổi latent vector cũ thành vector mới
        dist = Normal(mu, sigma)
        delta = dist.sample(z.shape).to(self.device)
        z_new = z + delta
        
        # Tạo dữ liệu mới từ latent vector
        with torch.no_grad():
            x_new = self.generator.decode(z_new, y_batch)
        
        # Tính reward của latent vector
        reward = self._calculate_reward(x_new.squeeze(0))
        
        # Hủy theo môi trường nếu không còn dữ liệu khả thi
        if self.X.shape[0] == 0:
            terminated = True
        
        # Hủy theo điều kiện nếu current_step vượt quá số lượng dữ liệu có thể sử dụng
        if self.current_step >= self.max_steps:
            truncated = True
            
        # Thêm dữ liệu mới vào buffer X
        self.buffer_X = torch.cat([self.buffer_X, x_new], dim=0)
        self.info['added_to_buffer'] = True
        self.info['current_step'] = self.current_step
        self.info['terminated'] = terminated
        self.info['truncated'] = truncated
        
        # Nếu chưa gặp tín hiệu hủy, lấy state mới
        if not terminated and self.states.shape[0] > 0:
            idx = random.randint(0, self.indices.shape[0] - 1)
            current_idx, self.indices = tensor_pop(self.indices, idx)
            self.current_state = self.X[current_idx].to(self.device)
            self.info['current_index'] = current_idx
        
        return x_new.squeeze(0), reward, terminated, truncated, self.info

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
        
        # Định nghĩa môi trường, chiều của điểm quan sát, chiều của hành động
        self.env = env
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # Khai báo device cuda nếu có
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Khai báo mạng chính sách (actor), mạng đánh giá (critic)
        self.actor_net = actor_net
        self.actor_net = self.actor_net.to(self.device)
        self.critic_net = critic_net
        self.critic_net = self.critic_net.to(self.device)

        # Khai báo hàm tối ưu
        self.actor_optim = Adam(self.actor_net.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic_net.parameters(), lr=self.lr)

        # Định nghĩa ma trận hiệp phương sai
        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5, device=self.device)
        self.cov_mat = torch.diag(self.cov_var)

    # Hàm lấy giá trị từ điểm quan sát
    def get_action(self, observation):
        # Đảm bảo điểm quan sát là tensor lưu trên device
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        else:
            observation = observation.to(self.device)

        # Lấy đầu ra của mạng actor là mean, sigma
        mean_sigma = self.actor_net(observation)
        # Tạo phân phối chuẩn đa biến
        dist = MultivariateNormal(mean_sigma, self.cov_mat)
        # Tạo 1 hành động ngẫu nhiên trong phân phối chuẩn
        action = dist.sample()
        # Tính log_prob của phân phối đó
        log_prob = dist.log_prob(action)

        return action.cpu().detach().numpy(), log_prob.detach()

    # Khởi tạo các hyperparameter        
    def _init_hyperparameters(self, hyperparameters):
        # Default hyperparameters
        # Bước chạy 1 batch
        self.timesteps_per_batch = 600
        # Bước chạy tối đa của 1 ep
        self.max_timesteps_per_episode = 200
        # Số lần update lại mạng
        self.n_updates_per_iteration = 10      
        self.lr = 3e-4                         
        self.gamma = 0.99                      
        self.clip = 0.2                        
        # self.render = False                    
        # self.render_every_i = 10              
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
