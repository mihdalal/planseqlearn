# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import torch.nn.functional as F

from planseqlearn import utils
from planseqlearn.models.actor import Actor
from planseqlearn.models.critic import Critic
from planseqlearn.models.encoder import DrQV2Encoder, PoolEncoder
from planseqlearn.models.random_shifts_aug import RandomShiftsAug


class DrQV2Agent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        feature_dim,
        hidden_dim,
        critic_target_tau,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
        stddev_clip,
        use_tb,
        use_pool_encoder,
        pool_encoder_latent_dim,
    ):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        # models
        if use_pool_encoder:
            self.encoder = PoolEncoder(obs_shape, repr_dim=pool_encoder_latent_dim).to(
                device
            )
        else:
            self.encoder = DrQV2Encoder(obs_shape).to(device)

        self.actor = Actor(
            self.encoder.repr_dim, action_shape, feature_dim, hidden_dim
        ).to(device)

        self.critic = Critic(
            self.encoder.repr_dim, action_shape, feature_dim, hidden_dim
        ).to(device)
        self.critic_target = Critic(
            self.encoder.repr_dim, action_shape, feature_dim, hidden_dim
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = obs["pixels"]
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(
        self, encoded_obs, action, reward, discount, encoded_next_obs, step
    ):
        metrics = {}

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(encoded_next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_q1, target_q2 = self.critic_target(encoded_next_obs, next_action)
            target_v = torch.min(target_q1, target_q2)
            target_q = reward + (discount * target_v)

        q1, q2 = self.critic(encoded_obs, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        if self.use_tb:
            metrics["critic_target_q"] = target_q.mean().item()
            metrics["critic_q1"] = q1.mean().item()
            metrics["critic_q2"] = q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        return metrics

    def update_actor(self, encoded_obs, step):
        metrics = {}

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(encoded_obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        q1, q2 = self.critic(encoded_obs, action)
        q = torch.min(q1, q2)

        actor_loss = -q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = {}

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = batch
        rgb_obs = obs["pixels"]
        next_rgb_obs = next_obs["pixels"]
        rgb_obs, action, reward, discount, next_rgb_obs = utils.to_torch(
            (rgb_obs, action, reward, discount, next_rgb_obs), self.device
        )

        # augment
        rgb_obs = self.aug(rgb_obs.float())
        next_rgb_obs = self.aug(next_rgb_obs.float())
        # encode
        encoded_obs = self.encoder(rgb_obs)
        with torch.no_grad():
            encoded_next_obs = self.encoder(next_rgb_obs)

        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()

        # update critic and decoder
        metrics.update(
            self.update_critic(
                encoded_obs, action, reward, discount, encoded_next_obs, step
            )
        )

        # update actor
        metrics.update(self.update_actor(encoded_obs.detach(), step))

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics

    def get_frames_to_record(self, obs):
        rgb_obs = obs["pixels"]
        frame = rgb_obs[-3:].transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
        return {"rgb": frame}

    def load_pretrained_weights(self, pretrain_path, just_encoder_decoders):
        if just_encoder_decoders:
            print("Loading pretrained encoder and decoders")
        else:
            print("Loading entire agent")

        payload = torch.load(pretrain_path, map_location="cpu")
        pretrained_agent = payload["agent"]

        self.encoder.load_state_dict(pretrained_agent.encoder.state_dict())

        if not just_encoder_decoders:
            self.actor.load_state_dict(pretrained_agent.actor.state_dict())
            self.critic.load_state_dict(pretrained_agent.critic.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
