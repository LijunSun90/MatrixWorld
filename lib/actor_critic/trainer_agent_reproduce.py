import torch
import torch.nn as nn

from .buffer_basic import Buffer


class TrainerAgent:

    def __init__(self, model_actor_critic, buffer_size, n_agents, dim_observation_actor, dim_observation_critic,
                 configuration):

        # Parameter.

        lr_value = configuration.lr_value
        lr_policy = configuration.lr_policy

        self.train_iterations_policy = configuration.train_iterations_policy
        self.train_iterations_value = configuration.train_iterations_value

        self.entropy_coefficient = configuration.entropy_coefficient
        self.use_gradient_norm = configuration.use_gradient_norm
        self.max_gradient_norm = configuration.max_gradient_norm

        # Model.

        self.model_actor_critic = model_actor_critic

        self.model_value = model_actor_critic.model_value

        self.model_policy = model_actor_critic.model_policy

        # Optimizer.

        self.optimizer_value = torch.optim.Adam(self.model_value.parameters(), lr=lr_value)

        self.optimizer_policy = torch.optim.Adam(self.model_policy.parameters(), lr=lr_policy)

        # Buffer.

        self.buffer = Buffer(buffer_size=buffer_size,
                             n_agents=n_agents,
                             dim_observation_actor=dim_observation_actor,
                             dim_observation_critic=dim_observation_critic,
                             configuration=configuration)

        # Share parameters.

        self.data = None

        pass

    def update_model(self):

        self.data = self.buffer.get()

        loss_policy = 0
        loss_value = 0

        for _ in range(self.train_iterations_policy):

            # Policy model update.

            self.optimizer_policy.zero_grad()

            loss_policy, distribution_entropy = self.compute_loss_policy()

            (loss_policy - distribution_entropy * self.entropy_coefficient).backward()

            if self.use_gradient_norm:

                nn.utils.clip_grad_norm_(self.model_policy.parameters(), self.max_gradient_norm)

            self.optimizer_policy.step()

        for _ in range(self.train_iterations_value):

            # Value model update.

            self.optimizer_value.zero_grad()

            loss_value = self.compute_loss_value()

            loss_value.backward()

            if self.use_gradient_norm:

                nn.utils.clip_grad_norm_(self.model_value.parameters(), self.max_gradient_norm)

            self.optimizer_value.step()

        return loss_policy.item(), loss_value.item()

    def compute_loss_policy(self):

        # Get data.

        observations_actor, actions, advantages, actions_log_probability_old = \
            self.data['obs_actor'], self.data['act'], self.data['adv'], self.data['logp']

        # ModelPolicy loss.

        actions_log_probability, distribution_entropy = self.model_policy.evaluate_action(observations_actor, actions)

        loss_policy = - (actions_log_probability * advantages).mean()

        return loss_policy, distribution_entropy

    def compute_loss_value(self):

        # Get data.

        observations_critic, returns, values_old = \
            self.data['obs_critic'], self.data['ret'], self.data['val']

        # Value loss.

        values = self.model_value(observations_critic)

        # Basic mse loss.

        loss_value = ((values - returns) ** 2).mean()

        return loss_value

