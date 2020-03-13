import copy

import numpy as np
import torch
import torch.nn.functional as F

from tools import saver
from datasets.executor import KarelExecutor
from rl_agent import utils
from rl_agent.logger import logger_task
from rl_agent.utils import StepExample, ReplayBuffer, State, Task
from rl_agent.distributions import entropy


def choice(total, sample):
    if total == 0:
        return []
    else:
        return np.random.choice(total, sample, replace=False)


def rollout(env, agent, args):
    with torch.no_grad():
        state = env.reset()
        agent.set_task(state.task)

        experience = []
        success = False

        for _ in range(args.max_rollout_length):
            action = agent.select_action(state.code)
            new_state, reward, done, _ = env.step(action)
            #(*) check if action given state corresponds to pi, if yes call select_action on new_state
            new_action = agent.select_action(new_state.code)

            assert len(new_state.code) <= args.max_token_per_code

            experience.append(StepExample(copy.deepcopy(state), action, reward, done, copy.deepcopy(new_state)))

            if done:
                success = True
                break

            state = new_state

    return success, experience


def update_step_example(trans: StepExample, new_goal, executor: KarelExecutor) -> StepExample:
    goal_code = new_goal
    new_code = trans.next_state.code

    inputs = trans.state.task.inputs.squeeze(0).data.numpy()

    outputs = np.zeros_like(inputs)

    assert inputs.shape[0] == 5

    done = True

    for ix in range(inputs.shape[0]):
        grid_inp, = np.where(inputs[ix].ravel())

        grid_out, trace = executor.execute(goal_code, None, grid_inp, record_trace=True)
        outputs[ix].ravel()[grid_out] = 1

        if done:
            _grid_out, trace = executor.execute(new_code, None, grid_inp, record_trace=True)

            if grid_out == _grid_out:
                done = False

    new_outputs = torch.from_numpy(outputs).unsqueeze(0)
    task = trans.state.task

    assert np.allclose(trans.reward, -1.)

    return StepExample(
        State(Task(task.inputs.clone(), new_outputs.clone()), trans.state.code),
        trans.action,
        trans.reward,
        done,
        State(Task(task.inputs.clone(), new_outputs.clone()), trans.next_state.code),
    )


class PolicyTrainer(object):
    def __init__(self, agent, env, args):
        self.args = args
        self.env = env
        self.vocab = env.vocab

        self.actor = agent(self.vocab, args)
        self.critic = agent(self.vocab, args)

        self.criterion = F.mse_loss
        self.optimizer = torch.optim.Adam(self.actor.model.grad_parameters(), lr=args.lr)

        if not self.args.train_from_scratch:
            self.step = saver.load_checkpoint(self.actor.model, self.optimizer, self.args.model_dir, step=430100)
        else:
            self.step = 0

        self.critic.update_with(self.actor)
        self.karel_executor = KarelExecutor()

    def train_actor_critic(self, batch: "list[StepExample]"):
        size = self.args.batch_size
        alpha = self.args.rl_alpha
        discount = self.args.rl_discount

        code_pad = self.vocab.stoi(self.vocab.itos(-1))

        # Loading tensors for training
        code_state = torch.full((size, self.args.max_token_per_code), code_pad, dtype=torch.int64)

        task_state_I = torch.cat([ex.state.task.inputs for ex in batch])
        task_state_O = torch.cat([ex.state.task.outputs for ex in batch])

        reward = torch.Tensor([b.reward for b in batch])
        actions = [] #[b.action for b in batch]

        code_next_state = torch.full((size, self.args.max_token_per_code), code_pad, dtype=torch.int64)
        task_next_state_I = torch.cat([ex.next_state.task.inputs for ex in batch])
        task_next_state_O = torch.cat([ex.next_state.task.outputs for ex in batch])

        for ix, (s, a, r, d, ns) in enumerate(batch):
            t_code = utils.prepare_code(s.code, self.env.vocab, tensor=True)
            code_state[ix, :len(s.code)] = t_code[0]

            t_code = utils.prepare_code(ns.code, self.env.vocab, tensor=True)
            code_next_state[ix, :len(ns.code)] = t_code[0]

            _, action = a
            actions.append(action)


        # Evaluating model
        # Computes armax Q_s_a_prime
        next_state_value, _ = self.critic.best_action_value(code_next_state,
                                                            utils.Task(task_next_state_I, task_next_state_O))

        # Fix this: Compute `next_state_value` only for actions that are non terminal states
        for i in range(size):
            if np.isclose(batch[i].done, 0):
                next_state_value[i] = 0.

        state_value, parameter_value, _ = self.actor.action_value_from_action(code_state,
                                                                           utils.Task(task_state_I, task_state_O),
                                                                           actions)

        target_state_value = state_value + alpha * (reward + discount * next_state_value - state_value)
        target_parameter_value = parameter_value + alpha * (reward + discount * next_state_value - parameter_value)

        loss_action = self.criterion(state_value, target_state_value)
        loss_parameter = self.criterion(parameter_value, target_parameter_value)
        loss = loss_action + loss_parameter

        assert int(torch.isnan(loss)) == 0, "{} ({})".format(loss,type(loss))

        # Updating model

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss)

    def train(self):
        replay_buffer = ReplayBuffer(self.args.replay_buffer_size)

        smooth_loss = 0.

        for step in range(self.step + 1, self.args.num_iterations + 1):
            logger_task.info("Step: {}".format(step))

            for ix in range(self.args.num_rollouts):
                logger_task.info("Start rollout: {}".format(ix))
                success, experience = rollout(self.env, self.actor, self.args)
                logger_task.info("Success: {} Experience length: {}".format(int(success),len(experience)))
                [replay_buffer.add(e) for e in experience]

                if self.args.her:
                    for e_ix, e in enumerate(experience):
                        future = len(experience) - e_ix - 1
                        samp = min(future, self.args.her_new_goals)

                        for c_ix in choice(future, samp):
                            new_exp = update_step_example(e, experience[-c_ix - 1].state.code, self.karel_executor)
                            replay_buffer.add(new_exp)

                logger_task.info(
                    ' '.join(["Replay buffer {}/{}".format(replay_buffer.done_examples,replay_buffer.size),
                              "({:.4f}%)".format(replay_buffer.done_examples/replay_buffer.size * 100)]))

            self.actor.set_train(True)

            for ix in range(self.args.num_training_steps):
                logger_task.info("Training step: {}".format(ix))
                batch = replay_buffer.sample(self.args.batch_size)
                loss = self.train_actor_critic(batch)
                smooth_loss = .99 * smooth_loss + .01 * loss
                logger_task.info("Loss: {}".format(smooth_loss))

            self.actor.set_train(False)

            if step % self.args.update_actor_it == 0:
                logger_task.info("Update critic with actor")
                self.critic.update_with(self.actor)

            if step % self.args.save_actor_it == 0:
                saver.save_checkpoint(self.actor.model, self.optimizer, step, self.args.model_dir)
                saver.save_args(self.args)


class PPOTrainer(PolicyTrainer):
    def __init__(self, agent, env, args):
        super(PPOTrainer, self).__init__(agent, env, args)
        
    def train_actor_critic(self, batch: "list[StepExample]"):
        size = self.args.batch_size
        alpha = self.args.rl_alpha
        discount = self.args.rl_discount
        K_epoch = 4
        lmbda = 0.99
        eps_clip = 0.2
        c1 = 0.5
        c2 = 0.5

        code_pad = self.vocab.stoi(self.vocab.itos(-1))

        # Loading tensors for training
        code_state = torch.full((size, self.args.max_token_per_code), code_pad, dtype=torch.int64)

        task_state_I = torch.cat([ex.state.task.inputs for ex in batch])
        task_state_O = torch.cat([ex.state.task.outputs for ex in batch])

        reward = torch.Tensor([b.reward for b in batch])
        actions = [] #[b.action for b in batch]
        action_probs = []

        code_next_state = torch.full((size, self.args.max_token_per_code), code_pad, dtype=torch.int64)
        task_next_state_I = torch.cat([ex.next_state.task.inputs for ex in batch])
        task_next_state_O = torch.cat([ex.next_state.task.outputs for ex in batch])

        done_mask = torch.ones((size))

        for ix, (s, a, r, d, ns) in enumerate(batch):
            t_code = utils.prepare_code(s.code, self.env.vocab, tensor=True)
            code_state[ix, :len(s.code)] = t_code[0]

            t_code = utils.prepare_code(ns.code, self.env.vocab, tensor=True)
            code_next_state[ix, :len(ns.code)] = t_code[0]

            if d:
                done_mask[ix]= 0

            action_prob, action = a
            action_probs.append(action_prob)
            actions.append(action)

            # (*) if you need PPO update for the continious actor please use actions here


        # Evaluating model
        next_state_value, _ = self.critic.best_action_value(code_next_state,
                                                            utils.Task(task_next_state_I, task_next_state_O))

        # Fix this: Compute `next_state_value` only for actions that are non terminal states
        for i in range(size):
            if np.isclose(batch[i].done, 0):
                next_state_value[i] = 0.

        state_value, parameter_value, parameter_value_list = self.actor.action_value_from_action(code_state,
                                                                           utils.Task(task_state_I, task_state_O),
                                                                           actions)

        ### PPO Update ###
        ##################

        # Compute original probability of action
        prob_a = F.softmax(torch.cat(action_probs),dim=1)

        # (*) Likewise you must do something similar to the parameters stuff -https://arxiv.org/pdf/1903.01344.pdf


        for i in range(K_epoch):

            # Compute pi and pi_next
            pi = self.compute_pi(code_state, task_state_I, task_state_O)
            pi_next = self.compute_pi(code_next_state, task_next_state_I, task_next_state_O)

            # Compute value function: v_pi = sum_a pi(a|s) q_pi(s,a)
            Q_s_a_prime = self.actor.action_value(code_next_state,utils.Task(task_next_state_I, task_next_state_O))
            Q_s_a = self.actor.action_value(code_state,utils.Task(task_state_I, task_state_O))

            v_s_prime = torch.sum(Q_s_a_prime*pi_next,dim=1)
            v_s = torch.sum(Q_s_a*pi,dim=1)

            # State value Q(s,a)
            td_target = reward + discount * v_s_prime * done_mask
            delta = td_target - v_s
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = discount * lmbda * advantage + delta_t
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).view(-1)

            # fix probabilities to be argmax probs!
            ratio = torch.exp(torch.log(pi)-torch.log(prob_a))
            mask = torch.zeros_like(ratio)
            for i in range(len(actions)):
                mask[i][actions[i].id]=1
            ratio = torch.sum(mask*ratio,dim=1)
            
            # Compute clip loss
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            Ld_clip =-torch.min(surr1, surr2) 
            loss_action = Ld_clip + c1*F.smooth_l1_loss(v_s , td_target.detach()) + c2* entropy(pi)

            self.optimizer.zero_grad()
            loss_action.mean().backward()
            self.optimizer.step()
            

        # The update rules of the two actor networks could follow policy gradient methods 
        # or other optimization methods for stochastic poli-cies such as PPO. 
        target_parameter_value = parameter_value + alpha * (reward + discount * next_state_value - parameter_value)
        loss_parameter = self.criterion(parameter_value, target_parameter_value)

        # Updating model

        self.optimizer.zero_grad()
        loss_parameter.backward()
        self.optimizer.step()

        loss = loss_action.mean() + loss_parameter

        assert int(torch.isnan(loss)) == 0, "{} ({})".format(loss,type(loss))

        return float(loss)

    def compute_pi(self, code_state, task_state_I, task_state_O):
        position_enc, code_enc = self.actor.model.encode_code(code_state)
        task_enc = self.actor.get_task_enc(utils.Task(task_state_I, task_state_O))
        pi = F.softmax(self.actor.model.operation_type(task_enc,code_enc),dim=1)

        return pi
