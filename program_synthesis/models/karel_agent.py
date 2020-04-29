import collections
import copy

import numpy as np
import random
import operator

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import pickle as pkl

import arguments
#from program_synthesis.common.tools import saver
from models import get_model
from datasets import data, dataset, set_vocab, executor
from datasets.karel import refine_env, mutation
from tools import saver
from tools.reporter import TensorBoardRLWrite
from models.radam import RAdam
from . import prepare_spec, karel_model
from . import beam_search
from .modules import karel, attention


# TODO: Make sure that it is possible to choose from vocab size action space
# Alternatively have 2 more: remove token, add token where add token is treated in a second selection

DISCOUNT = 0.99
lmbda = 0.95
EPISLON = 0.1
ALPHA = 0.7

StepExample = collections.namedtuple('StepExample', ['state', 'action', 'reward', 'new_state', 'value', 'lengths', 'acc'])
BeamSearchResult = collections.namedtuple('BeamSearchResult', ['sequence', 'total_log_prob', 'log_probs', 'log_probs_torch'])


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)   
    torch.cuda.manual_seed_all(seed)    
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class Categorical(nn.Module):
    def __init__(self):
        super(Categorical, self).__init__()

    def forward(self, x):
        return FixedCategorical(logits=x)



def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes):
        self.obs = [[] for i in range(num_steps)] # torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = [[] for i in range(num_steps)]#torch.zeros( num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)

        self.actions = torch.zeros(num_steps, num_processes)

        self.masks = torch.zeros(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.zeros(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        #self.obs = self.obs.to(device)
        #self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks):
        self.obs[self.step] = obs
        self.recurrent_hidden_states[self.step] = recurrent_hidden_states
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) #% self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[
                    step + 1] * self.masks[step +
                                            1] - self.value_preds[step]
                gae = delta + gamma * gae_lambda * self.masks[step +
                                                                1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma * self.masks[step + 1] + self.rewards[step]

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            #recurrent_hidden_states_batch = torch.stack(
            #    recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            #obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ


class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            
            data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch



class KarelEditEnv(object):

    def __init__(self, args):

        self.vocab = data.PlaceholderVocab(
            data.load_vocab(dataset.relpath('../data/karel/word.vocab')), 0)

        self.refiner = karel_model.KarelLGRLRefineBatchProcessor(args, self.vocab, False)
        self.dataset_loader, self.devset_loader = dataset.get_karel_dataset_nomodel(args, self.refiner)

        self.data_iter = self.dataset_loader.__iter__()
        self._cur_env = None
        self._cuda = args.cuda
        self.args = args

    def reset(self):
        example = self.data_iter.next()
 
        return example

    
    def rollout(self, state, agent, max_rollout_length, rollouts):

        init_state, masked_memory = agent.model.prepare_state(state)

        tt = torch.cuda if self.args.cuda else torch
        beam_size = 1
        batch_size = self.args.batch_size
        prev_tokens = Variable(tt.LongTensor(batch_size).fill_(0))
        prev_probs = Variable(tt.FloatTensor(batch_size, 1).fill_(0))
        prev_hidden = init_state
        finished = [[] for _ in range(batch_size)]
        result = [[BeamSearchResult(sequence=[], log_probs=[], total_log_prob=0, log_probs_torch=[])
                for _ in range(beam_size)] for _ in range(batch_size)]
        batch_finished = [False for _ in range(batch_size)]

        prev_masked_memory = masked_memory.expand_by_beam(beam_size)

        attn_list = []
        bp = 1
        can_stop = True
        # Restart work from here


        rollouts.insert(finished, (prev_tokens, prev_hidden, masked_memory, attn_list), torch.zeros(batch_size),
                            torch.zeros(batch_size,1),  torch.zeros(batch_size,1),  torch.zeros(batch_size,1), torch.ones(batch_size,1), torch.ones(batch_size,1))
        
        for step in range(max_rollout_length):
            print(step)
            #prev_masked_memory if step > 0 else memory
            # Make sure there will be no mismatch again when reproducing this stuff
            
            with torch.no_grad():
                #finished, recurrent_hidden_states, value, action, action_log_prob, bp
                #finished, value, action_log_probs, rnn_hxs, bp , dist_entropy, batch_finished, prev_probs, result = agent.act(
                #    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                #    batch_finished, bp, prev_probs, result)

                action_log_probs, dist_entropy,  value, prev_tokens, prev_hidden, prev_masked_memory, \
                masked_memory, attn_list, result, prev_probs, can_stop, finished, batch_finished = agent.act2(prev_tokens, prev_hidden, prev_masked_memory, step, masked_memory, attn_list, result, prev_probs, can_stop, finished, batch_finished)

            if can_stop: #bp==1:
                break

            masks = torch.FloatTensor(
                [[1.0] if done_ else [0.0] for done_ in batch_finished])
            bad_masks = masks

            rnn_hxs = prev_tokens, prev_hidden, prev_masked_memory, attn_list


            reward = torch.zeros(batch_size,1)

            rollouts.insert(finished, rnn_hxs, prev_tokens,
                            action_log_probs, value, reward, masks, bad_masks)


        
        sequences = beam_search.beam_search(
            self.args.batch_size,
            init_state,
            memory,
            agent.model.model.model.decoder.decode_token,
            1,
            cuda=self.args.cuda,
            max_decoder_length=max_rollout_length,
            return_beam_search_result=True,
            volatile=False,
            differentiable=True
        )
        breakpoint

        orig_examples = state[-1]

        output_code = self.model.model.decoder.postprocess_output([[x.sequence for x in y] for y in finished], memory)
        rewards = []
        for logit_beam, code_beam, example in zip(sequences, output_code, orig_examples):
            for i, (logits, code) in enumerate(zip(logit_beam, code_beam)):
                code = list(map(self.vocab.itos, code))
                run_cases = lambda tests: executor.evaluate_code(code, example.schema.args, tests,
                                                                 self.model.executor.execute)
                input_tests = run_cases(example.input_tests)
                reward = input_tests['correct'] / input_tests['total']
                if self.args.use_held_out_test_for_rl:
                    held_out_test = run_cases(example.tests)
                    reward += held_out_test['correct']  # worth as much as all the other ones combined

                rewards.append(reward)
        
        indicies= torch.max(rollouts.masks)[1]

        rewards = torch.tensor(rewards)

        if all_logits.is_cuda:
            rewards = rewards.cuda()
        
        return rollouts

    def step(self, action):

        (labels, probs, code_seq, values, reward) = action
        acc = np.array([0])

        # Similarly the next states should also be modified in this fasion
        next_state = labels  #self.edit_space_to_code_space_special_order(actions, code_seq, lengths[0], lengths[1])
        #obs, reward, done, info = self._cur_env.step(action)
        return next_state, reward, True, acc


class KarelEditPolicy(nn.Module):
    def __init__(self, args):
        super(KarelEditPolicy, self).__init__()
        self.args = args

        set_vocab(args)
        self.model = get_model(args)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def encode(self, input_grid, output_grid):
        return self.model.model.encoder(input_grid,output_grid)

    def encode_code(self, code_state, input_grid, output_grid, traces, trace_events):
        return self.model.model.code_encoder(code_state, input_grid, output_grid, traces, trace_events)

    def encode_trace(self, ref_code_memory, ref_trace_grids, ref_trace_events, cag_interleave ):
        return self.model.model.trace_encoder(ref_code_memory, ref_trace_grids,ref_trace_events, cag_interleave)

    def decode(self, io_embed, ref_code_memory, ref_trace_memory, code_seqs, dec_data):
        return self.model.model.decoder.rl_forward(io_embed, ref_code_memory, ref_trace_memory, code_seqs, dec_data)

    def decode_and_reward(self, input_grids, io_embed, orig_examples, ref_code, ref_code_memory,
                                       ref_trace_memory):
        init_state = self.model.model.decoder.init_state(
            ref_code_memory, ref_trace_memory,
            io_embed.shape[0], io_embed.shape[1])
        memory = self.model.model.decoder.prepare_memory(io_embed, ref_code_memory,
                                                   ref_trace_memory, ref_code)
        sequences = beam_search.beam_search(
            len(input_grids),
            init_state,
            memory,
            self.model.model.decode_token,
            self.args.max_beam_trees,
            cuda=self.args.cuda,
            max_decoder_length=self.args.max_decoder_length,
            return_beam_search_result=True,
            volatile=False,
            differentiable=True
        )

        output_code = self.model.model.decoder.postprocess_output([[x.sequence for x in y] for y in sequences], memory)
        all_logits = []
        rewards = []
        for logit_beam, code_beam, example in zip(sequences, output_code, orig_examples):
            for i, (logits, code) in enumerate(zip(logit_beam, code_beam)):
                code = list(map(self.model.vocab.itos, code))
                all_logits.append(torch.sum(torch.cat([x.view(1) for x in logits.log_probs_torch])))
                run_cases = lambda tests: executor.evaluate_code(code, example.schema.args, tests,
                                                                 self.model.executor.execute)
                input_tests = run_cases(example.input_tests)
                reward = input_tests['correct'] / input_tests['total']
                if self.args.use_held_out_test_for_rl:
                    held_out_test = run_cases(example.tests)
                    reward += held_out_test['correct']  # worth as much as all the other ones combined
                if reward == 2:
                    reward = 1
                else:
                    reward = 0
                rewards.append(reward)
        all_logits = torch.cat([x.view(1) for x in all_logits])
        rewards = torch.tensor(rewards)

        if all_logits.is_cuda:
            rewards = rewards.cuda()
        return rewards, all_logits, output_code

    def action_value(self, states):

        input_grids, output_grids, code_seqs, \
            dec_data, ref_code, ref_trace_grids,\
                ref_trace_events, cag_interleave, orig_examples = states

        if self.args.cuda:
            input_grids = input_grids.cuda(async=True)
            output_grids = output_grids.cuda(async=True)
            code_seqs = karel_model.maybe_cuda(code_seqs, async=True)
            dec_data = karel_model.maybe_cuda(dec_data, async=True)
            ref_code = karel_model.maybe_cuda(ref_code, async=True)
            ref_trace_grids = karel_model.maybe_cuda(ref_trace_grids, async=True)
            ref_trace_events = karel_model.maybe_cuda(ref_trace_events, async=True)

        io_embed = self.encode(input_grids, output_grids)
        # code_seq = ground thruth code, ref_code.ps.data = edited

        ref_code_memory = self.encode_code(ref_code, input_grids, output_grids, ref_trace_grids, ref_trace_events)

        ref_trace_memory = self.encode_trace( ref_code_memory, ref_trace_grids, ref_trace_events, cag_interleave)

        # calculate nll loss
        #logits, labels = self.model.model.decoder(io_embed, ref_code_memory, ref_trace_memory, code_seqs, dec_data)

        #print('crossentro')
        #print(self.criterion(
        #    logits.view(-1, logits.shape[-1]), labels.contiguous().view(-1)))

        # Calculate logits and reward using beam search

        reward, logits, labels = self.decode_and_reward(input_grids, io_embed, orig_examples, ref_code, ref_code_memory,
                                       ref_trace_memory)

        #logits, labels, dec_output, lengths = self.decode(io_embed, ref_code_memory,
        #                                   ref_trace_memory, code_seqs,
        #                                   dec_data)

        return logits, labels, torch.cat((io_embed.mean(dim=1),ref_code_memory.state[0].view(code_seqs.size(0),-1)),1), reward

    def prepare_state(self, states):
        input_grids, output_grids, code_seqs, \
            dec_data, ref_code, ref_trace_grids,\
                ref_trace_events, cag_interleave, orig_examples = states

        if self.args.cuda:
            input_grids = input_grids.cuda(async=True)
            output_grids = output_grids.cuda(async=True)
            code_seqs = karel_model.maybe_cuda(code_seqs, async=True)
            dec_data = karel_model.maybe_cuda(dec_data, async=True)
            ref_code = karel_model.maybe_cuda(ref_code, async=True)
            ref_trace_grids = karel_model.maybe_cuda(ref_trace_grids, async=True)
            ref_trace_events = karel_model.maybe_cuda(ref_trace_events, async=True)

        io_embed = self.encode(input_grids, output_grids)
        # code_seq = ground thruth code, ref_code.ps.data = edited

        ref_code_memory = self.encode_code(ref_code, input_grids, output_grids, ref_trace_grids, ref_trace_events)

        ref_trace_memory = self.encode_trace( ref_code_memory, ref_trace_grids, ref_trace_events, cag_interleave)

        init_state = self.model.model.decoder.init_state(
            ref_code_memory, ref_trace_memory,
            io_embed.shape[0], io_embed.shape[1])
        memory = self.model.model.decoder.prepare_memory(io_embed, ref_code_memory,
                                                   ref_trace_memory, ref_code)
        
        return init_state, memory

class KarelAgent(object):

    def __init__(self, env, args):
        self.vocab = env.vocab
        self.model = KarelEditPolicy(args)
        self.criterion = nn.MSELoss()
        self.args =args
        if args.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.model.model.parameters(), lr=args.lr)
        elif args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.model.model.parameters(), lr=args.lr)
        else:
            self.optimizer = RAdam(self.model.model.model.parameters(), lr=args.lr, eps=1e-10, weight_decay=0.001)
        self.critic = nn.Linear(256, 1, bias=False).cuda() if args.cuda else nn.Linear(256, 1, bias=False)
        self.dist = Categorical()

    def act(self, inputs, rnn_hxs, batch_finished, bp, prev_probs, result):

        finished, value, action_log_probs, actor_features, rnn_hxs, bp, batch_finished, prev_probs, result = self.base(inputs, rnn_hxs, batch_finished, bp, prev_probs, result)
        dist = self.dist(actor_features)

        action = rnn_hxs[0]
        action_log_probs = dist.log_probs(action)

        dist_entropy = dist.entropy().mean()

        return finished, value, action_log_probs, rnn_hxs, bp , dist_entropy, batch_finished, prev_probs, result

    def act2(self, prev_tokens, prev_hidden, prev_masked_memory, step, masked_memory, attn_list, result, prev_probs, can_stop, finished, batch_finished):
        log_probs, actor_features, value, prev_tokens, prev_hidden, prev_masked_memory, step, masked_memory, attn_list, result, prev_probs, can_stop, finished, batch_finished = self.base2(
            prev_tokens, prev_hidden, prev_masked_memory, step, masked_memory, attn_list, result, prev_probs, can_stop, finished, batch_finished)
        dist = self.dist(actor_features)

        action = prev_tokens
        action_log_probs = dist.log_probs(action)

        dist_entropy = dist.entropy().mean()

        return action_log_probs, dist_entropy,  value, prev_tokens, prev_hidden, prev_masked_memory, masked_memory, attn_list, result, prev_probs, can_stop, finished, batch_finished 


    def evaluate_actions(self, inputs, rnn_hxs, masks, action, prev_hidden):

        # modify base such that you input a new prev_hidden that you keep updating while the rest of the inputs from rnn_hxs and inputs comes from storage

        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

    def base2(self, prev_tokens, prev_hidden, prev_masked_memory, step, masked_memory, attn_list, result, prev_probs, can_stop, finished, batch_finished):
                
                
        hidden, logits, dec_output = self.model.model.model.decoder.decode_token(prev_tokens, prev_hidden, prev_masked_memory if step > 0 else
                        masked_memory, attn_list, return_dec_out=True)


        batch_size = self.args.batch_size
        beam_size = 1

        logit_size = logits.size(1)
        # log_probs: batch size x beam size x vocab size
        log_probs = F.log_softmax(logits, dim=-1).view(batch_size, -1, logit_size)
        total_log_probs = log_probs + prev_probs.unsqueeze(2)
        # log_probs_flat: batch size x beam_size * vocab_size
        log_probs_flat = total_log_probs.view(batch_size, -1)
        # indices: batch size x beam size
        # Each entry is in [0, beam_size * vocab_size)
        actual_beam_size = min(beam_size, log_probs_flat.size(1))
        prev_probs, indices = log_probs_flat.topk(actual_beam_size, dim=1)
        # prev_tokens: batch_size * beam size
        # Each entry indicates which token should be added to each beam.
        prev_tokens = (indices % logit_size).view(-1)
        # This takes a lot of time... about 50% of the whole thing.
        indices = indices.cpu()
        # k_idx: batch size x beam size
        # Each entry is in [0, beam_size), indicating which beam to extend.
        k_idx = (indices / logit_size)

        if beam_size == actual_beam_size:
            b_idx_to_use = Variable(torch.arange(0, batch_size, out=torch.LongTensor()).unsqueeze(1).repeat(1, beam_size).view(-1))
        else:
            b_idx_to_use = Variable(
                torch.arange(0, batch_size, out=torch.LongTensor()).unsqueeze(1).repeat(1, actual_beam_size).view(-1))

        idx = torch.stack([b_idx_to_use, k_idx.view(-1)])
        # prev_hidden: (batch size * beam size) x hidden size
        # Contains the hidden states which produced the top-k (k = beam size)
        # tokens, and should be extended in the step.
        prev_hidden = hidden.select_for_beams(batch_size, idx)

        prev_result = result
        result = [[] for _ in range(batch_size)]
        can_stop = True
        prev_probs_np = prev_probs.data.cpu().numpy()
        log_probs_np = log_probs.data.cpu().numpy()
        k_idx = k_idx.data.numpy()
        indices = indices.data.numpy()
        for batch_id in range(batch_size):
            if batch_finished[batch_id]:
                continue
            # print(step, finished[batch_id])
            if len(finished[batch_id]) >= beam_size:
                # If last in finished has bigger log prob then best in topk, stop.
                if finished[batch_id][-1].total_log_prob > prev_probs_np[batch_id, 0]:
                    batch_finished[batch_id] = True
                    continue
            for idx in range(actual_beam_size):
                token = indices[batch_id, idx] % logit_size
                kidx = k_idx[batch_id, idx]
                # print(step, batch_id, idx, 'token', token, kidx, 'prev', prev_result[batch_id][kidx], prev_probs.data[batch_id][idx])
                if token == 1:  # 1 == </S>
                    finished[batch_id].append(BeamSearchResult(
                        sequence=prev_result[batch_id][kidx].sequence,
                        total_log_prob=prev_probs_np[batch_id, idx],
                        log_probs=prev_result[batch_id][kidx].log_probs + [log_probs_np[batch_id, kidx, token]],
                        log_probs_torch=None))
                    result[batch_id].append(BeamSearchResult(sequence=[], log_probs=[], total_log_prob=0, log_probs_torch=[]))
                    prev_probs.data[batch_id][idx] = float('-inf')
                else:
                    result[batch_id].append(BeamSearchResult(
                        sequence=prev_result[batch_id][kidx].sequence + [token],
                        total_log_prob=prev_probs_np[batch_id, idx],
                        log_probs=prev_result[batch_id][kidx].log_probs + [log_probs_np[batch_id, kidx, token]],
                        log_probs_torch=None))
                    can_stop = False
            if len(finished[batch_id]) >= beam_size:
                # Sort and clip.
                finished[batch_id] = sorted(
                    finished[batch_id], key=lambda x: -x.total_log_prob)[:beam_size]
        #if can_stop:
            #break

        for batch_id in range(batch_size):
            # If there is deficit in finished, fill it in with highest probable results.
            if len(finished[batch_id]) < beam_size:
                i = 0
                while i < beam_size and len(finished[batch_id]) < beam_size:
                    if result[batch_id][i]:
                        finished[batch_id].append(result[batch_id][i])
                    i += 1

        value = self.critic(dec_output)

        # make max value
        min_val = torch.FloatTensor([-100]).cuda() if self.args.cuda else torch.FloatTensor([-100]) 
        log_probs = torch.max(log_probs,min_val)
    
        return log_probs, dec_output, value, prev_tokens, prev_hidden, prev_masked_memory, step, masked_memory, attn_list, result, prev_probs, can_stop, finished, batch_finished


    def base(self, finished, rnn_hxs, batch_finished, bp, prev_probs, result):

        prev_tokens, prev_hidden, prev_masked_memory, attn_list = rnn_hxs

        hidden, logits, dec_output = self.model.model.model.decoder.decode_token(prev_tokens, prev_hidden, prev_masked_memory, attn_list, return_dec_out=True)
        batch_size = self.args.batch_size
        beam_size=1

        logit_size = logits.size(1)
        # log_probs: batch size x beam size x vocab size
        log_probs = F.log_softmax(logits, dim=-1).view(batch_size, -1, logit_size)
        total_log_probs = log_probs + prev_probs.unsqueeze(2)
        # log_probs_flat: batch size x beam_size * vocab_size
        log_probs_flat = total_log_probs.view(batch_size, -1)
        # indices: batch size x beam size
        # Each entry is in [0, beam_size * vocab_size)
        actual_beam_size = min(beam_size, log_probs_flat.size(1))
        prev_probs, indices = log_probs_flat.topk(actual_beam_size, dim=1)
        # prev_tokens: batch_size * beam size
        # Each entry indicates which token should be added to each beam.
        prev_tokens = (indices % logit_size).view(-1)
        # This takes a lot of time... about 50% of the whole thing.
        indices = indices.cpu()
        # k_idx: batch size x beam size
        # Each entry is in [0, beam_size), indicating which beam to extend.
        k_idx = (indices / logit_size)

        if beam_size == actual_beam_size:
            b_idx_to_use = Variable(torch.arange(0, batch_size, out=torch.LongTensor()).unsqueeze(1).repeat(1, beam_size).view(-1))
        else:
            b_idx_to_use = Variable(
                torch.arange(0, batch_size, out=torch.LongTensor()).unsqueeze(1).repeat(1, actual_beam_size).view(-1))

        idx = torch.stack([b_idx_to_use, k_idx.view(-1)])
        # prev_hidden: (batch size * beam size) x hidden size
        # Contains the hidden states which produced the top-k (k = beam size)
        # tokens, and should be extended in the step.
        prev_hidden = hidden.select_for_beams(batch_size, idx)

        bp=1
        prev_result = result
        result = [[] for _ in range(batch_size)]
        prev_probs_np = prev_probs.data.cpu().numpy()
        log_probs_np = log_probs.data.cpu().numpy()
        k_idx = k_idx.data.numpy()
        indices = indices.data.numpy()
        for batch_id in range(batch_size):
            if batch_finished[batch_id]:
                continue
            # print(step, finished[batch_id])
            if len(finished[batch_id]) >= beam_size:
                # If last in finished has bigger log prob then best in topk, stop.
                if finished[batch_id][-1].total_log_prob > prev_probs_np[batch_id, 0]:
                    batch_finished[batch_id] = True
                    continue
            for idx in range(actual_beam_size):
                token = indices[batch_id, idx] % logit_size
                kidx = k_idx[batch_id, idx]
                # print(step, batch_id, idx, 'token', token, kidx, 'prev', prev_result[batch_id][kidx], prev_probs.data[batch_id][idx])
                if token == 1:  # 1 == </S>

                    finished[batch_id].append(BeamSearchResult(
                        sequence=prev_result[batch_id][kidx].sequence,
                        total_log_prob=prev_probs_np[batch_id, idx],
                        log_probs=prev_result[batch_id][kidx].log_probs + [log_probs_np[batch_id, kidx, token]],
                        log_probs_torch=None))
                    result[batch_id].append(BeamSearchResult(sequence=[], log_probs=[], total_log_prob=0, log_probs_torch=[]))
                    prev_probs.data[batch_id][idx] = float('-inf')
                else:
                    result[batch_id].append(BeamSearchResult(
                        sequence=prev_result[batch_id][kidx].sequence + [token],
                        total_log_prob=prev_probs_np[batch_id, idx],
                        log_probs=prev_result[batch_id][kidx].log_probs + [log_probs_np[batch_id, kidx, token]],
                        log_probs_torch=None))
                    bp = 0
            if len(finished[batch_id]) >= beam_size:
                # Sort and clip.
                finished[batch_id] = sorted(
                    finished[batch_id], key=lambda x: -x.total_log_prob)[:beam_size]

        for batch_id in range(batch_size):
            # If there is deficit in finished, fill it in with highest probable results.
            if len(finished[batch_id]) < beam_size:
                i = 0
                while i < beam_size and len(finished[batch_id]) < beam_size:
                    if result[batch_id][i]:
                        finished[batch_id].append(result[batch_id][i])
                    i += 1

            
        value = self.critic(dec_output)

        # make max value
        min_val = torch.FloatTensor([-100]).cuda() if self.args.cuda else torch.FloatTensor([-100]) 
        log_probs = torch.max(log_probs,min_val)


        return finished, value, log_probs, dec_output, (prev_tokens, prev_hidden, prev_masked_memory, attn_list), bp, batch_finished, prev_probs, result


    def select_action(self, state, return_true_code=False ):
        # labels: correct edit operations
        # probs: probability we assign each possible operation in sequence
        # orig_code: the code we try to recover
        # dec_output: output of decoder lstm + list of sequence sizes
        labels, probs, orig_code, values, reward = self.best_action_value(state, return_true_code)

        return (labels, probs, orig_code, values, reward)

    def best_action_value(self, states, return_true_code):
        logits, labels, dec_output, reward = self.model.action_value(states)
        # Yield states[-1] for orig code containing item.ref_example.code_sequence
        
        probs = torch.exp(logits)

        values = self.critic(dec_output)
        # Change to [2] for original code and [-1] for modified code
        if return_true_code:
            return labels, probs, states[-1], values, reward
        else:
            return labels, probs, [], values, reward



def print_stats(epoch, i, loss, errors, cum_reward):
    print('epoch {}'.format(epoch))
    print('i {}'.format(i))
    print('training/action_loss {}'.format(loss[0]))
    print('training/value_loss {}'.format(loss[1]))
    print('training/entropy {}'.format(loss[2]))
    print('training/reward {}'.format(loss[3]))
    print('errors {}'.format(errors))
    print('training/cummulative_reward {}'.format(cum_reward))

class PolicyTrainer(object):

    def __init__(self, args, agent_cls, env):
        self.args = args
        self.actor_critic = agent_cls(env, args)
        self.step = 0
        self.env = env
        self.broke = False
        self.update_lr = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_critic.optimizer, int(args.num_epochs*1116864/args.batch_size), eta_min=0, last_epoch=-1)

    def load_pretrained(self, model_dir, map_to_cpu, step):
        self.step = saver.load_checkpoint(self.actor_critic.model.model.model, self.actor_critic.optimizer, model_dir, map_to_cpu, step)

    def batchify(self, values, action_log_probs, adv_targ, _, lengths):

        start,end = 0,0
        start_,end_ = 0,0
        pi = []
        v_prime = []
        advantage = []
        for i,j in (zip(_,lengths)):
            end += i
            end_ += j
            v_prime.append(values[start:end].mean())
            pi.append(action_log_probs[start:end].sum())
            advantage.append(adv_targ[start_:end_].mean())
            start += i
            start_ += j
        
        pi = torch.stack(pi)
        v_prime = torch.stack(v_prime)
        advantage = torch.stack(advantage)

        return pi, v_prime, advantage

    def compute_return(self, reward, value_preds, lengths, use_gae=True):

        masks = np.ones(len(reward)+1)
        for length in lengths:
            masks[length] = 0
        masks[-1] = 0

        if use_gae:
            gae = reward[-1]-value_preds[-1]
            returns = [torch.tensor(reward[-1], dtype=torch.float32).cuda() if self.args.cuda else torch.tensor(reward[-1], dtype=torch.float32)]
            for step in reversed(range(len(reward)-1)):
                delta = reward[step] + DISCOUNT * value_preds[
                    step + 1] * masks[step +
                                            1] - value_preds[step]
                gae = delta + DISCOUNT * lmbda * gae
                returns.append(gae + value_preds[step])
        else:
            returns[0] = 0
            for step in reversed(range(len(reward))):
                returns[step] = returns[step + 1] * \
                    DISCOUNT * masks[step + 1] + rewards[step]

        returns = torch.stack(returns)
        return returns

    def prepare_ppo_batch(self, lengths, old_action_log_probs_batch, value_preds_batch, returns):

        prob_a = []
        value_preds_batch_ = []
        returns_ = []
        s,e =0,0
        for j in (lengths):
            e+= j
            prob_a.append(old_action_log_probs_batch[s:e].sum())
            value_preds_batch_.append(value_preds_batch[s:e].mean())
            returns_.append(returns[s:e].mean())
            s += j
        prob_a = torch.stack(prob_a)
        value_preds_batch_ = torch.stack(value_preds_batch_)
        returns_ = torch.stack(returns_)

        return prob_a, value_preds_batch_, returns_

    def extend_values(self,value,lengths):
        value = value.view(-1)
        value_next_state = []
        start, end = 0, 0
        not_avail_value = torch.cuda.FloatTensor([0]) if self.args.cuda else torch.FloatTensor([0])
        for i in lengths:
            end += i
            value_next_state.append(torch.cat([value[start+1:end],not_avail_value]))
            start += i
        value_next_state = torch.cat(value_next_state)

        return value_next_state

    def get_action_log_probs(self, old_action_probs_batch):

        old_action_log_probs_batch = torch.log(old_action_probs_batch)

        actions = torch.argmax(old_action_probs_batch,dim=1)

        old_action_log_probs_batch = torch.gather(old_action_log_probs_batch, dim=1, index=actions.view(-1,1))

        return old_action_log_probs_batch, actions

    def compute_action_loss(self, pi_a, prob_a, advantage, mask):
        safe_bounds = torch.cuda.FloatTensor([5]) if self.args.cuda else torch.FloatTensor([5])
        ratio = torch.exp(torch.min(pi_a - prob_a, safe_bounds))
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - self.args.clip_param,
                                1.0 + self.args.clip_param) * advantage
        action_loss = -torch.min(surr1, surr2)[mask.view(-1)==1].mean()
        return action_loss
    
    def compute_value_loss(self, value_preds_batch_, values, returns_, mask):

        if self.args.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch_ + \
                (values - value_preds_batch_).clamp(-self.args.clip_param, self.args.clip_param)
            value_losses = (values - returns_).pow(2)
            value_losses_clipped = (
                value_pred_clipped - returns_).pow(2)
            value_loss = 0.5 * torch.max(value_losses,
                                            value_losses_clipped)[mask.view(-1)==1].mean()
        else:
            value_loss = (0.5 * (returns_ - values).pow(2))[mask.view(-1)==1].mean()

        return value_loss
    
    def compute_total_loss(self, value_loss, action_loss, dist_entropy):
        self.actor_critic.optimizer.zero_grad()
        print('ppoloss')
        print((value_loss * self.args.value_loss_coef + action_loss - dist_entropy * self.args.entropy_coef))
        (value_loss * self.args.value_loss_coef + action_loss - dist_entropy * self.args.entropy_coef).backward()
        nn.utils.clip_grad_norm_(self.actor_critic.model.model.model.parameters(),
                                        self.args.max_grad_norm)
        self.actor_critic.optimizer.step()
        self.update_lr.step()
        return

    def special_order_to_batch(self, value, batches, order, action='sum'):

        value_batch = [[] for i in order]
        idx=0
        sequence = []
        for b in batches:
            sequence = sequence + list(order[:b])
        for idx, edit in enumerate(value):
            if idx == len(sequence):
                break
            else:
                i = sequence[idx]
                value_batch[i].append(value[idx])

        for i in order:
            value_batch[i] = torch.stack(value_batch[i])
            if action == 'sum':
                value_batch[i] = value_batch[i].sum()
            else:
                value_batch[i] = value_batch[i].mean()
        return torch.stack(value_batch)

    def action_batch(self, action_log_probs, lengths):
        
        start,end = 0,0
        pi = []
        for i in (lengths):
            end += i
            pi.append(action_log_probs[start:end].sum())
            start += i
        
        pi = torch.stack(pi)

        return pi

    def initialise_rollout_batch(self, batch_len):
        return torch.zeros(self.args.max_rollout_length, batch_len).cuda() if self.args.cuda else torch.zeros(self.args.max_rollout_length, batch_len)

    def mask_batch(self, batch_len):
        return torch.ones(self.args.max_rollout_length, batch_len).long().cuda() if self.args.cuda else torch.ones(self.args.max_rollout_length, batch_len).long()


    def Program_PPO_update(self, batch):

        # store data
        n= len(batch[0][2])
        mask = self.mask_batch(n)
        pi_old = self.initialise_rollout_batch(n)
        reward_s =  self.initialise_rollout_batch(n)
        value_preds_batch__origs =  self.initialise_rollout_batch(n)
        
        # Fetch stored rollout data
        for i in range(len(batch)):

            prob_a, reward, new_state, value_preds_batch, lengths, acc = batch[i][1:]
            
            pi_old[i] = prob_a
            reward_s[i] = reward

            value_preds_batch__origs[i] = value_preds_batch.view(-1)
            if i > 0:
                mask[i] = (1-reward_s[i-1].long())*mask[i-1]
        state_ = self.env.update_states(batch[-1][0], new_state)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0  

        for i in range(self.args.ppo_steps):
            
            # recalculate value
            values = self.initialise_rollout_batch(n)
            pi_a = self.initialise_rollout_batch(n)
            dist_entropy =  self.initialise_rollout_batch(n)

            # run over rollout
            for idx, b in enumerate(batch):
                labels, probs, _, value, _ = self.actor_critic.select_action(b[0],return_true_code=False)

                values[idx]=value.view(-1)

                action_log_probs = torch.log(probs)

                m = torch.distributions.Categorical(probs)

                dist_entropy[idx] = m.entropy().mean()

                pi_a[idx] = action_log_probs

            dist_entropy = dist_entropy.mean()

            # calculate essential values
            v_prime = values[1:]
            labels, probs, _, value, _ = self.actor_critic.select_action(state_,return_true_code=False)

            v_prime = torch.cat((v_prime,value.view(1,-1)))

            td_target = v_prime * DISCOUNT + reward_s
            delta = td_target - values

            # Simple PPO update
            delta = delta*mask.float()
            delta = delta.detach().cpu().numpy() if self.args.cuda else delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = DISCOUNT * lmbda * advantage + delta_t
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.cuda.FloatTensor(advantage_lst).view(-1) if self.args.cuda else torch.tensor(advantage_lst, dtype=torch.float).view(-1)
            pi_a = pi_a.view(-1)
            pi_old = pi_old.view(-1)
            td_target = td_target.view(-1)
            values = values.view(-1)
            value_preds_batch__origs = value_preds_batch__origs.view(-1)

            # Action loss
            action_loss = self.compute_action_loss(pi_a, pi_old, advantage, mask)

            # Value loss
            value_loss = self.compute_value_loss(value_preds_batch__origs, values, td_target, mask)
       
            # Total loss

            self.compute_total_loss(value_loss, action_loss, dist_entropy)

            value_loss_epoch += value_loss.item()
            action_loss_epoch += action_loss.item()
            dist_entropy_epoch += dist_entropy.item()

        value_loss_epoch /= self.args.ppo_steps
        action_loss_epoch /= self.args.ppo_steps
        dist_entropy_epoch /= self.args.ppo_steps
        reward_s = reward_s[mask==1]
        reward_epoch = reward_s.detach().cpu().numpy() if self.args.cuda else reward_s.detach().numpy()
        reward_epoch = np.mean(reward_epoch)

        return (action_loss_epoch, value_loss_epoch, dist_entropy_epoch, reward_epoch)
    
    def train(self):
        #if self.args.load_sl_model:
            #self.load_pretrained('/zhome/3f/6/108837/trained_models/trained_models/vanilla,trace_enc==none,batch_size==64,lr==1,lr_decay_steps=100000/',True, int(1769300)) #self.args.cuda
        #    self.load_pretrained('/zhome/3f/6/108837/program_synthesis/program_synthesis/models/1587190775lr001_dcs10000_sgd_roll3/',True, int(20000000)) #self.args.cuda

        #writer = TensorBoardRLWrite(self.args.model_dir, '_test1')
        agent = PPO(
            self.actor_critic,
            self.args.clip_param,
            self.args.ppo_steps,
            self.args.batch_size,
            self.args.value_loss_coef,
            self.args.entropy_coef,
            max_grad_norm=self.args.max_grad_norm)
        errors = 0
        runner = 0
        cum_reward = 0
        for epoch in range(self.args.num_epochs):
            for i, batch in enumerate(self.env.dataset_loader):
                rollouts = RolloutStorage(self.args.max_rollout_length, self.args.batch_size)
                with torch.no_grad():
                    rollouts = self.env.rollout(batch, self.actor_critic, self.args.max_rollout_length, rollouts)
                
                next_value = actor_critic.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()
                
                rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

                value_loss, action_loss, dist_entropy = agent.update(rollouts)

                            
                
                runner+=1*self.args.batch_size
                loss = self.Program_PPO_update(experience)
                cum_reward += loss[3]
                print_stats(epoch, i, loss, errors,cum_reward)
                
                #writer.add(runner,'training/action_loss', loss[0])
                #writer.add(runner,'training/value_loss', loss[1])
                #writer.add(runner,'training/entropy', loss[2])
                #writer.add(runner,'training/reward', loss[3])
                #writer.add(runner,'training/cummulative_reward', cum_reward)

                # Exact eval
                #if i % self.args.eval_every_n == 0:
                #    self.actor_critic.model.eval()
                #    stats = {'correct': 0, 'total': 0}
                #    for dev_idx, dev_batch in enumerate(self.env.devset_loader):
                #        batch_res = self.actor_critic.model.model.eval(dev_batch)
                #        stats['correct'] += batch_res['correct']
                #        stats['total'] += batch_res['total']
                #        if dev_idx > self.args.eval_n_steps:
                #            break
                #    accuracy = float(stats['correct']) / stats['total']
                #    print("Dev accuracy: %.5f" % accuracy)
                #    self.actor_critic.model.train()
                #writer.add(runner,'eval/acc', accuracy)

                #replay_buffer.add(experience)
            
            #for _ in range(self.args.num_training_steps):
            #    batch = replay_buffer.sample(1)

                #if i % self.args.eval_every_n == 0:
                #    saver.save_checkpoint(self.actor_critic.model.model.model, self.actor_critic.optimizer, int(epoch*(len(self.env.dataset_loader))+i), self.args.model_dir+self.args.model_nickname)




def main():

    parser = arguments.get_arg_parser('Training Text2Code', 'train')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    saver.save_args(args)

    # Set seed to redo nan error
    #set_seed(13)
    #torch.cuda.set_device(5)

    agent_cls = KarelAgent
    env = KarelEditEnv(args)

    trainer = PolicyTrainer(args, agent_cls, env)
    trainer.train()

if __name__ == "__main__":
    main()