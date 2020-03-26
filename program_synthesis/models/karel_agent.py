import collections
import copy

import numpy as np
import operator

import torch
import torch.nn as nn
from torch import optim

import arguments
#from program_synthesis.common.tools import saver
from models import get_model
from datasets import data, dataset, set_vocab
from datasets.karel import refine_env, mutation
from tools import saver
from tools.reporter import TensorBoardRLWrite
from . import prepare_spec, karel_model
from .modules import karel, attention


# TODO: Make sure that it is possible to choose from vocab size action space
# Alternatively have 2 more: remove token, add token where add token is treated in a second selection

DISCOUNT = 0.99
lmbda = 0.95
EPISLON = 0.1
ALPHA = 0.7

StepExample = collections.namedtuple('StepExample', ['state', 'action', 'reward', 'new_state', 'value', 'lengths'])


#  https://arxiv.org/pdf/1511.04143.pdf

class KarelEditEnv(object):

    def __init__(self, args):

        self.vocab = data.PlaceholderVocab(
            data.load_vocab(dataset.relpath('../data/karel/word.vocab')), 0)

        self.refiner = karel_model.KarelLGRLRefineBatchProcessor(args, self.vocab, False)
        self.dataset_loader = dataset.get_karel_dataset_nomodel(args, self.refiner)

        self.data_iter = self.dataset_loader.__iter__()
        self._cur_env = None
        self._cuda = args.cuda

    def reset(self):
        example = self.data_iter.next()
 
        return example

    def prepare_tasks(self, tasks):
        input_grids = torch.cat([t[0] for t in tasks], dim=0)
        output_grids = torch.cat([t[1] for t in tasks], dim=0)
        return (input_grids, output_grids)

    def prepare_states(self, states):
        return torch.cat(states, dim=0)

    def prepare_obs(self, obs):
        #current_code = prepare_spec.lists_padding_to_tensor(
        #    [obs['code']], self.vocab.stoi, cuda=False, volatile=True
        #)
        
        current_code = prepare_spec.lists_to_packed_sequence(
             [obs['code']], self.vocab.stoi, cuda=False, volatile=True)
        return current_code

    def prepare_actions(self, actions):
        return actions

    def compute_reward(self, ground_truth_edits, generated_edits, lengths, simple=True):
        rewards = []

        if not simple:
            start,end = 0,0
            for idx, length in enumerate(lengths):
                m = length
                reward = 0
                end += length
                g_t_edit = ground_truth_edits[start:end]
                gen_edit = generated_edits[start:end]
                for i in range(m):
                    if (i <= m):
                        token_true = ground_truth_edits[i]
                        token_gen = generated_edits[i]
                        if token_true == token_gen:
                            reward += 1
                        if token_true != token_gen:
                            reward -= 1
                    else:
                        reward -= 1
                reward /= length
                rewards.append(reward)
                start += length
        else:
            reward = 0
            for i in range(len(ground_truth_edits)):
                if ground_truth_edits[i] == generated_edits[i]:
                    reward = 1
                else:
                    reward = -1
                rewards.append(reward)

        return rewards

    def edit_space_to_code_space(self, edit_codes, code_seqs, lengths):
        
        def new_token(idx, edit_token, code_seq):
            #(*) make this guy
            #    0: <s>
            #   1: </s>
            #   2: keep
            #   3: delete
            #   4: insert vocab 0
            #   5: replace vocab 0
            #   6: insert vocab 1
            #   7: replace vocab 1
            if edit_token == 0:
                return 0, code_seq, idx
            if edit_token == 1:
                return 1, code_seq, idx
            if edit_token == 2:
                op = 'keep'
            if edit_token == 3:
                op = 'delete'
            if edit_token%2 == 0:
                op = 'insert'
            if edit_token%2 != 0:
                op = 'replace'
            
            computed_token = int(np.floor((edit_token.item() - 4)/2))
            if op == 'keep':
                return int(code_seq[idx]), code_seq, idx+1
            if op == 'delete':
                if idx==0:
                    return None, code_seq[1:], idx
                code_seq = code_seq[:idx-1] + code_seq[idx:]
                return None, code_seq, idx
            if op == 'insert':
                code_seq = torch.cat((code_seq[:idx],torch.tensor([computed_token]),code_seq[idx:]))
                return int(code_seq[idx]), code_seq, idx+1
            if op == 'replace':
                if idx >= len(code_seq):
                    return None, code_seq
                code_seq[idx] = computed_token
                return int(computed_token), code_seq, idx

        next_states = []
        start, end = 0,0
        for i in range(code_seqs.shape[0]):
            end += lengths[i]
            next_state = []
            edit_code = edit_codes[start:end]
            code_seq = code_seqs[i]

            idx = 0
            for edit_token in edit_code:
                value, code_seq, idx = new_token(idx, edit_token, code_seq)
                if value != None:
                    next_state.append(value)
            if idx < len(code_seq):
                for j in range(idx, len(code_seq)):
                    next_state.append(int(code_seq[j]))

            start += lengths[i]

            next_states.append(next_state)

        return next_states

    def update_states(self, states, new_states):

        def list_length(new_states):

            lengths_lst = []
            for i in range(len(new_states)):
                lengths_lst.append(len(new_states[i]))

            return lengths_lst

        input_grids, output_grids, code_seqs, \
            dec_data, ref_code, ref_trace_grids,\
                ref_trace_events, cag_interleave, orig_examples = states


        lists_sorted, sort_to_orig, orig_to_sort = prepare_spec.sort_lists_by_length(new_states)

        v = prepare_spec.numpy_to_tensor(prepare_spec.lists_to_numpy_novocab(lists_sorted, 0), False, False)
        lens = prepare_spec.lengths(lists_sorted)
        ref_code_new =  prepare_spec.PackedSequencePlus(torch.nn.utils.rnn.pack_padded_sequence(
                v, lens, batch_first=True), lens, sort_to_orig, orig_to_sort)

        ref_code_new = ref_code_new.with_new_ps(torch.nn.utils.rnn.PackedSequence(torch.tensor((ref_code_new.ps.data.numpy()>0) *ref_code_new.ps.data.numpy()), ref_code_new.ps.batch_sizes))

        dec_data_new = self.refiner.compute_edit_ops_no_char(new_states, code_seqs, ref_code_new)
        
        return (input_grids, output_grids, code_seqs, dec_data_new, ref_code_new, ref_trace_grids, ref_trace_events, cag_interleave, orig_examples)
    
    def rollout(self, state, agent, max_rollout_length):
        #state = env.reset()
        # (*) check if traces work here
        # Update they don't becuase the SL model doesn't even have traces: they are equal to none :/
        experience = []
        success = False
        for _ in range(max_rollout_length):
            # And here when the code_encoder is used 
            action = agent.select_action(state, return_true_code=True )
            new_state, reward, done, _ = self.step(action)
            experience.append(StepExample(state, action[1], reward, new_state, action[3], action[4]))
            new_state = self.update_states(state, new_state)
            state = new_state
            #if done:
            #    success = True
            #    break
        return success, experience


    def step(self, action):
        #(*) labels are for some reason very similar to gen results after 6 minutes..
        # Rewrite this guy to be about using the decoder output again!
        (labels, probs, code_seq, values, lengths) = action

        actions = torch.argmax(probs,dim=1)

        reward = self.compute_reward(labels, actions, lengths, simple=True)

        next_state = self.edit_space_to_code_space(actions, code_seq, lengths)

        #obs, reward, done, info = self._cur_env.step(action)
        return next_state, reward, True, {}


class KarelEditPolicy(nn.Module):
    def __init__(self, vocab_size, args):
        super(KarelEditPolicy, self).__init__()
        self.args = args

        set_vocab(args)
        self.model = get_model(args)

    def encode(self, input_grid, output_grid):
        return self.model.model.encoder(input_grid,output_grid)

    def encode_code(self, code_state, input_grid, output_grid, traces, trace_events):
        return self.model.model.code_encoder(code_state, input_grid, output_grid, traces, trace_events)

    def encode_trace(self, ref_code_memory, ref_trace_grids, ref_trace_events, cag_interleave ):
        return self.model.model.trace_encoder(ref_code_memory, ref_trace_grids,ref_trace_events, cag_interleave)

    def decode(self, io_embed, ref_code_memory, ref_trace_memory, code_seqs, dec_data):
        return self.model.model.decoder.rl_forward(io_embed, ref_code_memory, ref_trace_memory, code_seqs, dec_data)

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

        logits, labels, dec_output, lengths = self.decode(io_embed, ref_code_memory,
                                           ref_trace_memory, code_seqs,
                                           dec_data)
        
        if not self.args.use_code_level_state:
            return logits, labels, dec_output, lengths
        else:
            return logits, labels, torch.cat((io_embed.mean(dim=1),ref_code_memory.state[0].view(code_seqs.size(0),-1)),1), lengths
#torch.cat((io_embed.mean(dim=1),ref_code_memory.state[0].view(code_seqs.size(0),-1)),1)

class KarelAgent(object):

    def __init__(self, env, args):
        self.vocab = env.vocab
        self.model = KarelEditPolicy(30, args)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.model.model.parameters(), lr=args.lr)
        if not args.use_code_level_state:
            self.critic = nn.Linear(256, 1, bias=False).cuda() if args.cuda else nn.Linear(256, 1, bias=False)
        else:
            self.critic = nn.Linear(512*3, 1, bias=False).cuda() if args.cuda else nn.Linear(512*3, 1, bias=False)

    def select_action(self, state, return_true_code=False ):
        # labels: correct edit operations
        # probs: probability we assign each possible operation in sequence
        # orig_code: the code we try to recover
        # dec_output: output of decoder lstm + list of sequence sizes
        labels, probs, orig_code, values, lengths = self.best_action_value(state, return_true_code)

        return (labels, probs, orig_code, values, lengths)

    def best_action_value(self, states, return_true_code):
        logits, labels, dec_output, lengths = self.model.action_value(states)
        
        probs = nn.functional.softmax(logits,dim=1)

        values = self.critic(dec_output)
        if return_true_code:
            return labels, probs, states[2], values, lengths
        else:
            return labels, probs, [], values, lengths

class ReplayBuffer(object):

    def __init__(self, max_size, erase_factor):
        self.max_size = max_size
        self.erase_factor = erase_factor
        self.buffer = []

    @property
    def size(self):
        return len(self.buffer)

    def add(self, experience):
        self.buffer.extend(experience)
        if len(self.buffer) >= self.max_size:
            self.buffer = self.buffer[int(self.erase_factor * self.size):]

    def sample(self, size):
        replace_mode = size > len(self.buffer)
        index = np.random.choice(self.size, size=size, replace=replace_mode)
        return [self.buffer[idx] for idx in index]


def rollout(env, state, agent, epsilon_greedy, max_rollout_length):
    eps = EPISLON if epsilon_greedy else None
    #state = env.reset()
    # (*) check if traces work here
    # Update they don't becuase the SL model doesn't even have traces: they are equal to none :/
    experience = []
    success = False
    for _ in range(max_rollout_length):
        # And here when the code_encoder is used 
        action = agent.select_action(state, return_true_code=True )
        new_state, reward, done, _ = env.step(action)
        experience.append(StepExample(state, action[1], reward, new_state, action[3], action[4]))
        if done:
            success = True
            break
    return success, experience


class PolicyTrainer(object):

    def __init__(self, args, agent_cls, env):
        self.args = args
        self.actor_critic = agent_cls(env, args)
        self.env = env

    def train_actor_critic(self, batch):
        states = self.env.prepare_states([ex.state for ex in batch])
        new_states = self.env.prepare_states([ex.new_state for ex in batch])
        actions = self.env.prepare_actions([ex.action for ex in batch])
        targets = np.zeros(len(batch))
        # prepare_batch(batch)
        value = self.critic.action_value(states)
        new_value, _ = self.actor_critic.best_action_value(new_states)
        for idx, ex in enumerate(batch):
            Q_s_a = value[idx][ex.action]
            new_Q_s_a = 0 if ex.reward == 0 else (ex.reward + DISCOUNT * new_value[idx])
            targets[idx] = Q_s_a + ALPHA * (new_Q_s_a - Q_s_a)
        self.critic.train(tasks, states, actions, targets)

    def update_states(self, states, new_states):

        def list_length(new_states):

            lengths_lst = []
            for i in range(len(new_states)):
                lengths_lst.append(len(new_states[i]))

            return lengths_lst

        input_grids, output_grids, code_seqs, \
            dec_data, ref_code, ref_trace_grids,\
                ref_trace_events, cag_interleave, orig_examples = states


        lists_sorted, sort_to_orig, orig_to_sort = prepare_spec.sort_lists_by_length(new_states)

        v = prepare_spec.numpy_to_tensor(prepare_spec.lists_to_numpy_novocab(lists_sorted, 0), False, False)
        lens = prepare_spec.lengths(lists_sorted)
        ref_code_new =  prepare_spec.PackedSequencePlus(torch.nn.utils.rnn.pack_padded_sequence(
                v, lens, batch_first=True), lens, sort_to_orig, orig_to_sort)

        ref_code_new = ref_code_new.with_new_ps(torch.nn.utils.rnn.PackedSequence(torch.tensor((ref_code_new.ps.data.numpy()>0) *ref_code_new.ps.data.numpy()), ref_code_new.ps.batch_sizes))

        dec_data_new = self.env.refiner.compute_edit_ops_no_char(new_states, code_seqs, ref_code_new)
        
        return (input_grids, output_grids, code_seqs, dec_data_new, ref_code_new, ref_trace_grids, ref_trace_events, cag_interleave, orig_examples)
    
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

    def compute_action_loss(self, pi_a, prob_a, advantage):
        ratio = torch.exp(pi_a - prob_a)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - self.args.clip_param,
                                1.0 + self.args.clip_param) * advantage
        action_loss = -torch.min(surr1, surr2).mean()
        return action_loss
    
    def compute_value_loss(self, value_preds_batch_, values, returns_):

        if self.args.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch_ + \
                (values - value_preds_batch_).clamp(-self.args.clip_param, self.args.clip_param)
            value_losses = (values - returns_).pow(2)
            value_losses_clipped = (
                value_pred_clipped - returns_).pow(2)
            value_loss = 0.5 * torch.max(value_losses,
                                            value_losses_clipped).mean()
        else:
            value_loss = 0.5 * (returns_ - values).pow(2).mean()

        return value_loss
    
    def compute_total_loss(self, value_loss, action_loss, dist_entropy):
        self.actor_critic.optimizer.zero_grad()
        (value_loss * self.args.value_loss_coef + action_loss - dist_entropy * self.args.entropy_coef).backward()
        nn.utils.clip_grad_norm_(self.actor_critic.model.model.model.parameters(),
                                        self.args.max_grad_norm)
        self.actor_critic.optimizer.step()
        return

    def action_batch(self, action_log_probs, lengths):
        
        start,end = 0,0
        pi = []
        for i in (lengths):
            end += i
            pi.append(action_log_probs[start:end].sum())
            start += i
        
        pi = torch.stack(pi)

        return pi

    def initialise_rollout_batch(self):
        return torch.zeros(self.args.max_rollout_length, self.args.batch_size).cuda() if self.args.cuda else torch.zeros(self.args.max_rollout_length, self.args.batch_size)

    def Program_PPO_update(self, batch):
        
        # store data
        pi_old = self.initialise_rollout_batch()
        reward_s =  self.initialise_rollout_batch()
        value_preds_batch__origs =  self.initialise_rollout_batch()
        
        # Fetch stored rollout data
        for i in range(len(batch)):

            old_action_log_probs_batch, reward, new_state, value_preds_batch, lengths = batch[i][1:]
            
            get_action_log_probs, actions = self.get_action_log_probs(old_action_log_probs_batch)

            reward = torch.cuda.FloatTensor(reward) if self.args.cuda else torch.FloatTensor(reward)

            prob_a, _, reward_ = self.prepare_ppo_batch(lengths, get_action_log_probs, get_action_log_probs.view(-1), reward)
            pi_old[i] = prob_a
            reward_s[i] = reward_

            value_preds_batch__origs[i] = value_preds_batch.view(-1)

        state_ = self.update_states(batch[-1][0], new_state)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0  

        for i in range(self.args.ppo_steps):
            
            # recalculate value
            values = self.initialise_rollout_batch()
            pi_a = self.initialise_rollout_batch()
            dist_entropy =  self.initialise_rollout_batch()

            # run over rolloout
            for idx, b in enumerate(batch):
                labels, probs, _, value, _ = self.actor_critic.select_action(b[0],return_true_code=False)
                values[idx]=value.view(-1)

                # get log probs and entropy
                action_log_probs = torch.log(probs)

                m = torch.distributions.Categorical(probs)

                dist_entropy[idx] = m.entropy().mean()

                action_log_probs = torch.gather(action_log_probs, dim=1, index=torch.argmax(probs,dim=1).view(-1,1))
                
                pi_new = self.action_batch(action_log_probs, _)
                pi_a[idx] = pi_new

            dist_entropy = dist_entropy.mean()

            # calculate essential values
            v_prime = values[1:]
            labels, probs, _, value, _ = self.actor_critic.select_action(state_,return_true_code=False)
            v_prime = torch.cat((v_prime,value.view(1,-1)))

            td_target = v_prime * DISCOUNT
            delta = td_target - values

            # Simple PPO update
            delta = delta + reward_s
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
            action_loss = self.compute_action_loss(pi_a, pi_old, advantage)

            # Value loss
            value_loss = self.compute_value_loss(value_preds_batch__origs, values, td_target)
       
            # Total loss

            self.compute_total_loss(value_loss, action_loss, dist_entropy)

            value_loss_epoch += value_loss.item()
            action_loss_epoch += action_loss.item()
            dist_entropy_epoch += dist_entropy.item()

        value_loss_epoch /= self.args.ppo_steps
        action_loss_epoch /= self.args.ppo_steps
        dist_entropy_epoch /= self.args.ppo_steps
        reward_epoch = reward_s.detach().cpu().numpy() if self.args.cuda else reward_s.detach().numpy()
        reward_epoch = np.mean(reward_epoch)

        return (action_loss_epoch, value_loss_epoch, dist_entropy_epoch, reward_epoch)

    def Simple_PPO_update(self, batch):

        # Fetch rollout data
        state, old_action_log_probs_batch, reward, new_state, value_preds_batch, lengths = batch[0]

        get_action_log_probs, actions = self.get_action_log_probs(old_action_log_probs_batch)

        reward = torch.cuda.FloatTensor(reward) if self.args.cuda else torch.FloatTensor(reward)
        # convert rollout to a batch
        prob_a, value_preds_batch__orig, reward_ = self.prepare_ppo_batch(lengths, get_action_log_probs, value_preds_batch.view(-1), reward)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0  

        for i in range(self.args.ppo_steps):
            
            # try instead of using state_ to use state here and split the output value into v_s and v_prime as in PPO itself            

            # value function for next state
            labels, probs, _, values, _ = self.actor_critic.select_action(state,return_true_code=False)

            # extend values
            v_prime = self.extend_values(values,_)
            
            # get log probs and entropy
            action_log_probs = torch.log(probs)

            m = torch.distributions.Categorical(probs)

            dist_entropy = m.entropy().mean()

            action_log_probs = torch.gather(action_log_probs, dim=1, index=torch.argmax(probs,dim=1).view(-1,1))
        

            # batchify
            pi_a, not_used, not_used = self.batchify(action_log_probs, action_log_probs, value_preds_batch.view(-1), _, lengths)
            not_used, not_used, td_target = self.batchify(action_log_probs, action_log_probs, v_prime*DISCOUNT, _, lengths) 
            not_used, not_used, delta = self.batchify(action_log_probs, action_log_probs, v_prime*DISCOUNT-values, _, lengths)  
            not_used, not_used, values_ = self.batchify(action_log_probs, action_log_probs, v_prime, _, lengths) 
            not_used, not_used, value_preds_batch_ = self.batchify(action_log_probs, action_log_probs, values, _, lengths) 

            # Simple PPO update
            delta = delta + reward_
            delta = delta.detach().cpu().numpy() if self.args.cuda else delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = DISCOUNT * lmbda * advantage + delta_t
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.cuda.FloatTensor(advantage_lst).view(-1) if self.args.cuda else torch.tensor(advantage_lst, dtype=torch.float).view(-1)

            # Action loss
            action_loss = self.compute_action_loss(pi_a, prob_a, advantage)

            # Value loss
            value_loss = self.compute_value_loss(value_preds_batch__orig, value_preds_batch_, td_target)
       
            # Total loss

            self.compute_total_loss(value_loss, action_loss, dist_entropy)

            value_loss_epoch += value_loss.item()
            action_loss_epoch += action_loss.item()
            dist_entropy_epoch += dist_entropy.item()

        value_loss_epoch /= self.args.ppo_steps
        action_loss_epoch /= self.args.ppo_steps
        dist_entropy_epoch /= self.args.ppo_steps
        reward_epoch = reward.detach().cpu().numpy() if self.args.cuda else reward.detach().numpy()
        reward_epoch = np.mean(reward_epoch)

        return (action_loss_epoch, value_loss_epoch, dist_entropy_epoch, reward_epoch)

    def PPO_update(self, batch):

        # Fetch rollout data
        state, old_action_log_probs_batch, reward, new_state, value_preds_batch, lengths = batch[0]

        get_action_log_probs, actions = self.get_action_log_probs(old_action_log_probs_batch)

        # Compute advantage
        returns = self.compute_return(reward,value_preds_batch.view(-1), lengths, use_gae=True)
        advantages = returns - value_preds_batch.view(-1)
        adv_targ = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # convert rollout to a batch
        prob_a, value_preds_batch_, returns_ = self.prepare_ppo_batch(lengths, get_action_log_probs, value_preds_batch.view(-1), returns)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.args.ppo_steps):
            
            # get value of states using current policy
            labels, probs, _, values, _ = self.actor_critic.select_action(state,return_true_code=False)

            action_log_probs = torch.log(probs)

            m = torch.distributions.Categorical(probs)

            dist_entropy = m.entropy().mean()

            action_log_probs = torch.gather(action_log_probs, dim=1, index=torch.argmax(probs,dim=1).view(-1,1))

            # batchify
            
            pi_a, values, advantage = self.batchify(values, action_log_probs, adv_targ, _, lengths) 
                
            # Action loss
            action_loss = self.compute_action_loss(pi_a, prob_a, advantage)

            # Value loss
            value_loss = self.compute_value_loss(value_preds_batch_, values, returns_)
                
            # Total loss

            self.compute_total_loss(value_loss, action_loss, dist_entropy)

            value_loss_epoch += value_loss.item()
            action_loss_epoch += action_loss.item()
            dist_entropy_epoch += dist_entropy.item()

        value_loss_epoch /= self.args.ppo_steps
        action_loss_epoch /= self.args.ppo_steps
        dist_entropy_epoch /= self.args.ppo_steps
        reward_epoch = np.mean(np.array(reward))

        return (action_loss_epoch, value_loss_epoch, dist_entropy_epoch, reward_epoch)

    def train(self):
        #writer = TensorBoardRLWrite(self.args.model_dir, '_test1')
        replay_buffer = ReplayBuffer(int(self.args.replay_buffer_size), self.args.erase_factor)
        runner = 0
        cum_reward = 0
        for epoch in range(self.args.num_epochs):
            
            for i, batch in enumerate(self.env.dataset_loader):
            #for i in range(self.args.num_episodes):
                with torch.no_grad():
                    _, experience = self.env.rollout(batch, self.actor_critic, self.args.max_rollout_length)
                runner+=1*self.args.batch_size
                loss = self.Program_PPO_update(experience)
                print('epoch {}'.format(epoch))
                print('i {}'.format(i))
                print('training/action_loss {}'.format(loss[0]))
                print('training/value_loss {}'.format(loss[1]))
                print('training/entropy {}'.format(loss[2]))
                print('training/reward {}'.format(loss[3]))
                cum_reward += loss[3]
                print('training/cummulative_reward {}'.format(cum_reward))
                writer.add(runner,'training/action_loss', loss[0])
                writer.add(runner,'training/value_loss', loss[1])
                writer.add(runner,'training/entropy', loss[2])
                writer.add(runner,'training/reward', loss[3])
                writer.add(runner,'training/cummulative_reward', cum_reward)
                #replay_buffer.add(experience)
            
            #for _ in range(self.args.num_training_steps):
            #    batch = replay_buffer.sample(1)

            if (epoch + 1) % 1 == 0:
            #if i%int(self.args.num_episodes/2) ==0:
                saver.save_checkpoint(self.actor_critic.model.model.model, self.actor_critic.optimizer, epoch, self.args.model_dir)



def main():

    parser = arguments.get_arg_parser('Training Text2Code', 'train')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    agent_cls = KarelAgent
    env = KarelEditEnv(args)

    trainer = PolicyTrainer(args, agent_cls, env)
    trainer.train()

if __name__ == "__main__":
    main()