import os
import sys
sys.path = [os.path.abspath(os.path.dirname((os.path.dirname(__file__))))] + sys.path[1:]


from tools import saver
from rl_agent.agent import KarelAgent
from rl_agent.environment import KarelEditEnv
from rl_agent.trainer import PolicyTrainer, PPOTrainer


def shortcut():
    args = saver.ArgsDict(
        num_iterations=100, max_rollout_length=32, replay_buffer_size=16384, max_token_per_code=75,
        num_rollouts=16, num_training_steps=16, batch_size=32,
        update_actor_it=10, save_actor_it=10, lr=0.01,
        rl_discount=.9, rl_eps_action=.1, rl_eps_parameter=.5, rl_alpha=.7, her=True, her_new_goals=30,
        karel_io_enc='lgrl', cuda=False, train_task_encoder=False, train_from_scratch=True, model_dir='data')#'logdirs/20180207/karel-lgrl-ref-edit-m123-sgd-cl1-lr0.1-lds100k-ldr0.5') #train_sratch=False, dir='data'

    env = KarelEditEnv(args.max_token_per_code)

    trainer = PPOTrainer(KarelAgent, env, args)
    trainer.train()


if __name__ == "__main__":
    shortcut()
