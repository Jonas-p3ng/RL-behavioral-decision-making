import os
import git
import gym
import carla_gym
import inspect
import argparse
import numpy as np
import os.path as osp
from pathlib import Path
currentPath = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
# sys.path.insert(1, currentPath + '/agents/stable_baselines/')
import shutil

from stable_baselines.bench import Monitor
from stable_baselines.ddpg.policies import MlpPolicy as DDPGMlpPolicy
from stable_baselines.ddpg.policies import CnnPolicy as DDPGCnnPolicy
from stable_baselines.common.policies import MlpPolicy as CommonMlpPolicy
from stable_baselines.common.policies import MlpLstmPolicy as CommonMlpLstmPolicy
from stable_baselines.common.policies import CnnPolicy as CommonCnnPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from stable_baselines import PPO2
from stable_baselines import TRPO
from stable_baselines import A2C
from stable_baselines import SAC
from stable_baselines.sac.policies import MlpPolicy as SACMlpPolicy
from stable_baselines.sac.policies import CnnPolicy as SACCnnPolicy
from stable_baselines.common.policies import BasePolicy, nature_cnn, register_policy, sequence_1d_cnn, sequence_1d_cnn_ego_bypass_tc



from config import cfg, log_config_to_file, cfg_from_list, cfg_from_yaml_file


def apply_traffic_density(env, n_spawn_cars):
    """Synchronize traffic density across cfg, env and traffic module."""
    n_spawn_cars = int(n_spawn_cars)
    cfg.TRAFFIC_MANAGER.N_SPAWN_CARS = n_spawn_cars

    base_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    if hasattr(base_env, 'N_SPAWN_CARS'):
        base_env.N_SPAWN_CARS = n_spawn_cars
    if hasattr(base_env, 'traffic_module') and base_env.traffic_module is not None:
        base_env.traffic_module.N_SPAWN_CARS = n_spawn_cars

    print('Traffic density updated: N_SPAWN_CARS={}'.format(n_spawn_cars))


def parse_args_cfgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--env', help='environment ID', type=str, default='CarlaGymEnv-v1')
    parser.add_argument('--log_interval', help='Log interval (model)', type=int, default=100)
    parser.add_argument('--agent_id', type=int, default=None)
    parser.add_argument('--num_timesteps', type=float, default=1e7)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--play_mode', type=int, help='Display mode: 0:off, 1:2D, 2:3D ', default=0)
    parser.add_argument('--verbosity', help='Terminal mode: 0:Off, 1:Action,Reward 2:All', default=0, type=int)
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--test_model', help='test model file name', type=str, default='')
    parser.add_argument('--test_last', help='test model best or last?', action='store_true', default=False)
    parser.add_argument('--carla_host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    parser.add_argument('-p', '--carla_port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    parser.add_argument('--tm_port', default=8000, type=int, help='Traffic Manager TCP port to listen to (default: 8000)')
    parser.add_argument('--carla_res', metavar='WIDTHxHEIGHT', default='1280x720', help='window resolution (default: 1280x720)')
    parser.add_argument('--use_curriculum', dest='use_curriculum', action='store_true',
                        help='Enable staged curriculum training and override config CURRICULUM.ENABLED')
    parser.add_argument('--no_curriculum', dest='use_curriculum', action='store_false',
                        help='Disable staged curriculum training and override config CURRICULUM.ENABLED')
    parser.set_defaults(use_curriculum=None)


    args = parser.parse_args()

    args.num_timesteps = int(args.num_timesteps)

    if args.test and args.cfg_file is None:
        path = 'logs/agent_{}/'.format(args.agent_id)
        conf_list = [cfg_file for cfg_file in os.listdir(path) if '.yaml' in cfg_file]
        args.cfg_file = path + conf_list[0]

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    # visualize all test scenarios
    if args.test:
        args.play_mode = True

    return args, cfg


if __name__ == '__main__':
    args, cfg = parse_args_cfgs()
    print('Env is starting')
    env = gym.make(args.env)
    if args.play_mode:
        env.enable_auto_render()
    env.begin_modules(args)
    n_actions = env.action_space.shape[-1]  # the noise objects for DDPG

    # --------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------Training----------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------
    if cfg.POLICY.NAME == 'DDPG':
        policy = {'MLP': DDPGMlpPolicy, 'CNN': DDPGCnnPolicy}   # DDPG does not have LSTM policy
    elif cfg.POLICY.NAME == 'SAC':
        policy = {'MLP': SACMlpPolicy, 'CNN': SACCnnPolicy}     # SAC does not have LSTM policy
    else:
        policy = {'MLP': CommonMlpPolicy, 'LSTM': CommonMlpLstmPolicy, 'CNN': CommonCnnPolicy}

    if not args.test:  # training
        if args.agent_id is not None:
            # create log folder
            os.mkdir(currentPath + '/logs/agent_{}/'.format(args.agent_id))                             # create agent_id folder
            os.mkdir(currentPath + '/logs/agent_{}/models/'.format(args.agent_id))
            save_path = 'logs/agent_{}/models/'.format(args.agent_id)
            env = Monitor(env, 'logs/agent_{}/'.format(args.agent_id))    # logging monitor

            # log commit id
            repo = git.Repo(search_parent_directories=False)
            commit_id = repo.head.object.hexsha
            with open('logs/agent_{}/reproduction_info.txt'.format(args.agent_id), 'w') as f:  # Use file to refer to the file object
                f.write('Git commit id: {}\n\n'.format(commit_id))
                f.write('Program arguments:\n\n{}\n\n'.format(args))
                f.write('Configuration file:\n\n{}'.format(cfg))
                f.close()

            # save a copy of config file
            original_adr = currentPath + '/tools/cfgs/' + args.cfg_file.split('/')[-1]
            target_adr = currentPath + '/logs/agent_{}/'.format(args.agent_id) + args.cfg_file.split('/')[-1]
            shutil.copyfile(original_adr, target_adr)

        else:
            save_path = 'logs/'
            env = Monitor(env, 'logs/', info_keywords=('reserved',))                                   # logging monitor
        model_dir = save_path + '{}_final_model'.format(cfg.POLICY.NAME)                               # model save/load directory

        if cfg.POLICY.NAME == 'DDPG':
            action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                        sigma=float(cfg.POLICY.ACTION_NOISE) * np.ones(n_actions))

            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(cfg.POLICY.PARAM_NOISE_STD),
                                                 desired_action_stddev=float(cfg.POLICY.PARAM_NOISE_STD))
            model = DDPG(policy[cfg.POLICY.NET], env, verbose=1, param_noise=param_noise, action_noise=action_noise,
                         policy_kwargs={'cnn_extractor': eval(cfg.POLICY.CNN_EXTRACTOR)})
        elif cfg.POLICY.NAME == 'PPO2':
            model = PPO2(policy[cfg.POLICY.NET], env, verbose=1, model_dir=save_path, policy_kwargs={'cnn_extractor': eval(cfg.POLICY.CNN_EXTRACTOR)})
        elif cfg.POLICY.NAME == 'TRPO':
            model = TRPO(policy[cfg.POLICY.NET], env, verbose=1, model_dir=save_path, policy_kwargs={'cnn_extractor': eval(cfg.POLICY.CNN_EXTRACTOR)})
        elif cfg.POLICY.NAME =='A2C':
            model = A2C(policy[cfg.POLICY.NET], env, verbose=1, model_dir=save_path, policy_kwargs={'cnn_extractor': eval(cfg.POLICY.CNN_EXTRACTOR)})
        elif cfg.POLICY.NAME == 'SAC':
            sac_cfg = cfg.get('SAC', {})
            model = SAC(
                policy[cfg.POLICY.NET],
                env,
                verbose=1,
                gamma=float(sac_cfg.get('GAMMA', 0.99)),
                learning_rate=float(sac_cfg.get('LEARNING_RATE', 3e-4)),
                buffer_size=int(sac_cfg.get('BUFFER_SIZE', 50000)),
                learning_starts=int(sac_cfg.get('LEARNING_STARTS', 100)),
                train_freq=int(sac_cfg.get('TRAIN_FREQ', 1)),
                batch_size=int(sac_cfg.get('BATCH_SIZE', 64)),
                tau=float(sac_cfg.get('TAU', 0.005)),
                ent_coef=sac_cfg.get('ENT_COEF', 'auto'),
                target_update_interval=int(sac_cfg.get('TARGET_UPDATE_INTERVAL', 1)),
                gradient_steps=int(sac_cfg.get('GRADIENT_STEPS', 1)),
                target_entropy=sac_cfg.get('TARGET_ENTROPY', 'auto'),
                random_exploration=float(sac_cfg.get('RANDOM_EXPLORATION', 0.0)),
                policy_kwargs={'cnn_extractor': eval(cfg.POLICY.CNN_EXTRACTOR)}
            )
        else:
            print(cfg.POLICY.NAME)
            raise Exception('Algorithm name is not defined!')

        print('Model is Created')
        try:
            print('Training Started')

            def train_for_steps(total_timesteps, reset_num_timesteps):
                if cfg.POLICY.NAME == 'DDPG':
                    model.learn(total_timesteps=total_timesteps, log_interval=args.log_interval,
                                save_path=save_path, reset_num_timesteps=reset_num_timesteps)
                else:
                    model.learn(total_timesteps=total_timesteps, log_interval=args.log_interval,
                                reset_num_timesteps=reset_num_timesteps)

            curriculum_cfg = cfg.get('CURRICULUM', {})
            use_curriculum = curriculum_cfg.get('ENABLED', False) if args.use_curriculum is None else args.use_curriculum
            if use_curriculum:
                stages = curriculum_cfg.get('STAGES', [])
                stage_ratios = [float(stage.get('RATIO', 0.0)) for stage in stages]
                total_ratio = sum(stage_ratios)

                if len(stages) == 0 or total_ratio <= 0:
                    print('Curriculum is enabled but invalid, fallback to single-stage training.')
                    train_for_steps(args.num_timesteps, True)
                else:
                    allocated_steps = 0
                    for i, stage in enumerate(stages):
                        if i < len(stages) - 1:
                            stage_steps = int(args.num_timesteps * stage_ratios[i] / total_ratio)
                            allocated_steps += stage_steps
                        else:
                            stage_steps = args.num_timesteps - allocated_steps

                        stage_density = int(stage.get('N_SPAWN_CARS', cfg.TRAFFIC_MANAGER.N_SPAWN_CARS))
                        apply_traffic_density(env, stage_density)
                        print('Curriculum stage {}/{}: steps={}, N_SPAWN_CARS={}'.format(
                            i + 1, len(stages), stage_steps, stage_density
                        ))
                        train_for_steps(stage_steps, reset_num_timesteps=(i == 0))
            else:
                train_for_steps(args.num_timesteps, True)
        finally:
            print(100 * '*')
            print('FINISHED TRAINING; saving model...')
            print(100 * '*')
            # save model even if training fails because of an error
            model.save(model_dir)
            env.destroy()
            print('model has been saved.')

    # --------------------------------------------------------------------------------------------------------------------"""
    # ------------------------------------------------Test----------------------------------------------------------------"""
    # --------------------------------------------------------------------------------------------------------------------"""

    else:  # test
        if args.agent_id is not None:
            save_path = 'logs/agent_{}/models/'.format(args.agent_id)
        else:
            save_path = 'logs/'

        if args.test_model == '':
            best_last = 'best'
            if args.test_last:
                best_last = 'step'
            best_s = [int(best[5:-4])for best in os.listdir(save_path) if best_last in best]
            best_s.sort()
            args.test_model = best_last + '_{}'.format(best_s[-1])

        model_dir = save_path + args.test_model  # model save/load directory
        print('{} is Loading...'.format(args.test_model))
        if cfg.POLICY.NAME == 'DDPG':
            model = DDPG.load(model_dir)
            model.action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                              sigma=np.zeros(n_actions))
            model.param_noise = None
        elif cfg.POLICY.NAME == 'PPO2':
            model = PPO2.load(model_dir)
        elif cfg.POLICY.NAME == 'TRPO':
            model = TRPO.load(model_dir)
        elif cfg.POLICY.NAME == 'A2C':
            model = A2C.load(model_dir)
        elif cfg.POLICY.NAME == 'SAC':
            model = SAC.load(model_dir)
        else:
            print(cfg.POLICY.NAME)
            raise Exception('Algorithm name is not defined!')

        print('Model is loaded')
        try:
            obs = env.reset()
            while True:
                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                env.render()
                if done:
                    obs = env.reset()
        finally:
            env.destroy()
