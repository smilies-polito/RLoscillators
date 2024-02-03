import argparse
from pathlib import Path

from stable_baselines3 import PPO
import os
import time
from osc_env import OscEnv


# from stable_baselines3.common.env_checker import check_env

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Optimization of synthetic oscillatory biological networks through "
                                                 "Reinforcement Learning",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model-path", nargs='?', const=1, type=str, default="../data/y1_otero_rev.ant", help="model file path")
    parser.add_argument("-s", "--steps", nargs='?', const=1, type=int, default=32, help="number of steps")
    parser.add_argument("-b", "--batch-size", nargs='?', const=1, type=int, default=8, help="batch size")
    parser.add_argument("-l", "--learning-rate", nargs='?', const=1, type=float, default=0.001, help="learning")
    parser.add_argument("-e", "--episodes", nargs='?', const=1, type=int, default=1000, help="episodes")
    parser.add_argument("-v", "--verbose", action="store_true", help="increase verbosity")
    args = parser.parse_args()
    config = vars(args)
    print(config)

    models_dir = f"../output/models/{int(time.time())}/"
    log_dir = f"../output/logs/{int(time.time())}/"
    render_dir = f"../output/render/{int(time.time())}/"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(render_dir):
        os.makedirs(render_dir)

    tb_log_name = Path(config['model_path']).stem

    try:
        env = OscEnv(config['model_path'])
    except Exception as e:
        raise e
    # env.reset()
    # check_env(env)

    steps = config['steps']
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir, n_steps=steps,
                batch_size=config['batch_size'], learning_rate=config['learning_rate'])
    for episode in range(config['episodes']):
        env.reset()
        env.model.reset()
        env.plot(f"{render_dir}", f"{steps * episode}_start")
        env.model.reset()
        model.learn(total_timesteps=steps, reset_num_timesteps=False, progress_bar=True, tb_log_name=tb_log_name)
        env.model.reset()
        env.plot(f"{render_dir}", f"{steps * episode}_end")
        model.save(f"{models_dir}/{steps * episode}")
