import argparse
import numpy as np

def parse_args():
    # ++ Basic Configs ++
    parser = argparse.ArgumentParser(description="TD")
    parser.add_argument("--desc", type=str, default="No desc", help="train desc")
    # parser.add_argument("--seed", type=int, default=777, help="random seed")
    parser.add_argument("--seed", type=int, default=777, help="random seed")
    parser.add_argument("--device", type=str, default="cpu", help="train on device")
    
    # ++ Env Configs ++
    parser.add_argument("--env_max_time", type=int, default=100, help="num of time per epoch")
    parser.add_argument("--env_nue", type=int, default=5, help="num of ue")
    parser.add_argument("--nUE", type=int, default=5, help="n ue")
    parser.add_argument("--num_frame", type=int, default=1000, help="how many times agent train")
    
    # ++ Train Configs ++

    
    
    # ++ DQN Configs ++
    parser.add_argument("--memory_size", type=int, default=500, help="DQN replay memory size")
    parser.add_argument("--max_epsilon", type=float, default=1, help="max epsilon")
    parser.add_argument("--min_epsilon", type=float, default=0.1, help="min epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=1/50, help="epsilon decay")
    parser.add_argument("--batch_size", type=int, default=1, help="get from memory pool")

    
    # ++ Optim Configs ++
    parser.add_argument("--gamma", type=float, default=0.9, help="gamma")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--optim_frame", type=int, default=1, help="update one time every N frame")
    parser.add_argument("--target_update_frame", type=int, default=200, help="update target net every N frame")
    # parser.add_argument("--change_UEs_cycle", type=int, default=10000, help="every N step change UEs")
    
    # ++ Log Config ++
    parser.add_argument("--train_log_cycle", type=int, default=100, help="every N step log")

    # ++ TimeFreeze Config ++
    # parser.add_argument("--timefreeze_cycle", type=int, default=100, help="time freeze cycle")
    return parser.parse_args()
