import config
import random, torch
import numpy as np
import datetime
import logging
import os, sys
from modules.D3QN import D3QN
from torch.utils.tensorboard import SummaryWriter
from modules.extract_modules import State, TDNet

sys.path.append(os.path.dirname(__file__)+"/../")
from env.env import BS_TD
import pdb
import pickle


def inference(env, agent, inference_ues, start_time, end_time, best_tp):
    score = 0
    return score

def train(args):
    # 环境初始化 - 这一部分初始化环境和agent
    env = BS_TD(logger=logger,
                K=args.nUE,
                seed=args.seed,
               )
    agent = D3QN(args.device, net_class=TDNet, action_space=list(range(len(env.action_space))), memory_size=args.memory_size, max_epsilon=args.max_epsilon, min_epsilon=args.min_epsilon,
                 lr=args.lr, epsilon_decay=args.epsilon_decay, batch_size=args.batch_size, gamma=args.gamma)
    logger("NetWork: %s", agent.net)

    nUE = args.nUE # 每一批次的UE数
    UE = ue_pool = list(range(nUE))
    # Tensorboard配置
    cell_writer = SummaryWriter(f'{os.path.dirname(__file__)}/logs/{args.daytime}/Cell/')
    writers = []
    for ue in ue_pool:
        writers.append(SummaryWriter(f'{os.path.dirname(__file__)}/logs/{args.daytime}/UE{ue}/'))
    # pdb.set_trace()
    state = State(args.device, *env.get_state_batch(batch=nUE, action_id=10))

    # 推理设定 - 这一部分保存420个UE在0-200时间内的最佳TP以及随机动作得到的TP，便于后面计算分数
    max_score = 0 # 记录最大分数，每次超过最大分数就保存一次模型
    
    # 开始训练
    frame_idx = 0
    loss = 0
    time = 0
    loss_s = []    
    reward_s = []    
    ue_c_tp_s = [[],[],[],[],[]]
    ue_acc_s = [[],[],[],[],[]]
    eff_tp_s = []
    bl_eff_tp_s = []
    while frame_idx < args.num_frame:
        action = agent.select_action(state)
        next_state = State(args.device, *env.get_state_batch(batch=nUE, action_id=action))
        reward = env.reward
        done = frame_idx % args.env_max_time == 0
        agent.memory_pool.add_memory(state, reward, action, next_state, done)
        state = next_state
        frame_idx += 1

        if frame_idx % args.optim_frame == 0: # 反向传播更新梯度
            loss = agent.optim()
            if loss is not None:
                opt_y_value, opt_rate, opt_ee = env.TD_allocate()
                cell_writer.add_scalar("train/loss", loss, frame_idx)
                cell_writer.add_scalar("train/reward", reward, frame_idx)
                cell_writer.add_scalar("train/baseline_reward", opt_rate/100.0, frame_idx)
                cell_writer.add_scalar("train/baseline_tp", opt_rate, frame_idx)
                if frame_idx % 10 == 0:
                    loss_s.append(loss)
                    reward_s.append(reward)
                    eff_tp_s.append(reward*100)
                    bl_eff_tp_s.append(opt_rate)
                for i in range(len(UE)):
                    c_rate = env.communication_rate()
                    acc = env.sensing_accuracy()
                    writers[i].add_scalar("UE actions", action, frame_idx)
                    writers[i].add_scalar("TP", c_rate[i], frame_idx)
                    writers[i].add_scalar("ACC", acc[:,i], frame_idx)
                    if frame_idx % 10 == 0:
                        ue_c_tp_s[i].append(c_rate[i])
                        ue_acc_s[i].append(acc[:,i])
                    # pdb.set_trace()
                agent.update_epsilon()
                
        if frame_idx % args.target_update_frame == 0: # 更新target net
            agent.update_target_net()
        score = env.reward            
        if frame_idx % args.train_log_cycle == 0:
            print(f"\rCurrent Max Score: {max_score} Frame idx: {frame_idx}/{args.num_frame}"+ " "*10, end="")
        # 保存模型
            if score > max_score:
                torch.save(agent.net.state_dict(), f'{os.path.dirname(__file__)}/models/{args.daytime}_frame{frame_idx}_score{score}.mo')
                max_score = score
        agent.train()
    data = (loss_s,reward_s,ue_c_tp_s,ue_acc_s,eff_tp_s,bl_eff_tp_s)
    with open('data.pickle', 'wb') as f:
        pickle.dump(data, f)
            
def train_env_init(args):
    # 设定随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 初始化Log文件
    args.daytime = datetime.datetime.now().strftime('TD_%Y-%m-%d-%H-%M-%S')
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        filename=f'{os.path.dirname(__file__)}/logs/{args.daytime}.log')
    logger = logging.getLogger(__name__)
    print(args.desc)
    logger.info("desc: %s", args.desc)
    logger.info("parameters: %s", args)
    return logger.info

if __name__ == "__main__":
    args = config.parse_args()
    if args.desc == "No desc":
        print("Note: This train doesn't have description.")
    logger = train_env_init(args)
    train(args)
