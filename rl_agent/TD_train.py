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
    """ 使用模型推理，获得评分
    Args:
        env (_type_): 环境实例
        agent (_type_): agent实例
        inference_ues (list): 用于推理的UE
        start_time (_type_): 开始时间
        end_time (_type_): 结束时间
        best_tp (_type_): 最好的TP
        tp_rand (_type_): 随机的TP
    """
    time_length = end_time - start_time
    TPs = np.zeros((len(inference_ues),time_length))
    inference_action = np.zeros(len(inference_ues), dtype=np.int32)
    for i in range(0,time_length):
        times = np.ones_like(inference_ues)*(i+start_time)
        data = State("cuda", *env.get_state_batch(batch=len(inference_ues), time=times, ue=inference_ues, action=inference_action))
        result = agent.net(data)
        selected_action = list(torch.argmax(result, dim=-1).cpu().numpy())
        inference_action = selected_action
        env.test = True
        [_,PMI,CQI,RI,TP] = env.get_state_batch(batch=len(inference_ues), time=times, ue=inference_ues, action=inference_action)
        env.test = False
        TPs[:,i]= env.TP
    score = ((np.average(TPs))/np.average(best_tp) - 0.6)/0.4
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
    inference_ues = list(range(420))
    start_time = 0
    end_time = 200
    best_data = None
    max_score = 0 # 记录最大分数，每次超过最大分数就保存一次模型
    
    # 开始训练
    frame_idx = 0
    loss = 0
    time = 0
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
                cell_writer.add_scalar("train/loss", loss, frame_idx)
                cell_writer.add_scalar("train/loss", loss, frame_idx)
                for i in range(len(UE)):
                    c_rate = env.communication_rate()
                    acc = env.sensing_accuracy()
                    writers[i].add_scalar("UE actions", action, frame_idx)
                    writers[i].add_scalar("TP", c_rate[i], frame_idx)
                    writers[i].add_scalar("ACC", acc[:,i], frame_idx)
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
            
def train_env_init(args):
    # 设定随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 初始化Log文件
    args.daytime = datetime.datetime.now().strftime('BM_%Y-%m-%d-%H-%M-%S')
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
