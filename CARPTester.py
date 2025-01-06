import torch
import numpy as np
from logging import getLogger
from CARPEnv import CARPEnv as Env
from CARPModel import CARPModel as Model
from utils import *


class CARPTester:
    def __init__(self, env_params, model_params, tester_params):
        # 保存参数
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # 日志和结果文件夹
        self.logger = getLogger(name='tester')
        self.result_folder = get_result_folder()

        # cuda 设置
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # 初始化环境和模型
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # 恢复模型
        model_load = tester_params['model_load']
        if '.pt' in model_load['path']:
            checkpoint_fullname = model_load['path']
        else:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        print(checkpoint_fullname)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 初始化时间估算器
        self.time_estimator = TimeEstimator()

    def run(self):
        ##########################################################################################
        self.time_estimator.reset()

        score_AM = AverageMeter()

        if self.tester_params['test_data_load']['enable']:
            print(self.tester_params['test_data_load']['filename'])
            self.env.use_saved_problems(self.tester_params['test_data_load']['filename'], self.device)

        test_num_episode = self.tester_params['test_episodes']  # e.g., 1000
        episode = 0

        inferTime = []
        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            import time
            tik = time.time()

            score = self._test_one_batch(batch_size, episode)

            torch.cuda.synchronize()
            tok = time.time()
            inferTime.append(tok - tik)

            score_AM.update(score, batch_size)

            episode += batch_size

            ############################
            # 日志记录
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info(
                "Episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}".format(
                    episode, test_num_episode, elapsed_time_str, remain_time_str, score
                )
            )

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** 测试完成 *** ")
                self.logger.info(" 得分: {} ".format(score_AM.avg))

        return score_AM.avg, np.mean(inferTime)

    def _test_one_batch(self, batch_size, episode=None):
        # 准备模型
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, load_path=self.env_params['load_path'], episode=episode)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state, attn_type='qk_scaled')

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            state, reward, done = self.env.step(selected)

        # 返回结果
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # 获取 POMO 中的最好结果
        no_aug_score = -max_pomo_reward.float().mean()  # 负号表示最小化距离

        return no_aug_score.item()


def validate(model, env, batch_size, load_path=None):
    # 准备模型
    ###############################################
    model.eval()

    episode = 0
    test_num_episode = 1000
    score_list = []

    while episode < test_num_episode:
        remaining = test_num_episode - episode
        batch_size = min(batch_size, remaining)

        with torch.no_grad():
            env.load_problems(batch_size, load_path=load_path, episode=episode)
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            state, reward, done = env.step(selected)

        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # 获取 POMO 中的最好结果
        score = -max_pomo_reward.float().mean()  # 负号表示最小化距离
        score_list.append(score.item())
        episode += batch_size

    import numpy as np
    return np.mean(score_list)
