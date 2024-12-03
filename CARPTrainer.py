import torch
from torch import nn
from logging import getLogger
import random
import numpy as np
import torch.nn.functional as F
from CARPEnv import CARPEnv as Env
from CARPModel import CARPModel as Model
from CARPTester import validate

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from tensorboard_logger import Logger as TbLogger

from utils import *

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class CARPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        self.saved_problems_uniform = []
        self.saved_problems_cluster = []
        self.saved_problems_smallworld = []
        self.is_problems_generated = False
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        if trainer_params['distillation']:
            self.student_model_params = trainer_params['student_model_param']
        self.saved_problems = None

        # log and result folder
        self.logger = getLogger(name='trainer')
        if self.trainer_params['logging']['tb_logger']:
            self.tb_logger = TbLogger('./log/' + get_start_time() + self.trainer_params['tb_path'])
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # CUDA setting
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # initialize model and envs
        if trainer_params['distillation'] and trainer_params['distill_param']['multi_teacher']: 
            model_uniform = Model(**self.model_params)
            model_cluster = Model(**self.model_params)
            model_smallworld = Model(**self.model_params)
            self.model = {'uniform': model_uniform,
                          'cluster': model_cluster,
                          'smallworld': model_smallworld}
            self.env = Env(**self.env_params)
        else:
            self.model = Model(**self.model_params)
            self.env = Env(**self.env_params)

        # enable distillation
        if trainer_params['distillation']: 
            self.student_model = Model(**self.student_model_params)
            self.student_env = Env(**self.env_params)
            # only train student model
            self.optimizer = Optimizer(self.student_model.parameters(), **self.optimizer_params['optimizer'])
        else:
            self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])

        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # restore model
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            if trainer_params['distillation']:
                checkpoint_fullname = model_load['path']
                checkpoint = torch.load(checkpoint_fullname, map_location=device)
                self.student_model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info('已加载保存的学生模型！')
            else:
                checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
                checkpoint = torch.load(checkpoint_fullname, map_location=device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info('已加载保存的模型！')

            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch'] - 1

        # initialize timeEstimator
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        self.save_flag = False
        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1):        
            self.logger.info('=================================================================')

            self.scheduler.step()

            # train
            if self.trainer_params['distillation']:
                if self.trainer_params['distill_param']['distill_distribution']:
                    train_score, train_loss, RL_loss, KLD_loss, class_type = self._distill_one_epoch(epoch, teacher_prob=self.trainer_params['distill_param']['teacher_prob'])
                    locals()[class_type + '_train_score'] = train_score
                    locals()[class_type + '_train_loss'] = train_loss
                    locals()[class_type + '_RL_loss'] = RL_loss
                    locals()[class_type + '_KLD_loss'] = KLD_loss
                    self.trainer_params['distill_param']['count'][class_type] += 1
                else:
                    train_score, train_loss, RL_loss, KLD_loss = self._distill_one_epoch(epoch)
            else:
                train_score, train_loss = self._train_one_epoch(epoch)

            # test
            if self.trainer_params['distillation'] and self.trainer_params['distill_param']['adaptive_prob']:
                if epoch == 1 or (epoch % self.trainer_params['distill_param']['adaptive_interval']) == 0:
                    is_test = True  
            elif self.trainer_params['multi_test'] and (epoch % 50) == 0:
                is_test = True  # test every 50 epochs
            else:
                is_test = False

            if is_test:
                if self.trainer_params['distillation']:
                    val_model, val_env = self.student_model, self.student_env
                else:
                    val_model, val_env = self.model, self.env
                val_no_aug, val_aug, gap_no_aug, gap_aug = [], [], [], []
                g = 0
                # calculate the gap with the MAENS solution
                for k, v in self.trainer_params['val_dataset_multi'].items():
                    no_aug, aug = validate(model=val_model, env=val_env,
                                           batch_size=self.trainer_params['val_batch_size'],
                                           augment=True, load_path=v)
                    val_no_aug.append(no_aug)
                    val_aug.append(aug)
                    gap_no_aug.append((no_aug - self.trainer_params['MAENS_optimal'][g]) / self.trainer_params['MAENS_optimal'][g])
                    gap_aug.append((aug - self.trainer_params['MAENS_optimal'][g]) / self.trainer_params['MAENS_optimal'][g])
                    g += 1

            # update adaptive probability
            if self.trainer_params['distillation'] and self.trainer_params['distill_param']['adaptive_prob']:
                gap = gap_aug if self.trainer_params['distill_param']['aug_gap'] else gap_no_aug
                if any(map(lambda x: x < 0, gap)):
                    self.trainer_params['distill_param']['teacher_prob'] = [1 / 3, 1 / 3, 1 / 3]
                    self.logger.info('发现负 gap，重置教师概率为均匀分布！')
                else:
                    if self.trainer_params['distill_param']['adaptive_prob_type'] == 'softmax':
                        self.trainer_params['distill_param']['teacher_prob'] = softmax(gap)
                    elif self.trainer_params['distill_param']['adaptive_prob_type'] == 'sum':
                        self.trainer_params['distill_param']['teacher_prob'] = [gap[i] / sum(gap) for i in range(len(gap))]

            # log
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)
            if self.trainer_params['distillation']:
                self.result_log.append('RL_loss', epoch, RL_loss)
                self.result_log.append('KLD_loss', epoch, KLD_loss)
                if self.trainer_params['distill_param']['distill_distribution']:
                    self.result_log.append(class_type + '_train_score', epoch, train_score)
                    self.result_log.append(class_type + '_train_loss', epoch, train_loss)
                    self.result_log.append(class_type + '_RL_loss', epoch, RL_loss)
                    self.result_log.append(class_type + '_KLD_loss', epoch, KLD_loss)
                    self.result_log.append('class_type', epoch, class_type)
            if is_test:
                note = ['uniform', 'cluster', ' smallworld']
                for i in range(3):
                    self.result_log.append(note[i] + '_val_score_noAUG', epoch, val_no_aug[i])
                    self.result_log.append(note[i] + '_val_score_AUG', epoch, val_aug[i])
                    self.result_log.append(note[i] + '_val_gap_noAUG', epoch, gap_no_aug[i])
                    self.result_log.append(note[i] + '_val_gap_AUG', epoch, gap_aug[i])
                self.result_log.append('val_gap_AUG_mean', epoch, np.mean(gap_aug))
                self.result_log.append('val_gap_noAUG_mean', epoch, np.mean(gap_no_aug))       

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))
            if self.trainer_params['logging']['tb_logger']:
                self.tb_logger.log_value('train_score', train_score, epoch)
                self.tb_logger.log_value('train_loss', train_loss, epoch)
                if self.trainer_params['distillation']:
                    self.tb_logger.log_value('RL_loss', RL_loss, epoch)
                    self.tb_logger.log_value('KLD_loss', KLD_loss, epoch)
                    if self.trainer_params['distill_param']['distill_distribution']:
                        self.tb_logger.log_value(class_type + '_train_score', train_score, self.trainer_params['distill_param']['count'][class_type])
                        self.tb_logger.log_value(class_type + '_train_loss', train_loss, self.trainer_params['distill_param']['count'][class_type])
                        self.tb_logger.log_value(class_type + '_RL_loss', RL_loss, self.trainer_params['distill_param']['count'][class_type])
                        self.tb_logger.log_value(class_type + '_KLD_loss', KLD_loss, self.trainer_params['distill_param']['count'][class_type])
                if is_test:
                    note = ['uniform', 'cluster', 'smallworld']
                    for i in range(3):
                        self.tb_logger.log_value(note[i] + '_val_score_noAUG', val_no_aug[i], epoch)
                        self.tb_logger.log_value(note[i] + '_val_score_AUG', val_aug[i], epoch)
                        self.tb_logger.log_value(note[i] + '_val_gap_noAUG', gap_no_aug[i], epoch)
                        self.tb_logger.log_value(note[i] + '_val_gap_AUG', gap_aug[i], epoch)
                    self.tb_logger.log_value('val_gap_AUG_mean', np.mean(gap_aug), epoch)
                    self.tb_logger.log_value('val_gap_noAUG_mean', np.mean(gap_no_aug), epoch)

            # save model when all done 
            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("保存训练模型")
                if self.trainer_params['distillation']:
                    checkpoint_dict = {
                        'epoch': epoch,
                        'model_state_dict': self.student_model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'result_log': self.result_log.get_raw_data()
                    }
                else:
                    checkpoint_dict = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'result_log': self.result_log.get_raw_data()
                    }

                if self.trainer_params['distillation'] and self.trainer_params['distill_param']['distill_distribution']:
                    torch.save(checkpoint_dict, '{}/checkpoint-{}-{}.pt'.format(self.result_folder, epoch, class_type))
                else:
                    torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))
            
            if is_test:
                note = ['uniform', 'cluster', 'smallworld']
                for i in range(3):
                    self.logger.info("Epoch {:3d}/{:3d} Validate in {}: Gap: noAUG[{:.3f}] AUG[{:.3f}]; Score: noAUG[{:.3f}] AUG[{:.3f}]".format(
                        epoch, self.trainer_params['epochs'],note[i], gap_no_aug[i], gap_aug[i],val_no_aug[i],val_aug[i]))
                self.logger.info("Epoch {:3d}/{:3d} Validate! mean Gap: noAUG[{:.3f}] AUG[{:.3f}]".format(epoch,
                        self.trainer_params['epochs'], np.mean(gap_no_aug)*100, np.mean(gap_aug)*100))
                if self.trainer_params['best']==0:
                    print(self.trainer_params['best'])
                    self.trainer_params['best'] = np.mean(gap_aug)*100
                elif  np.mean(gap_aug)*100 < self.trainer_params['best']:
                    self.trainer_params['best'] = np.mean(gap_aug) * 100
                    self.logger.info("Saving best trained_model")
                    if self.trainer_params['distillation']:
                        checkpoint_dict = {
                            'epoch': epoch,
                            'best_gap': np.mean(gap_aug) * 100,
                            'model_state_dict': self.student_model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'result_log': self.result_log.get_raw_data()
                        }
                    else:
                        checkpoint_dict = {
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'result_log': self.result_log.get_raw_data()
                        }

                    # save the best trained model
                    if self.trainer_params['distillation'] and self.trainer_params['distill_param']['distill_distribution']:
                        torch.save(checkpoint_dict,'{}/checkpoint-{}-best.pt'.format(self.result_folder,class_type))
                    else:
                        torch.save(checkpoint_dict, '{}/checkpoint-best.pt'.format(self.result_folder))
    '''
    def _train_one_epoch(self, epoch):
        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)
            avg_score, avg_loss = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)
            episode += batch_size

            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))

        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):
       
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        state, reward, done = self.env.pre_step()

        while not done:
            selected, prob = self.model(state)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

       
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        log_prob = prob_list.log().sum(dim=2)
        loss = -advantage * log_prob  
        loss_mean = loss.mean()
       
        max_pomo_reward, _ = reward.max(dim=1)
        score_mean = -max_pomo_reward.float().mean() 

        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return score_mean.item(), loss_mean.item()
    '''
    def _distill_one_epoch(self, epoch, teacher_prob=0):
        distill_param = self.trainer_params['distill_param']
        self.logger.info("开始训练学生模型 epoch {}".format(epoch))

        # initialization
        if distill_param['distill_distribution']:
            uniform_score_AM, uniform_loss_AM, uniform_RL_loss_AM, uniform_KLD_loss_AM, uniform_feature_loss_AM = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
            cluster_score_AM, cluster_loss_AM, cluster_RL_loss_AM, cluster_KLD_loss_AM, cluster_feature_loss_AM = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
            smallworld_score_AM, smallworld_loss_AM, smallworld_RL_loss_AM, smallworld_KLD_loss_AM, smallworld_feature_loss_AM = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        else:
            score_AM = AverageMeter()
            loss_AM = AverageMeter()
            RL_loss_AM = AverageMeter()
            KLD_loss_AM = AverageMeter()
            feature_loss_AM = AverageMeter()

        # load teacher model
        if distill_param['multi_teacher'] and distill_param['distill_distribution']:  # if use multiple teachers
            for i in ['uniform', 'cluster', 'smallworld']:
                load_path = self.trainer_params['model_load']['load_path_multi'][i]
                self.logger.info(' [*] 从 {} 加载模型'.format(load_path))
                checkpoint = torch.load(load_path, map_location=self.device)
                self.model[i].load_state_dict(checkpoint['model_state_dict'])
                if distill_param['adaptive_prob'] and epoch > distill_param['start_adaptive_epoch']:  
                    class_type = np.random.choice(['uniform', 'cluster', 'smallworld'], size=1, p=distill_param['teacher_prob'])
                else:
                    class_type = np.random.choice(['uniform', 'cluster', 'smallworld'], 1)

        elif distill_param['distill_distribution']:  # choosing teacher model via distribution
            if distill_param['adaptive_prob'] and epoch > distill_param['start_adaptive_epoch']:  
                class_type = np.random.choice(['uniform', 'cluster', 'smallworld'], size=1, p=distill_param['teacher_prob'])
                load_path = self.trainer_params['model_load']['load_path_multi'][class_type.item()]
                self.logger.info(' [*] 从 {} 加载模型，概率：{}'.format(load_path, distill_param['teacher_prob']))
            else:
                class_type = np.random.choice(['uniform', 'cluster', 'smallworld'], 1)
                load_path = self.trainer_params['model_load']['load_path_multi'][class_type.item()]
                self.logger.info(' [*] 从 {} 加载模型'.format(load_path))
            checkpoint = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        else:  
            load_path = self.trainer_params['model_load']['path']
            self.logger.info(' [*] 从 {} 加载模型'.format(load_path))
            checkpoint = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0       
        batch_index = 0         
        name = "node" + str(50) + "edge" + str(100) + "_" + "features"
        file_path = os.path.join("mapinfo", name)
        # Ensure the main file path exists
        os.makedirs(file_path, exist_ok=True)
        batch_counter = 0

        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            class_type = None if not distill_param['distill_distribution'] else class_type
            if not isinstance(class_type, str) and class_type is not None:
                class_type = class_type.item()

            if epoch == 1 and not self.is_problems_generated:
                files_exist = all(
                    os.path.exists(os.path.join(file_path, f"{filename}.txt"))
                    for filename in ["depot_features_uniform", "customer_features_uniform", "customer_demand_uniform", 
                                    "graph_info_uniform", "D_uniform", "A_uniform",
                                    "depot_features_cluster", "customer_features_cluster", "customer_demand_cluster", 
                                    "graph_info_cluster", "D_cluster", "A_cluster",
                                    "depot_features_smallworld", "customer_features_smallworld", "customer_demand_smallworld", 
                                    "graph_info_smallworld", "D_smallworld", "A_smallworld"]
                )
                        
                if files_exist:
                    # load while files exist
                    self.logger.info("检测到已存在的文件，直接加载")
                    self.is_problems_generated = True  
                    
                elif files_exist == False:
                    # generate instances in the first epoch             
                    for dist in ['uniform', 'cluster', 'smallworld']:
                        avg_score, avg_loss, RL_loss, KLD_loss = self._distill_one_batch(batch_size, distribution=dist) 
                        depot_features = self.env.reset_state.depot_features.cpu()
                        customer_features = self.env.reset_state.customer_features.cpu()
                        customer_demand = self.env.reset_state.customer_demand.cpu()
                        graph_info = self.env.reset_state.graph_info.cpu()
                        A = self.env.reset_state.A.cpu()
                        D = self.env.reset_state.D.cpu()
                        if dist == 'uniform':
                            self.saved_problems_uniform.append([depot_features, customer_features, customer_demand, graph_info, D, A])
                        elif dist == 'cluster':
                            self.saved_problems_cluster.append([depot_features, customer_features, customer_demand, graph_info, D, A])
                        elif dist == 'smallworld':
                            self.saved_problems_smallworld.append([depot_features, customer_features, customer_demand, graph_info, D, A])
                    batch_counter += 1  

                    # determine whether enough instances are generated
                    if batch_counter >= train_num_episode / batch_size:
                        os.makedirs(file_path, exist_ok=True)                       
                        distributions = ['uniform', 'cluster', 'smallworld']                       
                        for dist in distributions:
                            # get the corresponding distribution data
                            saved_problems = getattr(self, f'saved_problems_{dist}')                            
                            all_depot_features = []
                            all_customer_features = []
                            all_customer_demand = []
                            all_graph_info = []
                            all_D = []
                            all_A = []
                            for problem in saved_problems:
                                depot_features, customer_features, customer_demand, graph_info, D, A = problem
                                all_depot_features.append(depot_features)
                                all_customer_features.append(customer_features)
                                all_customer_demand.append(customer_demand)
                                all_graph_info.append(graph_info)
                                all_D.append(D)
                                all_A.append(A)

                            # save files
                            torch.save(all_depot_features, os.path.join(file_path, f'depot_features_{dist}.txt'))
                            torch.save(all_customer_features, os.path.join(file_path, f'customer_features_{dist}.txt'))
                            torch.save(all_customer_demand, os.path.join(file_path, f'customer_demand_{dist}.txt'))
                            torch.save(all_graph_info, os.path.join(file_path, f'graph_info_{dist}.txt'))
                            torch.save(all_D, os.path.join(file_path, f'D_{dist}.txt'))
                            torch.save(all_A, os.path.join(file_path, f'A_{dist}.txt'))
                            self.logger.info(f"{dist} 分布的实例成功保存")

            if epoch > 1 or (epoch == 1 and files_exist):                
                # Check if data needs to be reloaded
                if not hasattr(self, 'cached_data') or self.cached_data.get('class_type') != class_type:
                    self.cached_data = {'class_type': class_type}  # record the current distribution type

                    if class_type == 'uniform':                   
                        self.cached_data['depot_features'] = torch.load(os.path.join(file_path, 'depot_features_uniform.txt'))
                        self.cached_data['customer_features'] = torch.load(os.path.join(file_path, 'customer_features_uniform.txt'))
                        self.cached_data['customer_demand'] = torch.load(os.path.join(file_path, 'customer_demand_uniform.txt'))
                        self.cached_data['graph_info'] = torch.load(os.path.join(file_path, 'graph_info_uniform.txt'))
                        self.cached_data['D'] = torch.load(os.path.join(file_path, 'D_uniform.txt'))
                        self.cached_data['A'] = torch.load(os.path.join(file_path, 'A_uniform.txt'))
                    elif class_type == 'cluster':                   
                        self.cached_data['depot_features'] = torch.load(os.path.join(file_path, 'depot_features_cluster.txt'))
                        self.cached_data['customer_features'] = torch.load(os.path.join(file_path, 'customer_features_cluster.txt'))
                        self.cached_data['customer_demand'] = torch.load(os.path.join(file_path, 'customer_demand_cluster.txt'))
                        self.cached_data['graph_info'] = torch.load(os.path.join(file_path, 'graph_info_cluster.txt'))
                        self.cached_data['D'] = torch.load(os.path.join(file_path, 'D_cluster.txt'))
                        self.cached_data['A'] = torch.load(os.path.join(file_path, 'A_cluster.txt'))
                    elif class_type == 'smallworld':                    
                        self.cached_data['depot_features'] = torch.load(os.path.join(file_path, 'depot_features_smallworld.txt'))
                        self.cached_data['customer_features'] = torch.load(os.path.join(file_path, 'customer_features_smallworld.txt'))
                        self.cached_data['customer_demand'] = torch.load(os.path.join(file_path, 'customer_demand_smallworld.txt'))
                        self.cached_data['graph_info'] = torch.load(os.path.join(file_path, 'graph_info_smallworld.txt'))
                        self.cached_data['D'] = torch.load(os.path.join(file_path, 'D_smallworld.txt'))
                        self.cached_data['A'] = torch.load(os.path.join(file_path, 'A_smallworld.txt'))

                # load data from cache
                depot_features = self.cached_data['depot_features'][batch_index].to(self.device)
                customer_features = self.cached_data['customer_features'][batch_index].to(self.device)
                customer_demand = self.cached_data['customer_demand'][batch_index].to(self.device)
                graph_info = self.cached_data['graph_info'][batch_index].to(self.device)
                D = self.cached_data['D'][batch_index].to(self.device)
                A = self.cached_data['A'][batch_index].to(self.device)
                avg_score, avg_loss, RL_loss, KLD_loss, feature_loss = self._distill_one_batch(batch_size, distribution=class_type, use_saved_problems=True, saved_problems=[depot_features, customer_features, customer_demand, graph_info, D, A])

            # upadate variables
            if distill_param['distill_distribution']:
                locals()[class_type + '_score_AM'].update(avg_score, batch_size)
                locals()[class_type + '_loss_AM'].update(avg_loss, batch_size)
                locals()[class_type + '_RL_loss_AM'].update(RL_loss, batch_size)
                locals()[class_type + '_KLD_loss_AM'].update(KLD_loss, batch_size)
                locals()[class_type + '_feature_loss_AM'].update(feature_loss, batch_size)

            else:
                score_AM.update(avg_score, batch_size)
                loss_AM.update(avg_loss, batch_size)
                RL_loss_AM.update(RL_loss, batch_size)
                KLD_loss_AM.update(KLD_loss, batch_size)
                feature_loss.update(feature_loss, batch_size)

            episode += batch_size
            # update batch index
            batch_index += 1  

           # print information for every batch          
            if distill_param['distill_distribution']:
                print('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f},  RL_Loss: {:.4f},  KLD_Loss: {:.4f},  feature_loss: {:.4f}'
                                    .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                            locals()[class_type + '_score_AM'].avg, locals()[class_type + '_loss_AM'].avg,
                                            locals()[class_type + '_RL_loss_AM'].avg, locals()[class_type + '_KLD_loss_AM'].avg, locals()[class_type + '_feature_loss_AM'].avg))

            else:
                print('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f},  RL_Loss: {:.4f},  KLD_Loss: {:.4f}'
                                    .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                            score_AM.avg, loss_AM.avg, RL_loss_AM.avg, KLD_loss_AM.avg, feature_loss_AM))        
                
        torch.cuda.empty_cache()

        # record log every epoch
        if distill_param['distill_distribution']:
            self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f},  RL_Loss: {:.4f},  KLD_Loss: {:.4f},  feature_loss: {:.4f}'
                            .format(epoch, 100. * episode / train_num_episode, locals()[class_type + '_score_AM'].avg,
                                    locals()[class_type + '_loss_AM'].avg, locals()[class_type + '_RL_loss_AM'].avg, locals()[class_type + '_KLD_loss_AM'].avg), locals()[class_type + '_feature_loss_AM'].avg)
            torch.cuda.empty_cache()
            return locals()[class_type + '_score_AM'].avg, locals()[class_type + '_loss_AM'].avg, locals()[class_type + '_RL_loss_AM'].avg, locals()[class_type + '_KLD_loss_AM'].avg, locals()[class_type + '_feature_loss_AM'].avg, class_type
        
        else:
            self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f},  RL_Loss: {:.4f},  KLD_Loss: {:.4f}'
                            .format(epoch, 100. * episode / train_num_episode, score_AM.avg, loss_AM.avg, RL_loss_AM.avg, KLD_loss_AM.avg, feature_loss_AM))
            torch.cuda.empty_cache()
            
        if epoch == 1 and not self.is_problems_generated:
            self.is_problems_generated = True
            return score_AM.avg, loss_AM.avg, RL_loss_AM.avg, KLD_loss_AM.avg, feature_loss_AM

    def _distill_one_batch(self, batch_size, distribution=None, use_saved_problems=False, saved_problems=None):
        distill_param = self.trainer_params['distill_param']
        self.student_model.train()

        if distill_param['multi_teacher'] and distill_param['distill_distribution']:
            for i in ['uniform', 'cluster', 'smallworld']:
                self.model[i].eval()
        else:
            self.model.eval()

        # load problem instances
        if use_saved_problems and saved_problems is not None:
            # use saved problems
            depot_features, customer_features, customer_demand, graph_info, D, A = saved_problems
            # load problems into environments
            self.env.load_problems(batch_size, copy=[depot_features, customer_features, customer_demand, graph_info, D, A])
            self.student_env.load_problems(batch_size, copy=[depot_features, customer_features, customer_demand, graph_info, D, A])

        else:
            # generate new problems
            self.env.load_problems(batch_size, distribution=distribution)
            # ensure student_env uses the same problems
            depot_features = self.env.reset_state.depot_features
            customer_features = self.env.reset_state.customer_features
            customer_demand = self.env.reset_state.customer_demand
            graph_info = self.env.reset_state.graph_info
            A = self.env.reset_state.A
            D = self.env.reset_state.D
            self.student_env.load_problems(batch_size, copy=[depot_features, customer_features, customer_demand, graph_info, D, A])

        if distill_param['router'] == 'teacher':
            with torch.no_grad():
                reset_state, _, _ = self.env.reset()
                self.model.pre_forward(reset_state, attn_type=None)  # No return!
                teacher_prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))  # shape: (batch, pomo, 0~edge_size)
                state, reward, done = self.env.pre_step()
                teacher_pi = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
                teacher_probs = torch.zeros(size=(batch_size, self.env.pomo_size, self.env.edge_size+1 , 0))
                # decode
                while not done:
                    selected, prob, probs = self.model(state, return_probs=True, teacher=True)
                    state, reward, done = self.env.step(selected)
                    teacher_prob_list = torch.cat((teacher_prob_list, prob[:, :, None]), dim=2)
                    teacher_pi = torch.cat((teacher_pi, selected[:, :, None]), dim=2)
                    teacher_probs = torch.cat((teacher_probs, probs[:, :, :, None]), dim=3)
                teacher_probs = teacher_probs + 0.00001  # avoid log0

            student_reset_state, _, _ = self.student_env.reset()
            # encoding process
            self.student_model.pre_forward(student_reset_state, attn_type=None)  # No return!
            student_prob_list = torch.zeros(size=(batch_size, self.student_env.pomo_size, 0))  # shape: (batch, pomo, 0~edge_size)
            student_state, student_reward, student_done = self.student_env.pre_step()
            student_pi = torch.zeros(size=(batch_size, self.student_env.pomo_size, 0))
            student_probs = torch.zeros(size=(batch_size, self.student_env.pomo_size, self.student_env.edge_size+1, 0))

            # decoding process
            while not student_done:
                student_selected, student_prob, probs = self.student_model(student_state, route=teacher_pi, return_probs=True)
                student_state, student_reward, student_done = self.student_env.step(student_selected)
                student_prob_list = torch.cat((student_prob_list, student_prob[:, :, None]), dim=2)
                student_pi = torch.cat((student_pi, student_selected[:, :, None]), dim=2)
                student_probs = torch.cat((student_probs, probs[:, :, :, None]), dim=3)
            student_probs = student_probs + 0.00001  # avoid log0
        
        # student model is the router
        else: 
            # use multiple teachers
            if self.trainer_params['distill_param']['multi_teacher']:
                student_reset_state, _, _ = self.student_env.reset()
                # encoding process
                self.student_model.pre_forward(student_reset_state, attn_type=None)  # No return!
                student_prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))  # shape: (batch, pomo, 0~edge_size)
                student_state, student_reward, student_done = self.student_env.pre_step()
                student_pi = torch.zeros(size=(batch_size, self.student_env.pomo_size, 0))
                student_probs = torch.zeros(size=(batch_size, self.student_env.pomo_size, self.student_env.edge_size+1, 0))
                # decoding process
                while not student_done:
                    student_selected, student_prob, probs = self.student_model(student_state, return_probs=True)
                    student_state, student_reward, student_done = self.student_env.step(student_selected)
                    student_prob_list = torch.cat((student_prob_list, student_prob[:, :, None]), dim=2)
                    student_pi = torch.cat((student_pi, student_selected[:, :, None]), dim=2)
                    student_probs = torch.cat((student_probs, probs[:, :, :, None]), dim=3)
                student_probs = student_probs + 0.00001  # avoid log0
               
                teacher_probs_multi = []
                for i in ['uniform', 'cluster', 'smallworld']:
                    with torch.no_grad():
                        reset_state, _, _ = self.env.reset()
                        # encoding process
                        self.model[i].pre_forward(reset_state, attn_type=None)  # No return!
                        teacher_prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))  # shape: (batch, pomo, 0~edge_size)
                        state, reward, done = self.env.pre_step()
                        teacher_pi = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
                        teacher_probs = torch.zeros(size=(batch_size, self.env.pomo_size, self.env.edge_size+1, 0))
                        # decoding process
                        while not done:
                            selected, prob, probs = self.model[i](state, route=student_pi, return_probs=True)
                            state, reward, done = self.env.step(selected)
                            teacher_prob_list = torch.cat((teacher_prob_list, prob[:, :, None]), dim=2)
                            teacher_pi = torch.cat((teacher_pi, selected[:, :, None]), dim=2)
                            teacher_probs = torch.cat((teacher_probs, probs[:, :, :, None]), dim=3)
                        teacher_probs = teacher_probs + 0.00001  
                        teacher_probs_multi.append(teacher_probs)

            # use single teacher
            else:
                # 获取教师模型的中间特征
                with torch.no_grad():
                    teacher_reset_state, _, _ = self.env.reset()
                    teacher_encoded_features, teacher_intermediate_features = self.model.encoder(
                        teacher_reset_state.depot_features,
                        teacher_reset_state.customer_features,
                        teacher_reset_state.customer_demand,
                        teacher_reset_state.A,
                        ReturnInnerFeatures=True
                    )

                # 获取学生模型的中间特征
                student_reset_state, _, _ = self.student_env.reset()
                student_encoded_features, student_intermediate_features = self.student_model.encoder(
                    student_reset_state.depot_features,
                    student_reset_state.customer_features,
                    student_reset_state.customer_demand,
                    student_reset_state.A,
                    ReturnInnerFeatures=True
                )

                # 计算中间特征蒸馏损失
                feature_distill_loss = 0.0
                for teacher_feature, student_feature in zip(teacher_intermediate_features, student_intermediate_features):
                    feature_distill_loss += F.mse_loss(student_feature, teacher_feature)
                feature_distill_loss /= len(teacher_intermediate_features)
            
                student_reset_state, _, _ = self.student_env.reset()               
                self.student_model.pre_forward(student_reset_state, attn_type=None)  # No return!
                student_prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))  # shape: (batch, pomo, 0~edge_size)
                student_state, student_reward, student_done = self.student_env.pre_step()
                student_pi = torch.zeros(size=(batch_size, self.student_env.pomo_size, 0))
                student_probs = torch.zeros(size=(batch_size, self.student_env.pomo_size, self.student_env.edge_size+1, 0))               
                while not student_done:
                    # * 根据当前 student_state 环境状态选择一个动作
                    # * student_selected：返回的动作选择，形状为 (batch_size, pomo_size)
                    # * student_prob：记录 student_selected 中每个动作对应的概率，形状为 (batch_size, pomo_size)
                    # * probs：每个 POMO 对所有可能动作的概率分布，形状为 (batch_size, pomo_size, edge_size + 1)
                    student_selected, student_prob, probs = self.student_model(student_state, return_probs=True)
                    # 根据学生模型选择的动作（student_selected），更新学生环境的状态
                    student_state, student_reward, student_done = self.student_env.step(student_selected)
                    # 将当前的动作概率（student_prob）附加到 student_prob_list 中
                    student_prob_list = torch.cat((student_prob_list, student_prob[:, :, None]), dim=2)
                    # 将当前选择的动作（student_selected）附加到 student_pi 中
                    student_pi = torch.cat((student_pi, student_selected[:, :, None]), dim=2)
                    # 将每个时间步的对所有可能动作的完整概率分布（probs）附加到 student_probs 张量中
                    student_probs = torch.cat((student_probs, probs[:, :, :, None]), dim=3)
                student_probs = student_probs + 0.00001 

                # the teacher follows the student path
                with torch.no_grad():
                    reset_state, _, _ = self.env.reset()
                    self.model.pre_forward(reset_state, attn_type=None)  # No return!
                    teacher_prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))  # shape: (batch, pomo, 0~edge_size)
                    state, reward, done = self.env.pre_step()
                    teacher_pi = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
                    teacher_probs = torch.zeros(size=(batch_size, self.env.pomo_size, self.env.edge_size+1, 0))                   
                    while not done:
                        selected, prob, probs = self.model(state, route=student_pi, return_probs=True)
                        state, reward, done = self.env.step(selected)
                        teacher_prob_list = torch.cat((teacher_prob_list, prob[:, :, None]), dim=2)
                        teacher_pi = torch.cat((teacher_pi, selected[:, :, None]), dim=2)
                        teacher_probs = torch.cat((teacher_probs, probs[:, :, :, None]), dim=3)
                    teacher_probs = teacher_probs + 0.00001  

        assert torch.equal(teacher_pi, student_pi), "教师路径和学生路径不相同！"

        # the loss of the student model
        ###############################################
        # advantage function
        advantage = student_reward - student_reward.float().mean(dim=1, keepdims=True)
        log_prob = student_prob_list.log().sum(dim=2)
        # REINFORCE loss
        task_loss = -advantage * log_prob
        task_loss = task_loss.mean()

        # soft loss (KL)
        if distill_param['meaningful_KLD']:
            if distill_param['multi_teacher']:
                for i in range(len(teacher_probs_multi)):
                    if i == 0:
                        soft_loss = (student_probs * (student_probs.log() - teacher_probs_multi[i].log())).sum(dim=2).mean() if distill_param['KLD_student_to_teacher'] \
                            else (teacher_probs_multi[i] * (teacher_probs_multi[i].log() - student_probs.log())).sum(dim=2).mean()
                    else:
                        soft_loss = soft_loss + (student_probs * (student_probs.log() - teacher_probs_multi[i].log())).sum(dim=2).mean() if distill_param['KLD_student_to_teacher'] \
                            else (teacher_probs_multi[i] * (teacher_probs_multi[i].log() - student_probs.log())).sum(dim=2).mean()
                soft_loss = soft_loss / 3
            else:
                soft_loss = (student_probs * (student_probs.log() - teacher_probs.log())).sum(dim=2).mean() if distill_param['KLD_student_to_teacher'] \
                    else (teacher_probs * (teacher_probs.log() - student_probs.log())).sum(dim=2).mean()              
        else:
            soft_loss = nn.KLDivLoss()(student_probs.log(), teacher_probs) if not distill_param['KLD_student_to_teacher'] \
                else nn.KLDivLoss()(teacher_probs.log(), student_probs)
        loss = task_loss * distill_param['rl_alpha'] + soft_loss * distill_param['distill_alpha'] + feature_distill_loss * 0.5

        # score
        max_pomo_reward, _ = student_reward.max(dim=1)  
        score_mean = -max_pomo_reward.float().mean()  

        self.student_model.zero_grad()
        loss.backward()
        self.optimizer.step()
        torch.cuda.empty_cache()

        return score_mean.item(), loss.item(), task_loss.item(), soft_loss.item(), feature_distill_loss.item()
    

from torch.cuda.amp import autocast        
def sinkhorn_wasserstein_distance(student_probs, teacher_probs, reg=0.1, max_iter=50, eps=1e-6):
    """
    使用 Sinkhorn 算法计算 Wasserstein 距离，支持批量计算，运行在 GPU 上，优化显存使用。
    """
    batch_size, pomo_size, edge_size_plus_one, num_steps = student_probs.size()
    device = student_probs.device

    # 定义固定的成本矩阵 M 和核矩阵 K，形状为 (edge_size_plus_one, edge_size_plus_one)
    x = torch.arange(edge_size_plus_one, dtype=torch.float32, device=device).unsqueeze(1)
    y = torch.arange(edge_size_plus_one, dtype=torch.float32, device=device).unsqueeze(0)
    M = (x - y).pow(2)  # 使用平方距离
    M /= M.max()

    K = torch.exp(-M / reg) + eps  # 添加一个小的 epsilon 防止数值问题
    K_T = K.t()  # 预计算 K 的转置以供后续使用

    K_M = K * M  # 预计算 K 与 M 的元素乘积

    # 确保分布归一化
    eps1 = 1e-8
    student_probs = student_probs / (student_probs.sum(dim=2, keepdim=True) + eps1)
    teacher_probs = teacher_probs / (teacher_probs.sum(dim=2, keepdim=True) + eps1)

    # 将张量重塑为 (batch_size * pomo_size, edge_size_plus_one, num_steps)
    batch_size_pomo = batch_size * pomo_size
    student_probs = student_probs.view(batch_size_pomo, edge_size_plus_one, num_steps)
    teacher_probs = teacher_probs.view(batch_size_pomo, edge_size_plus_one, num_steps)

    total_loss = 0.0

    for t in range(num_steps):
        # 获取当前时间步的概率分布，形状为 (batch_size_pomo, edge_size_plus_one)
        a = student_probs[:, :, t]
        b = teacher_probs[:, :, t]

        # 初始化 u 和 v，形状为 (batch_size_pomo, edge_size_plus_one)
        u = torch.ones_like(a, device=device)
        v = torch.ones_like(b, device=device)

        # Sinkhorn 迭代
        with autocast():
            for _ in range(max_iter):
                u_prev = u
                v_prev = v
    
                # 计算 K 与 v 的乘积
                K_v = torch.matmul(v, K_T)  # (batch_size_pomo, edge_size_plus_one)
                u = a / (K_v + eps)
    
                # 计算 K^T 与 u 的乘积
                K_t_u = torch.matmul(u, K)  # (batch_size_pomo, edge_size_plus_one)
                v = b / (K_t_u + eps)
    
                # 检查收敛性
                if torch.max(torch.abs(u - u_prev)) < eps and torch.max(torch.abs(v - v_prev)) < eps:
                    break

        # 计算 Wasserstein 距离，无需显式构建传输计划 T
        K_M_v = torch.matmul(v, K_M.T)  # (batch_size_pomo, edge_size_plus_one)
        loss = torch.sum(u * K_M_v, dim=1)  # 对 edge_size_plus_one 维求和

        total_loss += torch.mean(loss)

    # 计算所有时间步的平均 Wasserstein 距离
    avg_loss = total_loss / num_steps

    return avg_loss

def estimate_sigma(data, min_sigma=1e-5):
    """
    根据数据自动估计一个合适的 sigma 值，基于 L2 距离的标准差。
    
    参数：
    - data: 数据，形状为 (n_samples, feature_dim)
    - min_sigma: sigma 的最小值，避免为零
    
    返回：
    - sigma: 根据数据估算的 sigma 值
    """
    dist_matrix = torch.cdist(data, data, p=2)  # 计算所有点对之间的欧几里得距离
    sigma = torch.std(dist_matrix)  # 返回距离矩阵的标准差作为 sigma 的估计值
    # 确保 sigma 大于最小值
    return max(sigma.item(), min_sigma)

def rbf_kernel(x, y, sigma):
    """
    计算高斯核函数（RBF 核）：计算两个分布 x 和 y 之间的核函数值

    参数：
    - x, y: 输入的两个张量，形状为 (n_samples, feature_dim)
    - sigma: 高斯核的标准差，控制核的宽度

    返回：
    - kernel: 计算得到的高斯核矩阵
    """
    dist = torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, dim=-1)  # 计算欧几里得距离
    dist = torch.clamp(dist, max=1e6)  # 限制距离的最大值，防止溢出
    return torch.exp(-dist / (2 * sigma ** 2))  # 计算高斯核

def mmd_loss(student_probs, teacher_probs, sigma=None):
    """
    使用 MMD 计算学生和教师模型输出分布的差异。
    采用高斯核来计算 MMD。

    参数：
    - student_probs: 学生模型的概率分布，形状为 (batch_size, pomo_size, edge_size_plus_one, num_steps)
    - teacher_probs: 教师模型的概率分布，形状为 (batch_size, pomo_size, edge_size_plus_one, num_steps)
    - sigma: 高斯核的标准差，控制核的宽度，默认为 None 使用自适应估计

    返回：
    - MMD 损失值
    """
    batch_size, pomo_size, edge_size_plus_one, num_steps = student_probs.size()

    # 将 student_probs 和 teacher_probs 重塑为 (batch_size * pomo_size, edge_size_plus_one, num_steps)
    student_probs = student_probs.view(-1, edge_size_plus_one, num_steps)
    teacher_probs = teacher_probs.view(-1, edge_size_plus_one, num_steps)

    # 清理无效值 (NaN, Inf)
    student_probs = torch.nan_to_num(student_probs, nan=0.0, posinf=1.0, neginf=0.0)
    teacher_probs = torch.nan_to_num(teacher_probs, nan=0.0, posinf=1.0, neginf=0.0)

    # 计算 MMD 损失
    mmd_loss = 0.0
    for t in range(num_steps):
        student_step = student_probs[:, :, t]
        teacher_step = teacher_probs[:, :, t]

        # 如果没有给定 sigma，则根据数据自适应计算 sigma
        if sigma is None:
            sigma = estimate_sigma(torch.cat([student_step, teacher_step], dim=0))  # 联合计算学生和教师的sigma

        # 计算 MMD 的不同项
        xx = rbf_kernel(student_step, student_step, sigma)  # 学生分布内的 MMD
        yy = rbf_kernel(teacher_step, teacher_step, sigma)  # 教师分布内的 MMD
        xy = rbf_kernel(student_step, teacher_step, sigma)  # 学生和教师分布间的 MMD

        # 将高斯核的计算结果限制在合适范围内
        xx = torch.clamp(xx, min=0.0, max=1.0)
        yy = torch.clamp(yy, min=0.0, max=1.0)
        xy = torch.clamp(xy, min=0.0, max=1.0)

        # 计算每一层的 MMD 损失
        mmd_loss += torch.mean(xx + yy - 2 * xy)

    # 返回平均 MMD 损失
    return mmd_loss / num_steps