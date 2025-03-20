import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.battery_loader import get_battery_loader
from utils.logger import Logger
from tqdm import tqdm


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

# Early stopping Machanism
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型
        try:
            torch.save(model.state_dict(), path)
            self.val_loss_min = val_loss
            self.val_loss2_min = val_loss2
        except Exception as e:
            print(f"Error saving model to {path}: {e}")
            # 尝试使用临时目录
            temp_path = os.path.join(os.path.expanduser("~"), "temp_models")
            os.makedirs(temp_path, exist_ok=True)
            backup_path = os.path.join(temp_path, str(self.dataset) + '_checkpoint.pth')
            print(f"Attempting to save to backup location: {backup_path}")
            torch.save(model.state_dict(), backup_path)


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        # 设置默认值
        self.k = config.get('k', 3.0)  # 如果没有提供k，默认值为3.0
        
        self.__dict__.update(Solver.DEFAULTS, **config)
        
        # Create logs directory if it doesn't exist
        if not os.path.exists('./logs'):
            os.makedirs('./logs')
        
        # Initialize TensorFlow logger
        self.logger = Logger('./logs/' + self.dataset)

        # 获取target_id参数，如果不存在则默认为None
        target_id = config.get('target_id', None)  # 改为 target_id
        
        # Only the first loader prints info, others are silent
        self.train_loader = get_battery_loader(
            self.data_path, 
            batch_size=self.batch_size, 
            win_size=self.win_size,
            mode='train', 
            target_id=target_id,  # 改为 target_id
            silent=False
        )
        
        self.vali_loader = get_battery_loader(
            self.data_path, 
            batch_size=self.batch_size, 
            win_size=self.win_size,
            mode='val', 
            target_id=target_id,  # 改为 target_id
            silent=True
        )
        
        self.test_loader = get_battery_loader(
            self.data_path, 
            batch_size=self.batch_size, 
            win_size=self.win_size,
            mode='test', 
            target_id=target_id,  # 改为 target_id
            silent=True
        )
        
        self.thre_loader = get_battery_loader(
            self.data_path, 
            batch_size=self.batch_size, 
            win_size=self.win_size,
            mode='test', 
            target_id=target_id,  # 改为 target_id
            silent=True
        )

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # CPU或GPU设备
        if torch.cuda.is_available():
            self.model.cuda()
        else:
            self.model = self.model.to('cpu')

    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        for i, input_data in enumerate(vali_loader):
            if isinstance(input_data, (list, tuple)):
                input_data = input_data[0]  # Handle case where loader returns tuple
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):
        print("======================TRAIN MODE======================")
        
        # 确保模型保存目录存在
        os.makedirs(self.model_save_path, exist_ok=True)
        path = os.path.join(self.model_save_path, f"{self.dataset}_checkpoint.pth")
        
        print(f"Model will be saved to: {path}")
        
        # 检查目录是否可写
        if not os.access(os.path.dirname(path), os.W_OK):
            print(f"Warning: No write permission for {self.model_save_path}")
            # 使用临时目录
            temp_path = os.path.join(os.path.expanduser("~"), "temp_models")
            os.makedirs(temp_path, exist_ok=True)
            path = os.path.join(temp_path, f"{self.dataset}_checkpoint.pth")
            print(f"Using alternative save location: {path}")
        
        time_now = time.time()
        global_step = 0  # 初始化 global_step
        
        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=3, verbose=True)
        
        for epoch in range(self.num_epochs):
            iter_count = 0
            loss_list = []
            
            self.model.train()
            epoch_time = time.time()
            
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
            for i, input_data in enumerate(pbar):
                iter_count += 1
                self.optimizer.zero_grad()
                
                # 处理输入数据
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.model(input)
                
                # 计算重构损失
                rec_loss = torch.mean(self.criterion(output, input))  # 确保得到标量
                
                # 计算系列损失
                series_loss = 0.0
                prior_loss = 0.0
                
                for u in range(len(prior)):
                    if u == 0:
                        series_loss = torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
                        ).detach()))
                        prior_loss = torch.mean(my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)),
                            series[u].detach()
                        ))
                    else:
                        series_loss += torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
                        ).detach()))
                        prior_loss += torch.mean(my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)),
                            series[u].detach()
                        ))
                
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)
                
                # 计算总损失
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss
                
                loss_list.append(loss1.item())  # 现在应该可以正确转换为标量了
                
                # 记录训练指标
                if (i + 1) % 10 == 0:
                    self.logger.scalar_summary('train_loss', loss1.item(), global_step)
                    self.logger.scalar_summary('reconstruction_loss', rec_loss.item(), global_step)
                    self.logger.scalar_summary('series_loss', series_loss.item(), global_step)
                    self.logger.scalar_summary('prior_loss', prior_loss.item(), global_step)
                
                # 更新进度条描述
                pbar.set_postfix({
                    'loss': f'{loss1.item():.4f}',
                    'rec_loss': f'{rec_loss.item():.4f}'
                })
                
                # 更新进度信息
                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                # 反向传播和优化
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()
                
                global_step += 1  # 更新全局步数
            
            # 每个epoch结束后的处理
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss_list)
            
            vali_loss1, vali_loss2 = self.vali(self.test_loader)
            
            # 记录验证指标
            self.logger.scalar_summary('validation_loss1', vali_loss1, global_step)
            self.logger.scalar_summary('validation_loss2', vali_loss2, global_step)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                epoch + 1, train_steps, train_loss, vali_loss1))
            
            # 早停检查
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
            # 学习率调整
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
        
        # 保存最终模型
        final_model_path = os.path.join(self.model_save_path, f"{self.dataset}_checkpoint.pth")
        torch.save(self.model.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")

    def test(self):
        model_path = os.path.join(self.model_save_path, f"{self.dataset}_checkpoint.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        
        print(f"Loading model from {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduce=False)
        
        global_step = 0  # For logging

        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            global_step += 1  # Increment for logging

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)
        
        # Log training energy distribution
        self.logger.histo_summary('train_energy_distribution', train_energy, global_step)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            # Metric
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)
        
        # Log threshold value
        self.logger.scalar_summary('anomaly_threshold', thresh, global_step)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        test_string_ids = []  # 新增：记录每个测试样本所属的电池组ID
        
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)

            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)
            
            # 如果数据加载器提供了电池组ID信息，则记录
            if hasattr(self.thre_loader.dataset, 'all_data') and 'test_string_ids' in self.thre_loader.dataset.all_data:
                if i < len(self.thre_loader.dataset.all_data['test_string_ids']):
                    test_string_ids.append(self.thre_loader.dataset.all_data['test_string_ids'][i])

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)
        
        # Log test energy distribution
        self.logger.histo_summary('test_energy_distribution', test_energy, global_step)

        pred = (test_energy > thresh).astype(int)

        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
        anomaly_state = False
