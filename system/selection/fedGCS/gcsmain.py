from types import SimpleNamespace
import torch
from torch import nn
import numpy as np
import logging
import math
import random
import csv
from device_env import Evaluator, base_path
import os
import sys
import pickle
import warnings
from utils.logger import info, error
from autos.train import autos_selection

warnings.filterwarnings('ignore')

from plato.servers import fedavg
from plato.config import Config
from plato.clients import simple
from plato.trainers import basic


class Trainer(basic.Trainer):
    """A federated learning trainer used by the Oort that keeps track of losses."""

    def process_loss(self, outputs, labels) -> torch.Tensor:
        """Returns the loss from CrossEntropyLoss, and records the sum of
        squaures over per_sample loss values."""
        loss_func = nn.CrossEntropyLoss(reduction="none")
        per_sample_loss = loss_func(outputs, labels)

        # Stores the sum of squares over per_sample loss values
        self.run_history.update_metric(
            "train_squared_loss_step",
            sum(np.power(per_sample_loss.cpu().detach().numpy(), 2)),
        )

        return torch.mean(per_sample_loss)

    def get_loss_criterion(self):
        """Returns the loss criterion."""
        return self.process_loss


class Client(simple.Client):
    """
    A federated learning client that calculates its statistical utility
    """

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        """Wrap up generating the report with any additional information."""
        train_squared_loss_step = self.trainer.run_history.get_metric_values(
            "train_squared_loss_step"
        )

        report.statistical_utility = report.num_samples * np.sqrt(
            1.0 / report.num_samples * sum(train_squared_loss_step)
        )

        # power_config = Config().clients.power_config
        delta = 5
        # power_init = self.client_id//len(power_config) + random.uniform(-delta, delta)

        # report.client_power = power_init

        return report


class FedGCSServer(fedavg.Server):
    """A federated learning server using oort client selection."""

    def __init__(
            self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )

        # Clients that will no longer be selected for future rounds.
        self.blacklist = []

        # All clients' utilities
        self.client_utilities = {}

        # All clients‘ training times
        self.client_durations = {}

        # Keep track of each client's last participated round.
        self.client_last_rounds = {}

        # Number of times that each client has been selected
        self.client_selected_times = {}

        # The desired duration for each communication round
        self.desired_duration = Config().server.desired_duration

        self.explored_clients = []
        self.unexplored_clients = []

        self.exploration_factor = Config().server.exploration_factor
        self.step_window = Config().server.step_window
        self.pacer_step = Config().server.desired_duration

        self.penalty = Config().server.penalty

        self.penalty_beta = Config().server.penalty_beta

        self.total_clients = Config().clients.total_clients

        # Keep track of statistical utility history.
        self.util_history = []

        # Cut off for sampling client utilities
        self.cut_off = (
            Config().server.cut_off if hasattr(Config().server, "cut_off") else 0.95
        )

        # Times a client is selected before being blacklisted
        self.blacklist_num = (
            Config().server.blacklist_num
            if hasattr(Config().server, "blacklist_num")
            else 10
        )

        self.selction_model_state_dict = None

    def configure(self) -> None:
        """Initialize necessary variables.以初始化父类所需的变量"""
        super().configure()

        self.client_utilities = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }
        self.client_durations = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }
        self.client_power = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }
        self.client_last_rounds = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }
        self.client_selected_times = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }

        self.round_energy = 0

        self.unexplored_clients = list(range(1, self.total_clients + 1))

    def weights_aggregated(self, updates):
        """Method called at the end of aggregating received weights.在接收到客户端的模型更新后，
        更新每个客户端的效用、训练时间、参与轮次等信息，
        同时调整 pacer 以控制每个训练轮次的持续时间，
        并且根据客户端的选择次数对过于频繁参与的客户端进行黑名单处理。"""
        for update in updates:
            # Extract statistical utility and local training times
            # 提取客户端的统计效用和本地训练时长，并将其保存到相应的字典中
            # 更新客户端的效用
            self.client_utilities[update.client_id] = update.report.statistical_utility
            self.client_durations[update.client_id] = update.report.training_time
            self.client_last_rounds[update.client_id] = self.current_round
            # self.client_power[update.client_id] = update.report.client_power

            # Calculate client utilities of explored clients --- oort
            self.client_utilities[update.client_id] = self.calc_client_util(
                update.client_id
            )

        # Adjust pacer # 更新效用历史记录
        self.util_history.append(
            sum(update.report.statistical_utility for update in updates)
        )
        # 判断是否满足调整 pacer 的条件
        if self.current_round >= 2 * self.step_window:
            # 计算最近两段历史的效用总和
            last_pacer_rounds = sum(
                self.util_history[-2 * self.step_window: -self.step_window]
            )  # 计算当前段的效用总和
            current_pacer_rounds = sum(self.util_history[-self.step_window:])
            # 如果历史效用总和大于当前效用总和，增加 desired_duration
            if last_pacer_rounds > current_pacer_rounds:
                self.desired_duration += self.pacer_step

        # Blacklist clients who have been selected self.blacklist_num times
        # 将选择次数超过一定次数的客户端加入黑名单
        for update in updates:
            if self.client_selected_times[update.client_id] > self.blacklist_num:
                self.blacklist.append(update.client_id)

    def choose_clients_fovar(self, clients_pool, clients_count):
        """Choose a subset of the clients to participate in each round."""
        selected_clients = []

        return selected_clients

    def choose_clients_fedmarl(self, clients_pool, clients_count):
        """Choose a subset of the clients to participate in each round."""
        selected_clients = []

        return selected_clients

    def choose_clients_oort(self, clients_pool, clients_count):
        """Choose a subset of the clients to participate in each round."""
        selected_clients = []
        #
        return selected_clients

    def convert_to_binary_list(self, selected_indices, list_size):
        '''将一组选中的客户端索引 selected_indices 转换为一个二进制列表（binary list）。其中，选中的客户端的对应位置会被标记为 1，未选中的客户端对应位置标记为 0。'''
        binary_list = [0] * list_size
        for index in selected_indices:
            if 0 <= index < list_size:
                binary_list[index] = 1
        return torch.FloatTensor(binary_list)

    # 实际选择方法未实现？
    def gen_device_selection(self, device_eval_, clients_pool, clients_count):

        '''生成客户端选择的历史记录，并使用不同的客户端选择策略进行选择。通过这三个不同的选择策略 (OORT, FedMARL, favor)，方法模拟了客户端的选择过程，并记录了每次选择的性能表现。'''
        # max_accuracy, optimal_set, k = gen_marlfs(fe_, N_ACTIONS=2, N_STATES=64, EXPLORE_STEPS=300)
        # 循环 100 次，用于生成不同的客户端选择和性能记录
        for i in range(100):
            # 使用 OORT 策略选择客户端
            selected_numbers = self.choose_clients_oort(clients_pool, clients_count)
            # 这里是
            ## 计算所选客户端的效用
            performances = self.cal_selected_utility(selected_numbers)
            # 将所选客户端转换为二进制表示的列表形式
            selected_numbers = self.convert_to_binary_list(selected_numbers, len(clients_pool))
            # 将选择的客户端和对应的性能记录到 device_eval_ 中
            device_eval_._store_history(selected_numbers, performances)


            # 使用 FedMARL 策略选择客户端
            selected_numbers = self.choose_clients_fedmarl(clients_pool, clients_count)
            performances = self.cal_selected_utility(selected_numbers)
            selected_numbers = self.convert_to_binary_list(selected_numbers, len(clients_pool))
            device_eval_._store_history(selected_numbers, performances)


            # 使用 Favor 策略选择客户端
            selected_numbers = self.choose_clients_favor(clients_pool, clients_count)
            performances = self.cal_selected_utility(selected_numbers)
            selected_numbers = self.convert_to_binary_list(selected_numbers, len(clients_pool))
            device_eval_._store_history(selected_numbers, performances)


    def choose_clients(self, clients_pool, clients_count,base_path):
        # best_selection_test = self.choose_clients_oort(clients_pool, clients_count)
        # 如果客户端效用和训练时长相等（这可能是某种特殊条件）方法根据不同条件（如客户端的效用和训练时长的关系）选择客户端。
        # 如果客户端的效用与训练时长相等，则直接使用 OORT 策略选择客户端；否则，使用更复杂的流程，
        # 包括生成历史记录、评估不同策略（OORT、FedMARL、favor）的表现，并使用机器学习模型（如 autos_selection）来最终确定最佳的客户端选择。
        # 这一过程确保了选择的客户端能够最大化系统的效用，同时记录了每次选择的历史。
        # 则使用 OORT 策略选择客户端
        if self.client_utilities == self.client_durations:
            best_selection_test = self.choose_clients_oort(clients_pool, clients_count)
        else:
            print("GCS, choose clients")
            # 创建一个 Evaluator 对象，用于记录和评估客户端选择的历史表现
            device_eval = Evaluator()
            # 调用 gen_device_selection 方法，生成客户端选择的历史记录并使用不同策略评估
            #self.gen_device_selection(device_eval, clients_pool, clients_count)
            file_path = f"{base_path}/history/device_env.pkl"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # 将 `device_eval` 对象序列化并保存到文件
            with open(f'{base_path}/history/device_env.pkl', 'wb') as f:
                pickle.dump(device_eval, f)

            # 使用一个模型来生成新的客户端选择
            new_selection, self.selction_model_state_dict = autos_selection(self.num_clients,
                                                                            self.selction_model_state_dict)
            best_selection_test = None
            best_optimal_test = -1000

            for s in new_selection:
                # 获取选择的客户端索引
                indice_select = torch.arange(0, self.total_clients)[s.operation == 1]
                # 计算该选择下的效用
                test_result = self.cal_selected_utility(indice_select.tolist())
                # 如果当前选择的效用更好，则更新最优选择
                if test_result > best_optimal_test:
                    best_selection_test = indice_select.tolist()
                    best_optimal_test = test_result
                    info(f'found best on test : {best_optimal_test}')
            # 更新每个客户端被选中的次数
            for client in best_selection_test:
                self.client_selected_times[client] += 1
            # for client_id in best_selection_test:
            #     self.unexplored_clients.remove(client_id)

        info(f'found test generation in our method! the choice is {best_selection_test}')
        # 设置每轮选择的客户端数量
        self.clients_per_round = len(best_selection_test)
        return best_selection_test

    def calc_client_util(self, client_id):
        """Calculate the client utility.计算并返回给定客户端的效用值。客户端效用基于两个因素：

客户端的基础效用（self.client_utilities[client_id]）。
客户端在不同轮次的训练表现，通过一个基于轮次的修正项来调整。
如果客户端的训练时长超过期望时长（self.desired_duration），则对其效用应用一个惩罚因子，降低其被选中的概率。"""
        client_utility = self.client_utilities[client_id] + math.sqrt(
            0.1 * math.log(self.current_round) / self.client_last_rounds[client_id]
        )

        if self.desired_duration < self.client_durations[client_id]:
            global_utility = (
                                     self.desired_duration / self.client_durations[client_id]
                             ) ** self.penalty
            client_utility *= global_utility
        return client_utility

    def cal_selected_utility(self, selected_clients):
        '''计算给定一组被选中客户端的总效用'''
        client_utility = []
        # client_energy =[]
        for client_id in selected_clients:
            # client_energy.append(self.client_durations[client_id]*self.client_power[client_id])
            # desired_energy = self.desired_duration*self.client_power[client_id]
            # if desired_energy < self.client_durations[client_id]*self.client_power[client_id]:
            #     energy_utility = (
            #         self.desired_energy[client_id]/ (self.client_durations[client_id]*self.client_power[client_id])
            #     )** self.penalty_beta
            client_utility.append(self.client_utilities[client_id])

        # self.round_energy = float(sum(client_energy))
        total_utility = float(sum(client_utility))
        return total_utility

    # def get_logged_items(self) -> dict:
    #     super().get_logged_items()
    #     logged_items["energy"] = self.round_energy


def main():
    np.random.seed(1)
    trainer = Trainer
    client = Client(trainer=trainer)
    server = Server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()