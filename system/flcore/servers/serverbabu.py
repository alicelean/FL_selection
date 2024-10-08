import random
from flcore.clients.clientbabu import clientBABU
from flcore.servers.serverbase import Server
from threading import Thread
import time,os


class FedBABU(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.method = "FedBABU"
        self.fix_ids = False
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientBABU)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()


    def train(self):
        print(f"*************************** {self.method} train ***************************")
        self.writeparameters()
        # 新增1————————————————————————————————————————————————————————————————————————————————
        colum_value = []
        select_id = []
        if self.fix_ids:
            self.read_fix_id()

        # ————————————————————————————————————————————————————————————————————————————————

        for i in range(self.global_rounds+1):

            s_t = time.time()
            # 新增2————————————————————————————————————————————————————————————————————————————————
            ids = self.get_selected_clients(i)
            select_id.append([i, ids])
            # ————————————————————————————————————————————————————————————————————————————————

            print(f"*************************** 2.server {self.method} select_clients ***************************\n")


            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                res = self.evaluate(i)
                # 新增3————————————————————————————————————————————————————————————————————————————————
                # 记录当前模型的状态，loss,accuracy
                resc = self.addvalue(res)
                colum_value.append(resc)
                # ————————————————————————————————————————————————————————————————————————————————

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()
            self.Budget.append(time.time() - s_t)
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))

        for client in self.clients:
            client.fine_tune()
        print("\n-------------Evaluate fine-tuned personalized models-------------")
        _=self.evaluate(i)
        # 新增4————————————————————————————————————————————————————————————————————————————————
        self.write_info(select_id, colum_value)
        # ————————————————————————————————————————————————————————————————————————————————

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientBABU)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model.base)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            for e in range(self.fine_tuning_epoch):
                client.fine_tune()