import random
import time
from flcore.clients.clientrep import clientRep
from flcore.servers.serverbase import Server
from threading import Thread


class FedRep(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.fix_ids = False
        self.method = "FedRep"
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientRep)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


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

            print(f"*************************** 2.server {self.method} select_clients ***************************\n")

            s_t = time.time()
            # 新增2————————————————————————————————————————————————————————————————————————————————
            ids = self.get_selected_clients(i)
            select_id.append([i, ids])
            # ————————————————————————————————————————————————————————————————————————————————

            # -------------------------------------------------------------------------

            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
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
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        # 新增4————————————————————————————————————————————————————————————————————————————————
        self.write_info(select_id, colum_value)
        # ————————————————————————————————————————————————————————————————————————————————
        self.save_results()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientRep)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
        

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.num_join_clients))

        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model.base)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples