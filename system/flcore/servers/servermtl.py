import time
import torch
from flcore.clients.clientmtl import clientMTL
from flcore.servers.serverbase import Server
from threading import Thread


class FedMTL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.fix_ids = False
        self.method = "FedMTL"

        self.dim = len(self.flatten(self.global_model))
        self.W_glob = torch.zeros((self.dim, self.num_join_clients), device=args.device)
        self.device = args.device

        I = torch.ones((self.num_join_clients, self.num_join_clients))
        i = torch.ones((self.num_join_clients, 1))
        omega = (I - 1 / self.num_join_clients * i.mm(i.T)) ** 2
        self.omega = omega.to(args.device)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientMTL)
            
        print(f"\nJoin clients / total clients: {self.num_join_clients} / {self.num_clients}")
        print("Finished creating server and clients.")


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

            print(f"*************************** 2.server {self.method} select_clients ***************************\n")

            s_t = time.time()
            # 新增2————————————————————————————————————————————————————————————————————————————————
            ids = self.get_selected_clients(i)
            select_id.append([i, ids])
            # ————————————————————————————————————————————————————————————————————————————————

            # -------------------------------------------------------------------------

            self.aggregate_parameters()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                res = self.evaluate(i)
                # 新增3————————————————————————————————————————————————————————————————————————————————
                # 记录当前模型的状态，loss,accuracy
                resc = self.addvalue(res)
                colum_value.append(resc)
                # ————————————————————————————————————————————————————————————————————————————————
                
            for idx, client in enumerate(self.selected_clients):
                start_time = time.time()
                
                client.set_parameters(self.W_glob, self.omega, idx)

                client.send_time_cost['num_rounds'] += 1
                client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]
            self.Budget.append(time.time() - s_t)
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break


        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        # 新增4————————————————————————————————————————————————————————————————————————————————
        self.write_info(select_id, colum_value)
        # ————————————————————————————————————————————————————————————————————————————————

        self.save_results()


    def flatten(self, model):
        state_dict = model.state_dict()
        keys = state_dict.keys()
        W = [state_dict[key].flatten() for key in keys]
        return torch.cat(W)

    def aggregate_parameters(self):
        self.W_glob = torch.zeros((self.dim, self.num_join_clients), device=self.device)
        for idx, client in enumerate(self.selected_clients):
            self.W_glob[:, idx] = self.flatten(client.model)
