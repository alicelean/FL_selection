import copy
import torch
from flcore.clients.clientdyn import clientDyn
from flcore.servers.serverbase import Server
from threading import Thread
import time


class FedDyn(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.method = "FedDyn"
        self.fix_ids = False
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientDyn)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.alpha = args.alpha
        
        self.server_state = copy.deepcopy(args.model)
        for param in self.server_state.parameters():
            param.data = torch.zeros_like(param.data)


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
            self.update_server_state()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nBest local accuracy.")
        print("\nAveraged time per iteration.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        # 新增4————————————————————————————————————————————————————————————————————————————————
        self.write_info(select_id, colum_value)
        # ————————————————————————————————————————————————————————————————————————————————
        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientDyn)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def add_parameters(self, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() / self.num_join_clients

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)
            
        for client_model in self.uploaded_models:
            self.add_parameters(client_model)

        for server_param, state_param in zip(self.global_model.parameters(), self.server_state.parameters()):
            server_param.data -= (1/self.alpha) * state_param

    def update_server_state(self):
        assert (len(self.uploaded_models) > 0)

        model_delta = copy.deepcopy(self.uploaded_models[0])
        for param in model_delta.parameters():
            param.data = torch.zeros_like(param.data)

        for client_model in self.uploaded_models:
            for server_param, client_param, delta_param in zip(self.global_model.parameters(), client_model.parameters(), model_delta.parameters()):
                delta_param.data += (client_param - server_param) / self.num_clients

        for state_param, delta_param in zip(self.server_state.parameters(), model_delta.parameters()):
            state_param.data -= self.alpha * delta_param
