from flcore.clients.clientbn import clientBN
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data
from threading import Thread
import time


class FedBN(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.method = "FedBN"
        self.fix_ids = False
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientBN)

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
            print(f"*************************** 2.server {self.method} select_clients ***************************\n")

            s_t = time.time()
            # 新增2————————————————————————————————————————————————————————————————————————————————
            ids = self.get_selected_clients(i)
            select_id.append([i, ids])
            # ————————————————————————————————————————————————————————————————————————————————

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
            self.aggregate_parameters()
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

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientBN)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
