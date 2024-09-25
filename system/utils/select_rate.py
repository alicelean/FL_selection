import random
import pandas as pd

def random_select_k_elements(original_list, k):
    if k > len(original_list):
        print("Error: k is larger than the length of the list.")
        return None

    selected_elements = random.sample(original_list, k)
    return selected_elements

def process_row(row, k):
    id_list=[int(i) for i in row['id_list'][1:-1].split(",")]
    newlist = random_select_k_elements(id_list, k)
    print("lenght is",len(newlist))
    row['id_list']=newlist
    return row
def read_data_file(filename,target_length):
    df = pd.read_csv(filename, header=0)
    # 使用apply方法对DataFrame的每一行应用process_row函数
    new_df = df.apply(process_row, axis=1, args=(target_length,))
    new_df=new_df.drop(columns='num')
    return new_df







if __name__ == "__main__":

    client_num=50
    rate=0.5
    newrate=0.2
    target_length=int(0.2*client_num) # 替换为你想要的新的id_list长度

    input_file = "/Users/alice/Desktop/python/PFL/res/selectids/Cifar100_select_client_ids"+str(client_num)+"_"+str(rate)+".csv"
    output_file ="/Users/alice/Desktop/python/PFL/res/selectids/Cifar100_select_client_ids"+str(client_num)+"_"+str(newrate)+".csv"

    new_df=read_data_file(input_file,target_length)

    # if target_length <= len(id_list):
    #
    #     print(f"新的id_list已经写入到{output_file}文件中！")
    # else:
    #     print("目标长度超过原始id_list的长度。")

    new_df.to_csv(output_file)

