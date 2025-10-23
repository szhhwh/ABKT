import pandas as pd
import math
import numpy as np

def load_dataset(dataset,type,min_length):
    if dataset == 'ASSISTment2009':
        path = "./Datasets/" + dataset + "/raw.csv"
        usecols = ['order_id', 'user_id', 'problem_id', 'correct', 'skill_id', 'type']
        print('Loading dataset from', path, ' with cols:', usecols)
        csv_data = pd.read_csv(path, usecols=usecols)
        # 按时间顺序对数据排序
        csv_data.sort_values(['order_id'], ascending=True)
        # 选择特定类型（type）的数据
        print('Choosing data where type=', type)
        type_data = csv_data[csv_data['type'] == type]
        user_list = type_data['user_id'].tolist()
        problem_list = type_data['problem_id'].tolist()
        correct_list = type_data['correct'].tolist()
        skill_list = type_data['skill_id'].tolist()
        # 每一行为[user_id，problem_id，correct，skill_id]
        data_raw = [user_list, problem_list, correct_list, skill_list]
    elif dataset == 'AICFE':
        path = "./Datasets/" + dataset + "/"+type+"/unit-"+type+".csv"
        file = open(path)
        lines = file.readlines()
        skip = 0
        user_list = []
        problem_list = []
        correct_list = []
        skill_list = []
        for line in lines:
            if skip != 0:
                data = line.strip('\n').split(',')
                user = data[0]
                problem = data[3]
                skill = data[4]
                score = data[5]
                full_score = data[6]
                if skill != 'n.a.' and full_score != 'n.a.':
                    user_list.append(user)
                    problem_list.append(problem)
                    skill_list.append(skill)
                    if score == full_score:
                        correct_list.append(1)
                    else:
                        correct_list.append(0)
            else:
                skip += 1
        data_raw = [user_list, problem_list, correct_list, skill_list]

    #统计用户序列长度
    user_count = {}
    # user_count 是一个字典，[user_id] = length
    for i in range(data_raw[0].__len__()):
        userid = data_raw[0][i]
        if user_count.__contains__(userid):
            user_count[userid] += 1
        else:
            user_count[userid] = 1
    #过滤与重新编号
    user_id = {}
    item_id = {}
    skill_id = {}
    user_list_filtered = [] # 用户列表
    item_list_filtered = [] # 题目列表
    correct_list_filtered = [] # 正确与否列表
    filtered_Q_matrix = [] # 每个题目对应的技能列表
    print('Filting data where sequence length>=',min_length)
    # 每一行为[user_id，problem_id，correct，skill_id]
    for i in range(data_raw[0].__len__()):
        user = data_raw[0][i]
        item = data_raw[1][i]
        correct = data_raw[2][i]
        # 分割技能ID
        if dataset == 'ASSISTment2009':
            skillids = data_raw[3][i].split(',')
        else:
            skillids = data_raw[3][i].split('~~')

        if user_count[user] >= min_length: # 过滤最小序列长度
            if not user_id.__contains__(user):
                user_id[user] = user_id.__len__()
            if not item_id.__contains__(item):
                item_id[item] = item_id.__len__()
                skills = []
                for skill in skillids:
                    if not skill_id.__contains__(skill):
                        skill_id[skill] = skill_id.__len__()
                    skills.append(skill_id[skill])
                filtered_Q_matrix.append(skills)
            user_list_filtered.append(user_id[user])
            item_list_filtered.append(item_id[item])
            correct_list_filtered.append(correct)
    print(user_id)
    print(skill_id)
    return [user_list_filtered,item_list_filtered,correct_list_filtered,filtered_Q_matrix]


def get_split_triplet(dataset, type, min_length):
    [user_list, item_list, correct_list,Q_matrix] = load_dataset(dataset, type, min_length)
    user_num = max(user_list) + 1
    item_num = max(item_list) + 1
    skill_num = max([max(i) for i in Q_matrix]) + 1
    record_num = user_list.__len__()
    # all_sequences = {userid:[[itemids,...],[correct,...]]}
    all_sequences = {}
    for i in range(user_list.__len__()):
        if all_sequences.__contains__(user_list[i]):
            all_sequences[user_list[i]][0].append(item_list[i])
            all_sequences[user_list[i]][1].append(correct_list[i])
        else:
            all_sequences[user_list[i]] = [[item_list[i]],[correct_list[i]]]
    # train_triplet [[userid,itemid,corect],...]
    # test_triplet [[userid,itemid,corect],...]
    train_triplet = []
    test_triplet = []
    for user in all_sequences:
        sequence_length = all_sequences[user][0].__len__()
        train_length = sequence_length-1
        for index in range(sequence_length):
            if index < train_length:
                train_triplet.append([user,
                                      all_sequences[user][0][index],
                                      all_sequences[user][1][index]])
            else:
                test_triplet.append([user,
                                      all_sequences[user][0][index],
                                      all_sequences[user][1][index]])

    return user_num,item_num,skill_num,record_num,train_triplet,test_triplet,Q_matrix


def get_split_sequences(dataset, type, min_length):
    [user_list, item_list, correct_list, Q_matrix] = load_dataset(dataset, type, min_length)
    user_num = max(user_list) + 1
    item_num = max(item_list) + 1
    skill_num = max([max(i) for i in Q_matrix]) + 1
    record_num = user_list.__len__()
    # all_sequences = {userid:[[itemids,...],[correct,...]]}
    all_sequences = {}
    for i in range(user_list.__len__()):
        if all_sequences.__contains__(user_list[i]):
            all_sequences[user_list[i]][0].append(item_list[i])
            all_sequences[user_list[i]][1].append(correct_list[i])
        else:
            all_sequences[user_list[i]] = [[item_list[i]], [correct_list[i]]]

    train_sequences = {}
    test_triplet = []
    # train_sequences = {userid:[[itemids,...],[correct,...]]}
    # test_triplet [[userid,itemid,corect],...]
    for user in all_sequences:
        sequence_length = all_sequences[user][0].__len__()
        train_length = sequence_length - 1
        train_sequences[user] = [[all_sequences[user][0][0:train_length]],
                                 [all_sequences[user][1][0:train_length]]]
        test_item_sequence = all_sequences[user][0][train_length:]
        test_correct_sequence = all_sequences[user][1][train_length:]
        for i in range(test_item_sequence.__len__()):
            test_triplet.append([user,test_item_sequence[i],test_correct_sequence[i]])
    return user_num,item_num,skill_num,record_num,train_sequences,test_triplet,Q_matrix


def get_kfold_sequences(dataset, type, min_length, n_splits=5, fold_id=0, seed=42):
    """
    以用户为单位进行K折划分：指定fold作为测试集，其余作为训练集；
    - 训练集：每个训练用户的序列去掉最后一次（与原有逻辑一致）
    - 测试集：仅包含测试用户序列的最后一次三元组（user, item, correct）

    (user_num, item_num, skill_num, record_num, train_sequences, test_triplet, Q_matrix)
    """
    # user_list：用户列表
    # item_list：题目列表
    # correct_list：正确与否列表
    # Q_matrix：题目-技能对应表
    # Q_matrix = [
    #     [0, 2],      # 题目0涉及技能0和2
    #     [1],         # 题目1仅涉及技能1
    #     [0, 1, 3]    # 题目2涉及技能0、1、3
    # ]
    [user_list, item_list, correct_list, Q_matrix] = load_dataset(dataset, type, min_length)
    user_num = max(user_list) + 1
    item_num = max(item_list) + 1
    skill_num = max([max(i) for i in Q_matrix]) + 1
    record_num = user_list.__len__()

    # 按用户聚合序列
    all_sequences = {}
    for i in range(len(user_list)):
        u = user_list[i]
        it = item_list[i]
        cor = correct_list[i]
        if u in all_sequences:
            all_sequences[u][0].append(it)
            all_sequences[u][1].append(cor)
        else:
            all_sequences[u] = [[it], [cor]]

    unique_users = np.array(sorted(all_sequences.keys()))
    assert n_splits >= 2, "n_splits 至少为 2"
    assert 0 <= fold_id < n_splits, "fold_id 必须在 [0, n_splits) 内"

    # 可复现的用户打乱与均匀分块
    rng = np.random.RandomState(seed)
    perm_users = rng.permutation(unique_users)
    fold_sizes = np.full(n_splits, len(perm_users) // n_splits, dtype=int)
    fold_sizes[: len(perm_users) % n_splits] += 1
    indices = np.cumsum(fold_sizes)
    starts = np.concatenate(([0], indices[:-1]))
    folds = [perm_users[s:e] for s, e in zip(starts, indices)]

    test_users = set(folds[fold_id].tolist())
    train_users = set(unique_users.tolist()) - test_users

    train_sequences = {}
    test_triplet = []

    # 构建训练与测试集
    for u in unique_users:
        items = all_sequences[u][0]
        labels = all_sequences[u][1]
        seq_len = len(items)
        train_len = seq_len - 1
        train_sequences[u] = [[items[0:train_len]], [labels[0:train_len]]]

    for u in test_users:
        items = all_sequences[u][0]
        labels = all_sequences[u][1]
        seq_len = len(items)
        train_len = seq_len - 1
        # 仅最后一次进入测试三元组（保持与原有评估一致）
        for idx in range(train_len, seq_len):
            test_triplet.append([u, items[idx], labels[idx]])

    return user_num, item_num, skill_num, record_num, train_sequences, test_triplet, Q_matrix



if __name__ == '__main__':
    # dataset = 'ASSISTment2009'
    # type = 'RandomIterateSection'
    dataset = 'AICFE'
    type = 'math'
    min_length = 10
    load_dataset(dataset, type, min_length)

