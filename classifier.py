import re
import jieba.analyse
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split


# 读取文件内容，拆分出每篇文章，并去除account名称，推送日期，标点符号，换行符等
# 传入文件路径，公众号名称
# 返回一个list，存储处理好的文章
def split_file(file_path, account_name):
    split_pattern = re.compile('\n--------------------------------\n', re.S)   # 用于拆分文章的pattern
    file_content = open(file_path, 'r', encoding='utf-8').read()
    split_list = split_pattern.split(file_content)
    i = 0
    # 循环，对每篇文章摘掉无用信息，存入split_list
    while i < len(split_list):
        split_list[i] = re.sub('Article No.*', '', split_list[i])
        split_list[i] = re.sub('\d{4}-\d{2}-\d{2} ', '', split_list[i])
        split_list[i] = re.sub('[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;：。？、~@#￥%……&*（）-]+', '', split_list[i])
        split_list[i] = split_list[i].replace(account_name, '')
        split_list[i] = split_list[i].replace('\n', '')
        i += 1
    return split_list


# 载入停用词表
# 传入停用词文件路径，返回一个list，存储停用词
def load_stop_word(file_path):
    file = open(file_path, 'r')
    stop_word = []
    for line in file.readlines():
        line = line.strip()
        stop_word.append(line)
    file.close()
    return stop_word


# 提取整个account的关键词
# 1.用jieba对每篇文章分词，过滤停用词
# 2.将所有文章里的词记入词典，并统计词频
# 3.对词典key值按出现词频排序，取前30个作为公众号关键词
# 传入文章列表，停用词表，特征集合
def extract_keyword(splitList, stopWordList, key_set):
    cut_result = []
    tag_dic = {}
    for content in splitList:
        word_list = list(jieba.cut(content, False))
        outstring = ''
        for word in word_list:
            if word not in stopWordList:
                outstring = outstring + word + ' '
        cut_result.append(outstring)
        keywords = jieba.analyse.extract_tags(outstring, -1, withWeight=True)
        for item in keywords:
            if item[0] in tag_dic:
                tag_dic[item[0]] += 1
            else:
                tag_dic[item[0]] = 1
    pairs = list(tag_dic.items())
    items = [[x, y] for (y, x) in pairs]   # 对换key和value
    items.sort()
    key_list = []
    for i in range(len(items) - 1, len(items) - 1 - 30, -1):
        key_list.append(items[i][1])
    # 将关键词放入特征集合，相当于取交集
    key_set = key_set.update(set(key_list))


# 获得特征向量
# 1.遍历每篇文章出现的词语，在词典中，则特征向量相应位置赋值为1
# 2.将特征向量对应label赋值为公众号名称
# 传入特征词典，文章列表，公众号名称，特征向量，label列表
def get_vector(dic, splitList, accountName, xtrain, ytrain):
    for essay in splitList:
        words = jieba.analyse.extract_tags(essay, -1)
        vector = [0] * len(dic)
        for i in words:
            if i in dic:
                vector[dic[i]] = 1
        xtrain += [vector]
        ytrain += [accountName]
    for i in range(0, len(xtrain), 1):
        print(xtrain[i], ytrain[i])


if __name__ == "__main__":
    # 装载停用词列表
    stopWordPath = "stopwords-utf8.txt"
    stopWordList = load_stop_word(stopWordPath)

    # 创建集合用于存储特征
    key_set = set()
    # 分别获取三个公众号的文章列表和关键词
    filePath1 = "aaa.txt"
    accountName1 = "aaa"
    splitList1 = split_file(filePath1, accountName1)
    extract_keyword(splitList1, stopWordList, key_set)

    filePath2 = "bbb.txt"
    accountName2 = "bbb"
    splitList2 = split_file(filePath2, accountName2)
    extract_keyword(splitList2, stopWordList, key_set)

    filePath3 = "ccc.txt"
    accountName3 = "ccc"
    splitList3 = split_file(filePath3, accountName3)
    extract_keyword(splitList3, stopWordList, key_set)

    # 建立索引
    fea_dic = {}
    count = 0
    for keyword in key_set:
        fea_dic[keyword] = count
        count += 1
    print(fea_dic)

    # 获得三个account对应文章的特征向量
    xtrain1 = []
    ytrain1 = []
    get_vector(fea_dic, splitList1, accountName1, xtrain1, ytrain1)

    xtrain2 = []
    ytrain2 = []
    get_vector(fea_dic, splitList2, accountName2, xtrain2, ytrain2)

    xtrain3 = []
    ytrain3 = []
    get_vector(fea_dic, splitList3, accountName3, xtrain3, ytrain3)

    # 合并特征向量与相应label，并按7:3分配训练集和测试集
    Xset = xtrain1 + xtrain2 + xtrain3
    Yset = ytrain1 + ytrain2 + ytrain3
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xset, Yset, test_size=0.05, random_state=42)

    # 调整参数
    max_depth_options = list(range(1, 30, 1))
    n_estimators_options = list(range(1, 100, 5))
    results = []
    for max_depth in max_depth_options:
        for n_estimators_size in n_estimators_options:
            alg = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators_size, random_state=50)
            alg.fit(Xtrain, Ytrain)
            predict = alg.predict(Xtest)
            results.append((max_depth, n_estimators_size, (Ytest == predict).mean()))
            print((Ytest == predict).mean(), max_depth, n_estimators_size)
    print(max(results, key=lambda x: x[2]))


