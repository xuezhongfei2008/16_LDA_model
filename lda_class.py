#!usr/bin/env python
#encoding:utf-8
import os
import jieba
from gensim import corpora,models,similarities
import pickle
import jieba.posseg as pseg

# step1: 文件输入
q_path = '../generate_data/knowledge_all_session.csv'
user_dict = "../original_data/finWordDict.txt"
stop_words_path = "../original_data/stop_words.txt"
model_path = "./tf-idf_model"
ldaModel_path="./Theme_Model"

class SimQuestion():
    def __init__(self,q_path,user_dict,stop_words_path,model_path):
        jieba.load_userdict(user_dict)
        self.q_path = q_path
        self.stop_words_path=stop_words_path
        self.model_path=model_path
        self.stopwords=self.get_stop_words()

    #读取停止词
    def get_stop_words(self):
        # 去停用词
        stopwords = [line.strip() for line in open(self.stop_words_path, 'r', encoding='utf-8').readlines()]
        stop_words_list = set(stopwords)
        return stop_words_list

    #读取文档，并提取标准问题
    def read_data(self):
        all_doc_list=[]
        question_list=[]
        with open(self.q_path,'r',encoding='utf-8') as f:
            line=f.readline()
            while line:
                if(len(line))>0:
                    question_list.append(line.split(sep="\t")[0])    #提取标准问题
                    #正常且分词
                    # raw_words=list(jieba.cut(line,cut_all=False))
                    # all_doc_list.append([word for word in raw_words if word not in  self.stopwords])
                    #提取动名词
                    s_cut = list(pseg.cut(line))
                    flag = ['n', 'v', 'x']
                    all_doc_list.append(list(ii.word for ii in s_cut if ii.flag in flag and ii.word not in self.stopwords))
                line=f.readline()
        print("read_data:完毕")
        return all_doc_list,question_list

    def build_tfidf_model(self):
        if (os.path.exists(self.model_path+"/all_doc.dict")):
            # print("加载存在的数据：字典、问题列表、corpus词频数据.....")
            #加载字典
            dictionary = corpora.Dictionary.load(self.model_path+'/all_doc.dict')
            #加载问题列表
            pickle_file=open(self.model_path+'/question_list.pkl','rb')
            question_list=pickle.load(pickle_file)
            #加载corpus 词频数据
            corpus = corpora.MmCorpus(self.model_path+'/all_doc.mm')
        else:
            print("第一次运行生产数据：字典、问题列表、corpus词频数据......")
            all_doc_list,question_list=self.read_data()
            #生成字典并保存
            dictionary=corpora.Dictionary(all_doc_list)
            dictionary.save(self.model_path+'/all_doc.dict')
            #保存问题列表
            pickle_file=open(self.model_path+'/question_list.pkl','wb')
            pickle.dump(question_list,pickle_file)
            pickle_file.close()
            #生产corpus词频数据，并保存
            corpus=[dictionary.doc2bow(doc) for doc in all_doc_list]
            corpora.MmCorpus.serialize(self.model_path+'/all_doc.mm',corpus)

        if (os.path.exists(self.model_path+"/tfidf_model")):
            # print("加载已经存在的模型：tfidf_model.....")
            #加载tfidf模型
            tfidf_model = models.TfidfModel.load(self.model_path+'/tfidf_model')

        else:
            print("需要训练新模型，并保存：tfidf_model.....")
            #生产tfidf模型并保存
            tfidf_model=models.TfidfModel(corpus)
            tfidf_model.save(self.model_path+'/tfidf_model')
        #应用corpus_tfidf模型
        corpus_tfidf = tfidf_model[corpus]
        #生产相似性矩阵索引
        corpus_simi_matrix=similarities.SparseMatrixSimilarity(corpus_tfidf,num_features=len(dictionary.keys()))
        return dictionary,question_list,corpus,tfidf_model,corpus_simi_matrix


def test(text,dictionary,corpus,tfidf_model,stop_words,ldaModel_path):
    #step1： 加载lda模型
    if os.path.exists(ldaModel_path + "/lda_model"):
        print("加载已经存在的模型：lda_model.....")
        # 加载 lda 模型
        lda_model=models.ldamodel.LdaModel.load(ldaModel_path + "/lda_model")
    else:
        print("需要训练新模型，并保存：lda_model.....")
        # 生产 lda 模型并保存
        corpus_tfidf = tfidf_model[corpus]
        lda_model = models.ldamodel.LdaModel(corpus=corpus_tfidf,
                                             id2word=dictionary,
                                             num_topics=10,
                                             update_every=0,
                                             passes=20)
        lda_model.save(ldaModel_path + "/lda_model")

    ## 打印前 topic 的词分布
    for i in range(10):
        print("10个主题词分布： ",lda_model.print_topic(i))
        pass

    #setp3: 对新文档进行主题 分类


    text_list=[]
    s_cut = list(pseg.cut(text))
    flag = ['n', 'v', 'x']
    text_list.append(list(ii.word for ii in s_cut if ii.flag in flag and ii.word not in stop_words))
    print("text_list", text_list)
    text_corpus=dictionary.doc2bow(text_list)
    doc_lda=lda_model[text_corpus]
    for topic in sorted(doc_lda,key=lambda item:-item[1]):
        print("新文档主题分布",topic[0],":",topic[1],"---->>",lda_model.print_topic(topic[0]))

if __name__=='__main__':
    simi_tfidf=SimQuestion(q_path,user_dict,stop_words_path,model_path)
    dictionary, question_list, corpus, tfidf_model, corpus_simi_matrix = simi_tfidf.build_tfidf_model()
    stop_words = simi_tfidf.get_stop_words()
    text = "请问一下任性贷、任性付是苏宁的什么产品？"
    test(text, dictionary, corpus, tfidf_model, stop_words, ldaModel_path)
