#encoding:utf-8

import jieba.analyse
import jieba.posseg
from simi_tfidf.tfidf_similarity import *
from gensim import models

jieba.load_userdict('../original_data/finWordDict.txt')

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
    test_cut=jieba.lcut(text)
    text_list=[]
    for item in test_cut:
        if item not in stop_words:
            text_list.append(item)
    print("text_list", text_list)
    text_corpus=dictionary.doc2bow(text_list)
    doc_lda=lda_model[text_corpus]
    # print("doc_lda",doc_lda)
    for topic in sorted(doc_lda,key=lambda item:-item[1]):
        print("新文档主题分布",topic[0],":",topic[1],"---->>",lda_model.print_topic(topic[0]))


if __name__=='__main__':

    # step1: 文件输入
    q_path = '../generate_data/knowledge_all_session.csv'
    user_dict = "../original_data/finWordDict.txt"
    stop_words_path = "../original_data/stop_words.txt"
    #存放了client_question_2017.csv的tf-idf模型
    model_path = "../Ex_keywords/keyWords_Model"
    ldaModel_path="./Theme_Model"

    # step2:加载tfidf模型
    SimQuestion_model = SimQuestion(q_path, user_dict, stop_words_path, model_path)
    dictionary, question_list, corpus,tfidf_model, corpus_simi_matrix=SimQuestion_model.build_tfidf_model()
    stop_words = SimQuestion_model.get_stop_words()

    text="请问一下任性贷、任性付是苏宁的什么产品？"
    test(text,dictionary,corpus,tfidf_model,stop_words,ldaModel_path)