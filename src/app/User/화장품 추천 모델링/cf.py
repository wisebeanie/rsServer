import sys
from unittest import result
import warnings
import numpy as np
import pandas as pd
import surprise
import requests
import json
from sklearn.decomposition import NMF # Use this for training Non-negative Matrix Factorization
from sklearn.utils.extmath import randomized_svd # Use this for training Singular Value Decomposition
from sklearn.manifold import TSNE # Use this for training t-sne manifolding

def getData():
    dir = 'src/app/User/화장품 추천 모델링/최종데이터/'
    df_product = pd.read_csv(dir + 'basic_data_edit.csv', usecols=['00.상품_URL','00.상품코드','01.브랜드','02.상품명','03.가격','04.제품 주요 사양','05.모든 성분','06.총 평점','07.리뷰 개수','08_1.별점 1점','08_2.별점 2점','08_3.별점 3점','08_4.별점 4점','08_5.별점 5점','09_1.피부타입_건성','09_2.피부타입_복합성','09_3.피부타입_지성','10_1.피부고민_보습','10_2.피부고민_진정','10_3.피부고민_주름/미백','11_1.피부자극_없음','11_2.피부자극_보통','11_3.피부자극_있음'],encoding='cp949')
    df_review = pd.read_csv(dir + 'total_review.csv', usecols=['code','user','type','tone','problem','rating','feature','review','total_rating'],encoding='cp949')

    user_review_count = df_review['user'].value_counts()
    user_review_count = pd.DataFrame(user_review_count)
    user_review_count = user_review_count.reset_index()
    user_review_count.columns = ['user','count']

    df_review_count = pd.merge(df_review,user_review_count,on='user',how='left')
    df_review_count = df_review_count[df_review_count['count']>=2]
    
    A = df_review_count.pivot_table(index = 'code', columns = 'user',values = 'total_rating')
    A = A.copy().fillna(0)
    
    final_df = df_review_count[['user','code','total_rating']]
    
    reader = surprise.Reader(rating_scale = (1,5))

    col_list = ['user','code','total_rating']
    data = surprise.Dataset.load_from_df(final_df[col_list], reader)
    
    trainset = data.build_full_trainset()
    option = {'name' : 'pearson'}
    algo = surprise.KNNBasic(sim_options = option)

    algo.fit(trainset)
    
    name_list = final_df['user'].unique()
    name_list = pd.Series(name_list)

    index = name_list[name_list == '뮹뮹'].index[0]

    name_list = final_df['user'].unique()
    name_list = pd.Series(name_list)
    
    result = algo.get_neighbors(index,k=5)
    
    code_list = []
    for r1 in result:
        max_rating = data.df[data.df['user']==name_list[r1]]['total_rating'].max()
        cos_id = data.df[(data.df['total_rating']==max_rating)&(data.df['user']==name_list[r1])]['code'].values
        
        code_list.append(cos_id)
    
    result = []
    
    for codes in code_list:
        for code in codes:
            #result.append((df_product[df_product['00.상품코드']==code]['02.상품명'].to_json(orient='index', force_ascii=False)))
            result.append(code)
            
    result1 = json.dumps(result)
    
    print(result1)       


    
    
    
if __name__ == "__main__":
    getData()