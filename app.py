from fastapi import FastAPI
from pydantic import BaseModel
from datetime import date
from loguru import logger
from typing import List
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import HTTPException, Depends
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine
from datetime import datetime
import catboost as ctb
import os
import time

app = FastAPI()

class PostGet(BaseModel):
    id: int
    text: str
    topic: str
    
    class Config:
        orm_mode = True

class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
 #       MODEL_PATH = 'C:/Users/Andrey Grigorovich/Final/catboost_model'
    return MODEL_PATH
    
# "catboost_model"

def load_models():
    model_path = get_model_path("/my/super/path")
    CTB = ctb.CatBoostClassifier()  # здесь не указываем параметры, которые были при обучении, в дампе модели все есть
    CTB.load_model(model_path)
    return CTB




model = load_models()


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 20000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []

    
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
        #os.system('cls')

    conn.close()
    return pd.concat(chunks, ignore_index=True)
    
  
    
def load_features(quary: str) -> pd.DataFrame:
    return batch_load_sql(quary)


# Preparation table with watched posts
start_time = time.time()
print('Loading watched posts...')
df_watched = load_features("""SELECT user_id, posts FROM scetch_watched_posts_lesson_22""")
print('Watch posts loaded')
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
print('Loading Users...')
df_user = load_features("""SELECT user_id, gender, age, country, city, exp_group, min, max FROM scetch_features_dates_lesson_22""")
print('Users loaded')
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
print('Loading posts...')
df_posts = load_features("""SELECT post_id, topic, tfidf, top_month, text, watched_month  FROM scetch_posts_lesson_22""")
print('Posts loaded')
print("--- %s seconds ---" % (time.time() - start_time))

df_watched.set_index('user_id', inplace=True)
#cols = ['hour', 'dayofweek', 'timestamp_delta', 'counter', 'gender', 'age', 'country', 'city', 'exp_group', 'topic', 'tfidf', 'watched_month', 'top_month']
cols = ['gender', 'age', 'country', 'city', 'exp_group', 'topic', 'tfidf', 'watched_month', 'top_month', 'hour', 'dayofweek', 'timestamp_delta', 'counter', 'delta']
print('...Finish loading')
      
      
@app.get('/post/recommendations', response_model=List[PostGet])
def recommended_posts(id: int, time: datetime=datetime.now(), limit: int = 5) -> List[PostGet]:
    
    if id not in df_user['user_id'].values:
        
        posts = df_posts[['post_id', 'text', 'topic', 'top_month']].sort_values('top_month', ascending=False).reset_index().drop('index', axis=1) 
        for i in range(5):
            top = PostGet(
                id = posts['post_id'].iloc[i],
                text = posts['text'].iloc[i],
                topic = posts['topic'].iloc[i]
            )
            top5.append(top) 
        return top5
    else:

        #Sort out watched posts
        list_viewed = list(map(int, df_watched.loc[id][0].strip('{}').split(',')))
        df = df_posts[~df_posts['post_id'].isin(list_viewed)].reset_index().drop('index', axis=1).copy()
        
    #    df_user['min'] = pd.to_datetime(df_user['min'])
     #   df_user['max'] = pd.to_datetime(df_user['max'])
        
        df_user['user_id'] = df_user['user_id'].astype('int32')
        df_user['gender'] = df_user['gender'].astype('int8')
        df_user['age'] = df_user['age'].astype('int8')
        df_user['country'] = df_user['country'].astype('int8')
        df_user['city'] = df_user['city'].astype('int16')
        df_user['exp_group'] = df_user['exp_group'].astype('int8')
     #   df_user['os'] = df_user['os'].astype('int8')
     #   df_user['source'] = df_user['source'].astype('int8')
     #   df_user['lst_topic'] = df_user['lst_topic'].astype('int8')
     #   df_user['max_topic'] = df_user['max_topic'].astype('int8')
     

        df_posts['post_id'] = df_posts['post_id'].astype('int16')
        df_posts['topic'] = df_posts['topic'].astype('int8')
        df_posts['tfidf'] = df_posts['tfidf'].astype('int32')
        df_posts['top_month'] = df_posts['top_month'].astype('int16')
        df_posts['watched_month'] = df_posts['watched_month'].astype('int16')
        
        
        #Prepare dataset for prediction   
        gender = df_user[df_user['user_id']==id]['gender'].values[0]
        age = df_user[df_user['user_id']==id]['age'].values[0]
        country = df_user[df_user['user_id']==id]['country'].values[0]
        city = df_user[df_user['user_id']==id]['city'].values[0]
        exp_group = df_user[df_user['user_id']==id]['exp_group'].values[0]
        timestamp_delta = (time - df_user[df_user['user_id']==id]['max'].item()).total_seconds()
        counter = (time - df_user[df_user['user_id']==id]['min'].item()).total_seconds()
        delta = (time - df_user[df_user['user_id']==id]['max'].item()).total_seconds()
    #    os = df_user[df_user['user_id']==id]['os'].values[0]
    #    source = df_user[df_user['user_id']==id]['source'].values[0]
    #    lst_topic = df_user[df_user['user_id']==id]['lst_topic'].values[0]
    #    max_topic = df_user[df_user['user_id']==id]['max_topic'].values[0]

        df['hour'] = time.hour
        df['dayofweek'] = time.weekday()
        df['timestamp_delta'] = timestamp_delta
        df['counter'] = counter
        df['delta'] = delta
        df['gender'] = gender
        df['age'] = age
        df['country'] = country
        df['city'] = city
        df['exp_group'] = exp_group
    #    df['os'] = os
    #    df['source'] = source
    #    df['lst_topic'] = lst_topic
    #    df['max_topic'] = max_topic
        
        df['hour'] = df['hour'].astype('int8')
        df['dayofweek'] = df['dayofweek'].astype('int8')

        df['timestamp_delta'] = df['timestamp_delta'].map(int).astype('int32')
        df['counter'] = df['counter'].map(int).astype('int32')
                  
        back_text = {"topic":  {3: 'movie', 
                                1: 'covid', 
                                5: 'sport', 
                                4: 'politics', 
                                0: 'business', 
                                2: 'entertainment',
                                6: 'tech'}
                }
                   
        
        probs = model.predict_proba(df[cols])[:, 1]

        posts = pd.concat([df[['post_id', 'text', 'topic', 'top_month']], pd.DataFrame(probs, columns=['Predict'])], axis=1).sort_values(by='Predict', ascending=False)
        posts['post_id'] = posts['post_id'].astype('int32')
        posts = posts.replace(back_text, inplace=False).reset_index().drop('index', axis=1)

        top5 = []
        if posts['Predict'].iloc[4] >= 0.4:                
            for i in range(5):
                top = PostGet(
                    id = posts['post_id'].iloc[i],
                    text = posts['text'].iloc[i],
                    topic = posts['topic'].iloc[i]
                )
                top5.append(top)
        else:
            posts = posts.sort_values('top_month', ascending=False).reset_index().drop('index', axis=1)
            for i in range(5):
                top = PostGet(
                    id = posts['post_id'].iloc[i],
                    text = posts['text'].iloc[i],
                    topic = posts['topic'].iloc[i]
                )
                top5.append(top) 
        
    
        return top5