import pandas as pd
import psycopg2
from sqlalchemy import create_engine

user = 'postgres'
pwd = 'password123'
host = 'insideout.c49fqzysd1id.us-east-1.rds.amazonaws.com'
port = '5432'
db = 'insideout'

postgreseng = create_engine('postgresql+psycopg2://{0}:{1}@{2}:{3}/{4}'.format(user, pwd, host, port, db))
# raw_results = pd.read_sql_table('results_raw', con=postgreseng)


def insert_raw_results(user_id, sensibility_test_factor, amazon_comprehend_mixed_score,
                       amazon_comprehend_neutral_score, amazon_comprehend_positive_score,
                       amazon_comprehend_negative_score, personal_NLP_analytics_score,
                       audio_spectogram_model_score, final_result, transcribe_text):
    # sql_statement = """INSERT INTO results_raw(user_id, sensibility_test_factor, amazon_comprehend_mixed_score,
    #                    amazon_comprehend_neutral_score, amazon_comprehend_positive_score,
    #                    amazon_comprehend_negative_score, personal_NLP_analytics_score,
    #                    audio_spectogram_model_score, final_result) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)"""

    sql_statement = """INSERT INTO results_raw VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    try:
        connection = psycopg2.connect(user=user, password=pwd, host=host, port=port, database=db)
        cursor = connection.cursor()
        record_to_insert = (user_id, sensibility_test_factor, amazon_comprehend_mixed_score,
                       amazon_comprehend_neutral_score, amazon_comprehend_positive_score,
                       amazon_comprehend_negative_score, personal_NLP_analytics_score,
                       audio_spectogram_model_score, final_result, transcribe_text)
        cursor.execute(sql_statement, record_to_insert)
        connection.commit()
        print('Record inserted successfuly into mobile table.')
    except (Exception, psycopg2.Error) as error:
        print('Failed to insert record into raw_results table', error)

insert_raw_results('test', 0, 0, 0, 0, 0, 0, 0, 0)