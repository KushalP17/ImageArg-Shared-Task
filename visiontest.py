import ollama
import pandas as pd

file_path = 'data/abortion_test.csv'
df = pd.read_csv(file_path)
for index, row in df.iterrows():
    tweet_id = str(row['tweet_id'])
    tweet_text = row['tweet_text']
    stance = row['stance']
    persuasiveness = row['persuasiveness']

    input_string_p = f"""You are a sentiment analysis bot that classifies an image based on topic content. If the image is unrelated to abortion or no image is given, simply respond with "null". If it is related to abortion, classify the content to distinguish whether it supports or opposes Abortion.

                        Please be aware of certain sentiments and slogans such as pro-life, Christian, conservative, republican, etc. which opposes abortion and pro-choice, liberal, democrat, women's choice, etc. which supports abortion. 

                        Give your answer to this question using a one word response of one of these three options: ["support", "oppose", "null"]. Answer: """

    image_path = './data/images/test/images/image/' + tweet_id + '.jpg'


    response2 = ollama.chat(model='llama3.2-vision', messages=[{'role': 'user','content': input_string_p, 'images': [image_path]}])

    print(response2['message']['content'])