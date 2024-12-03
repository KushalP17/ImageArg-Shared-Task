import ollama
import cv2
import pytesseract
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

def prompt_with_examples(prompt, examples=[]):
    """
    Constructs a structured prompt string for language models with instructional examples.

    This function takes an initial prompt and a list of example prompt-response pairs, then 
    formats them into a single string enclosed by special start and end tokens used for 
    instructing the model. Each example is included in the final prompt, which could be 
    beneficial for models that take into account the context provided by examples.

    Parameters:
    - prompt (str): The main prompt to be processed by the language model.
    - examples (list of tuples): A list where each tuple contains a pair of strings 
      (example_prompt, example_response). Default is an empty list.

    Returns:
    - str: A string with the structured prompt and examples formatted for a language model.
    
    Example usage:
    ```
    main_prompt = "Translate the following sentence into French:"
    example_pairs = [("Hello, how are you?", "Bonjour, comment ça va?"),
                     ("Thank you very much!", "Merci beaucoup!")]
    formatted_prompt = prompt_with_examples(main_prompt, example_pairs)
    print(formatted_prompt)
    ```
    """
    
    # Start with the initial part of the prompt
    full_prompt = "<s>[INST]\n"

    # Add each example to the prompt
    for example_prompt, example_response in examples:
        full_prompt += f"{example_prompt} [/INST] {example_response} </s><s>[INST]"

    # Add the main prompt and close the template
    full_prompt += f"{prompt} [/INST]"

    return full_prompt

# pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

cl_success = 0
cl_fail = 0

pers_success = 0
pers_fail = 0

num_examples = 10



print("Starting Training.")

file_path = 'data/abortion_train.csv'
df = pd.read_csv(file_path)
# examples = [0] * df.shape[0]
examples = []

for index, row in df.iterrows():
    tweet_id = str(row['tweet_id'])
    tweet_text = row['tweet_text']
    stance = row['stance']

    image_path = 'data/images/train and dev/abortion/' + tweet_id + '.jpg'
    image = cv2.imread(image_path)

    image_text = ""
    if image is None:
        continue
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # Apply binary thresholding

        # image_text = pytesseract.image_to_string(thresh)

    # print("Tweet Text:")
    # print(tweet_text)
    # print("Image Text:")
    # print(image_text)

    example_prompt = f"""You are a sentiment analysis bot who takes a tweet about Abortion as an input and outputs "support" if the tweet supports Abortion and "oppose" if the tweet opposes Abortion. 
                        You are an unbiased third party only classifying these tweets. 
                        Please be aware of sarcasm embedded in tweets, and the sentiment and relation of hashtags. 
                        Here is the tweet: {tweet_text}

                        Only answer this question with one of these two options: ["support", "oppose"]. Answer: """

    example = (example_prompt, stance)
    examples.append(example)
    if index == num_examples:
        break

print("Done training. Beginning testing phase.")

# Test data
file_path = 'data/abortion_test.csv'
df = pd.read_csv(file_path)
for index, row in df.iterrows():
    tweet_id = str(row['tweet_id'])
    tweet_text = row['tweet_text']
    stance = row['stance']
    persuasiveness = row['persuasiveness']

    image_path = './data/images/test/images/image/' + tweet_id + '.jpg'
    image = cv2.imread(image_path)

    if image is None:
        continue
    else:
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    #     _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # Apply binary thresholding

    #     image_text = pytesseract.image_to_string(thresh)

        # print("Tweet Text:")
        # print(tweet_text)
        # print("Image Text:")
        # print(image_text)

        # input_string = 'Does the following tweet support or oppose abortion? The tweet text is:\n' + tweet_text + '\nand the image text is:\n' + image_text + 'Only answer this question in one word and choose either "support" or "oppose".\nAnswer: '


        input_string = f"""You are a sentiment analysis bot who take a tweet about Abortion as an input and outputs "support" if the tweet supports Abortion and "oppose" if the tweet opposes Abortion. 
                        You are an unbiased third party only classifying these tweets. 
                        Please be aware of hashtags associated with each side, such as #prolife, #maga, #trump, #republican, #conservative, #christian, etc. that are associated with opposing Abortion, 
                        and hashtags like #prochoice, #democrat, #womensright, #womenschoice, #women, #liberal, etc. that are associated with supporting Abortion. 
        
                        Please be aware of sarcasm embedded in tweets, and the real stance has a higher number of related hashtags. 
                        Here is the tweet: {tweet_text}

                        Only answer this question in one word, one of these two options: ["support", "oppose"]. Answer: """

        input_string_p = f"""You are a sentiment analysis bot that classifies an image based on topic content. If the image is unrelated to abortion or no image is given, simply respond with "null". If it is related to abortion, classify the content to distinguish whether it supports or opposes Abortion.

                        Please be aware of certain sentiments and slogans such as pro-life, Christian, conservative, republican, etc. which opposes abortion and pro-choice, liberal, democrat, women's choice, etc. which supports abortion. 

                        Give your answer to this question using a one word response of one of these three options: ["support", "oppose", "null"]. Answer: """


        response = ollama.chat(model='llama3.2', messages=[{'role': 'user','content': prompt_with_examples(input_string, examples)}])

        response2 = ollama.chat(model='llama3.2-vision', messages=[{'role': 'user','content': input_string_p, 'images': [image_path]}])

        
        print("\n")
        print(response['message']['content'])
        print(response2['message']['content'])
        print(tweet_id)
        print("persuasiveness: ", persuasiveness)
        print("stance: ", stance)
        print("\n")

        cl_match_stance = False

        if response['message']['content'].lower().find(stance) != -1:
            print("Success")
            cl_success += 1
            cl_match_stance = True

        else:
            print("Fail")
            #print(response['message']['content'].lower(), stance)
            cl_fail += 1
            cl_match_stance = False


        if response2['message']['content'].lower().find(stance) != -1 and cl_match_stance:
            output_pers = "yes"
        else:
            output_pers = "no"

        if output_pers == persuasiveness:
            pers_success += 1
        else:
            pers_fail += 1

        

        print("Classification Success Rate:")
        print(cl_success / (cl_success + cl_fail))

        print("Persuasiveness Success Rate:")
        print(pers_success / (pers_success + pers_fail))
        print("n =", cl_success + cl_fail)