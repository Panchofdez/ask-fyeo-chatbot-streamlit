import streamlit as st
import random
import string
import requests
import time
import re
import os
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from dotenv import load_dotenv
from streamlit_js_eval import streamlit_js_eval

    
load_dotenv()

@st.cache_data
def setup():
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    return


def authenticate(url):
    try:
        print("AUTHENTICATE")
        resp = requests.post(url, json={ "email": st.secrets["email"], "password": st.secrets["password"]})
        resp.raise_for_status()
        token = resp.json()["token"]
           
        return token
    except Exception as e:
        print("ERROR: ", e)

@st.cache_data
def get_data(url, token):
    try:
        print("RETRIEVE DATA")
        resp = requests.get(url, headers={"authorization" : f"Bearer {token}"})
        resp.raise_for_status()
        data = resp.json()     
        return data["FAQ"]
    except Exception as e:
        print("ERROR: ", e)
        
@st.cache_data
def get_pattern_embeddings(_transformer_model, patterns):      
    print("GET PATTERNS")  
    return transformer_model.encode(patterns)

@st.cache_resource            
def load_transformer_model():
    print("LOAD MODEL")
    return SentenceTransformer("multi-qa-mpnet-base-cos-v1")

@st.cache_resource            
def load_stemmer_model():
    return PorterStemmer() 

def tokenize(sentence):
    return nltk.word_tokenize(sentence)
       
def remove_punc(text):
    stop_punc_words = set(list(string.punctuation))
    filtered_text = [token for token in text.split() if token not in stop_punc_words]
    
    return " ".join(filtered_text)       

def check_response(tag, patterns, question, response, stemmer):
    '''
    Determines the validity of the chatbot's response
    '''
    
    question = tokenize(question.lower())
    patterns = ' '.join(patterns).lower()
    response = response.lower()
    tag =tag.lower()
    ignore_words = ['?', '!', '.', ',', 'are', 'you', 'can', 'and', 'let','where', 'why', 'what', 'how' , 'when', 'who', 'the' , 'need', 'for', 'have', 'but']
    stemmed_words  = [stemmer.stem(w) for w in question if w not in ignore_words and len(w) > 2 ] # avoid punctuation or words like I , a , or 

    if len(stemmed_words) == 0:
        stemmed_words = [stemmer.stem(w) for w in question]

    found = [ w for w in stemmed_words if re.search(w, response) or re.search(w,tag ) != None or re.search(w, patterns)] #check if the question has words related in the response
    print("FOUND", found)
    return len(found) > 0
    
def get_response(query, transformer_model, stemmer_model, data, pattern_embeddings, all_patterns):
    default_answer = ("", "Hmm... I do not understand that question. Please try again or ask a different question")
 
    query_embedding = transformer_model.encode(query)    

    # similarity = self.sentence_transformer_model.similarity(query_embedding, pattern_embeddings)
    # print("SIMILARITY", similarity)
    scores = util.dot_score(query_embedding, pattern_embeddings)[0].cpu().tolist()
    
    pattern_score_pairs = list(zip(all_patterns, scores))
    
    #Sort by decreasing score
    pattern_score_pairs = sorted(pattern_score_pairs, key=lambda x: x[1], reverse=True)

    # #Output passages & scores
    # print("DOT SCORE")
    # for pattern, score in pattern_score_pairs[:20]:
    #     print(score, pattern)
        
    target_pattern, target_score = pattern_score_pairs[0]
    target_tag = pattern_tag_map[target_pattern]
    result = (target_tag, target_pattern, target_score)
    print("FINAL ANSWER", result)
    
    for faq in data :
        tag = faq["tag"]
        patterns = faq["patterns"]
        responses = faq["responses"]
        if target_tag == tag:
            resp = random.choice(responses)     
            if check_response(tag, patterns, query, resp, stemmer_model):
                return  (target_tag, f"{resp}")
        
    return default_answer    


# Streamed response emulator
def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)    
        
def startConversation():
    return False

def form_callback():
    if not student_number or not first_name or not last_name or not program or not email:
        st.write(f":red[Error: Missing Information]") 
    
    elif len(student_number) != 9 and student_number.isnumeric():
        st.write(f":red[Error: Invalid Student Number]")
    
    elif email.find("@") == -1 or email.lower().split("@")[1]  != "ryerson.ca" and email.lower().split("@")[1] != "torontomu.ca":
        st.write(f":red[Error: Invalid Email]")
        
    else:    
        try:
        #     startConversation()
        #     raise Exception('Invalid details')
            
            st.session_state.student_number = student_number
            st.session_state.first_name = first_name
            st.session_state.last_name = last_name
            st.session_state.program = program
            st.session_state.email = email
            st.session_state.conversation_mode = True
            st.session_state.disabled = True
        except Exception as e:
            st.write(f":red[Error: {str(e)}]")

        
setup()

print("HELLO WORLD")
url = "https://ask-fyeo-chatbot-68o6.onrender.com"
if "token" not in st.session_state:
    st.session_state.token = authenticate(f"{url}/login")


data = get_data(f"{url}/faq", st.session_state.token)

pattern_tag_map = {}
all_patterns = []

for faq in data:
    tag = faq["tag"]
    patterns = faq["patterns"]

    for sent in patterns:
        clean_sent = remove_punc(sent.lower())
        pattern_tag_map[clean_sent] = tag
        all_patterns.append(clean_sent)    

transformer_model = load_transformer_model()
stemmer_model = load_stemmer_model()

pattern_embeddings = get_pattern_embeddings(transformer_model, all_patterns)

st.title("Ask FYEO")

if "conversation_mode" not in st.session_state:
    st.session_state.conversation_mode = False
    
if "feedback_mode" not in st.session_state:
    st.session_state.feedback_mode = False    
    
if "finish_mode" not in st.session_state:
    st.session_state.finish_mode = False        
    
if "disabled" not in st.session_state:
    st.session_state.disabled = False    
    

if not st.session_state.conversation_mode:
    with st.form("student_details",clear_on_submit=True):
        student_number = st.text_input("Student Number")
        first_name = st.text_input("First Name", "")
        last_name = st.text_input("Last Name", "")
        program = st.selectbox(
            "Select Program",
            ("Aerospace", "Biomedical", "Chemical", "Civil", "Computer", "Electrical" , "Industrial", "Mechanical" ),
        )
  
        email = st.text_input("Email", "")

        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit" , on_click=form_callback, args=())         
          
elif st.session_state.conversation_mode:
    print(st.session_state.student_number, st.session_state.first_name, st.session_state.last_name, st.session_state.program, st.session_state.email)
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": f"Hello {st.session_state.first_name}, it's nice to meet you! I am the FYEO chatbot and I'm here to answer any of your questions about your first year of engineering." })   

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
            
#     if st.session_state.feedback_mode:
#         get_feedback = "Was I able to answer your question?"    
#         feedback_response = None
#         with st.chat_message("assistant"):
#             st.markdown(get_feedback)
            
#         with st.chat_message("user"):
#             sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
#             feedback = st.feedback("thumbs")
#             print("FEEDBACK", feedback)
#             if feedback is not None:
#                 feedback_response = f"You selected: {sentiment_mapping[feedback]}" 
#                 st.markdown(feedback_response)
                
              
        
#         st.session_state.messages.append({"role": "assistant", "content": get_feedback })   
#         if feedback_response is not None:    
#             st.session_state.messages.append({"role": "assistant", "content": feedback_response })  
       
#         st.session_state.feedback_mode = False
#         st.session_state.finish_mode = True
        
    # elif st.session_state.finish_mode:
    #     continue_conversation = "Please ask me another question!"
    #     with st.chat_message("assistant"):
    #         st.markdown(continue_conversation)  
    #     st.session_state.messages.append({"role": "assistant", "content": continue_conversation }) 
    #     st.session_state.finish_mode = False    
            
    # Accept user input
    if prompt := st.chat_input("Ask me your question!"):
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        tag, response = get_response(prompt, transformer_model, stemmer_model, data, pattern_embeddings, all_patterns)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response, unsafe_allow_html=True)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response, "tag" : tag })
        st.session_state.feedback_mode = True
        
                   
        
            
        


    # print(st.session_state.messages)