from resp.apis.serp_api import Serp
from resp.apis.cnnp import connected_papers
from resp.apis.semantic_s import Semantic_Scholar
from resp.apis.acm_api import ACM
from resp.apis.arxiv_api import Arxiv
from resp.resp import Resp
import streamlit as st
import openai
import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import os
import json
import random
from email.mime.text import MIMEText
import smtplib
import replicate
import bs4



st.title('Find Daily Research Papers!')
st.markdown("**This tool helps you find academic paper from major journals!**")

st.subheader('How to use?')
st.write('**1.🐸 Enter topic of research interest. It can be anything from computer science, physics, RAG, LLM etc.**')
st.write('**2. 📝 Select publications from - Google Scholar, Arxiv, American Society for Microbiology (ASM), Semantic Scholar, Association for Computation Linguistics (ACL), Proceedings of Machine Learning Research (PMLR), Neurips**')

col3, col4 = st.columns([0.45, 0.55])
with col3:
    st.write('**3. 📕 Use checkbox to add to Reading list.**')
    
with col4:
    st.checkbox("", key=f"dummy")

st.markdown("""---""")

# class SessionState:
#     def __init__(self, **kwargs):
#         for key, val in kwargs.items():
#             setattr(self, key, val)

# def get_session_state(**kwargs):
#     ctx = st.report_thread.get_report_ctx()
#     if not hasattr(ctx, "session_state"):
#         ctx.session_state = SessionState(**kwargs)
#     return ctx.session_state

# state = get_session_state(counter=0, name="", another_var="")

# PAGES = {
#     "Home": app,
#     "Reading list": reading_list,
# }

# st.sidebar.title('Navigation')
# selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# page = PAGES[selection]
# page(state)


Paper_names = ['Zero-shot learning with common sense knowledge graphs']
# keyword     = ['Zero-shot learning']
serp_api_key = st.sidebar.text_input("Enter your Google scholar API key (optional):")

paper_engine = Resp(serp_api_key)
columns = ['title', 'link']
reading_list_df = pd.DataFrame(columns=columns)

def dataframe_to_markdown(df):
        markdown_lines = ["# Reading List\n"]
        for _, row in df.iterrows():
            markdown_lines.append(f"- [{row['title']}]({row['link']})\n")
        return "".join(markdown_lines)


def reading_list():
    
    global reading_list_df
    urls = []
    st.header('Reading List')
    # summarize_add = st.sidebar.text_input('Enter number to summarize: 👇')
    # replicate_api_token = st.sidebar.text_input('Enter your replicate api key: ')
    openai_api = st.sidebar.text_input("Enter your [OpenAI](https://openai.com/index/openai-api/) API key to prompt (optional):")
    if openai_api:
        prompt = st.sidebar.text_area('Custom prompt... 👇', placeholder="""- Summarize text in 1\n- what is 4 about?\n- how is 5 relevant to ML?""")

    replicate_api = st.sidebar.text_input('Your [replicate](https://replicate.com/) API to enable speak (optional): 👇')
    
    os.environ["REPLICATE_API_TOKEN"] = replicate_api
    api = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])
    timestamp = datetime.now().strftime("%Y/%m/%d")
    container = st.container(border=True)
    
    def get_summary_num(prompt):
        openai.api_key = openai_api
        if prompt:
            completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f'from the given text extract the number and only return that: {prompt}'
                },
            ],
            )
            # print(completion.choices[0].message.content)
            response = completion.choices[0].message.content
            # st.write(response)
            return response
    
    
    def summary(url):
        response = requests.get(url,headers={'User-Agent': 'Mozilla/5.0'})
        soup = bs4.BeautifulSoup(response.text,'lxml')

        text = soup.body.get_text(' ', strip=True)
        
        # link = df['link'].iloc[index]
        # st.header('Summarize url')
        # url = st.text_input('Please enter your URL here:')
        # try:
        if url:
            openai.api_key = openai_api
            if prompt:
                completion = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": f'In the following prompt, replace any number with this paper: {prompt}. Answer based on the contents of this text: {text}'
                    },
                ],
                )
                # print(completion.choices[0].message.content)
                response = completion.choices[0].message.content
                # st.write(response)
                return response
        
            else:
                completion = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": f"generate a summary about it this text in one line: {text}"
                    },
                ],
                )
                # print(completion.choices[0].message.content)
                response = completion.choices[0].message.content
                return response
        
        else:
            ""
        # except:
        #     st.error('Please enter OpenAI API', icon="🚨")

    # st.write(reading_list_df)
    if st.session_state.reading_list:
        if reading_list_df.empty:
        # st.write(st.session_state.reading_list)
            reading_list_df = pd.DataFrame(st.session_state.reading_list)
            
        reading_list_df._append(st.session_state.reading_list, ignore_index=True)
        reading_list_df.drop_duplicates('title', inplace=True)
        reading_list_df.reset_index(drop=True, inplace=True)
        reading_list_df = reading_list_df.shift(periods=1).fillna(0)
        reading_list_df.drop_duplicates()
        
        # st.write(reading_list_df)
        # count = 0
        for i, row in reading_list_df.iloc[1:].iterrows():
            # count += 1
            st.markdown(f"{i}.[{row['title']}]({row['link']})")
        try:
            if openai_api:
                summarize_add = get_summary_num(prompt)
                # st.write(summarize_add)
            
                if summarize_add:
                    counter = 0
                    for i, row in reading_list_df.iloc[1:].iterrows():
                        # print('i', i)
                        if i == int(summarize_add):
                            st.success(f"🤖 {row['title']}!")
                            url = row['link']
                            res = summary(url)
                            with container:
                                st.success(f'🤖 {res}')
                            if res:
                                if st.button('Speak Summary (beta)', type = 'primary'):
                                    output = api.run(
                                        "suno-ai/bark:b76242b40d67c76ab6742e987628a2a9ac019e11d56ab96c4e91ce03b79b2787",
                                        input = {"prompt": res, 
                                                'text_temp': 0.7,
                                                'waveform_temp': 0.7
                                                }
                                    )
                                    # print(output)
                                    # st.write(output)
                                    url = output['audio_out']
                                    audio = requests.get(url)
                                    audio_bytes = audio.content
                                    st.audio(audio_bytes, format="audio/mpeg")
                            counter+=1
        except:
            st.error('Error: Try changing the prompt or rerunning it! (note: try to keep 1 number in prompt)')

        # print('test', reading_list_df.iloc[:,1])
        # if summarize_add:
        #     val = reading_list_df.iloc[int(summarize_add), 1]
        #     print('val', val)
            
        #     summary(val)
        
        csv = reading_list_df.to_csv(index=False).encode('utf-8')
        markdown_content = dataframe_to_markdown(reading_list_df)
        # st.table(reading_list_df)
        col5, col6 = st.columns(2)
        with col5:
            st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name=f'reading_list_{timestamp}.csv',
                    mime='text/csv',
                )
        with col6:
            st.download_button(
                    label="Download as Markdown",
                    data=markdown_content,
                    file_name=f'reading_list_{timestamp}.md',
                    mime='text/markdown',
                )
        
        st.subheader('Email me this list!')
        email_sender = 'researchread375@gmail.com'
        email_receiver = st.text_input('My email:')
        subject = f'Reading list for {timestamp}'
        body = markdown_content
        # password = st.text_input('Password', type="password", disabled=True)  
        
        if st.button("Send Email"):
            try:
                msg = MIMEText(body)
                msg['From'] = email_sender
                msg['To'] = email_receiver
                msg['Subject'] = subject

                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login(st.secrets["email"]["gmail"], st.secrets["email"]["password"])
                server.sendmail(email_sender, email_receiver, msg.as_string())
                server.quit()

                st.success('Email sent successfully! 🚀')
            except Exception as e:
                st.error(f"Failed to send email: {e}")
        
        
        
    else:
        st.write("Your reading list is empty.")
    
    return urls

def serp():
    try:
        qs = Serp(serp_api_key)
        result = qs.get_related_pages(Paper_names[0])
        # print('result', result)
        first_key = next(iter(result))
        value = result[first_key]
        return value
    except:
        st.error('API Limit exhaust')

def arxiv(keyword):
    ap = Arxiv()
    if keyword:
        arxiv_result = ap.arxiv(keyword, max_pages = 1)
        return arxiv_result
    

def asm(keyword):
    if keyword:
        ac = ACM()
        acm_result = ac.acm(keyword, max_pages = 1)
        return acm_result

def semantic_scholar(keyword):
    if keyword:
        sc = Semantic_Scholar()
        sc_result = sc.ss(keyword, max_pages = 1)
        return sc_result

def acl(keyword):
    if keyword:
        paper_engine = Resp(serp_api_key)
        acl_result = paper_engine.acl(keyword, max_pages = 2)        
        return acl_result

def pmlr(keyword):
    if keyword:
        pmlr_result = paper_engine.pmlr(keyword, max_pages = 2)
        return pmlr_result

def neurips(keyword):
    if keyword:
        nips_result = paper_engine.nips(keyword, max_pages = 2)
        return nips_result



def app():
    publications = ['Google Scholar (api needed)', 'Arxiv', 'ASM', 'Semantic Scholar', 'ACL (api needed)', 'PMLR (api needed)', 'NeurIps (api needed)']
    
    key = st.sidebar.text_input("Enter a Keyword*:", placeholder='ML')
    if 'key' not in st.session_state:
        st.session_state.key = ''
        
    selected = st.sidebar.selectbox("Select a publication*:", publications, index=None)
    
    
    if 'selected' not in st.session_state:
        st.session_state.selected = ''
    
    # index = st.sidebar.text_input("Enter index number for summarization:")
    # index = st.sidebar.number_input('Select row index for summary(optional)', min_value=0, max_value=100, step=1)
    
    # reading_add = st.sidebar.text_area('Add indexes to reading list (comma seperated): ')
    # if index:
    # index = int(index)
    col1, col2 = st.columns(2)
    
    if 'reading_list' not in st.session_state:
        st.session_state.reading_list = []
    
    if 'res' not in st.session_state:
            st.session_state.res = ''
            
        
    if selected == 'Google Scholar (api needed)' and key:
        try:
            res = serp()
            res_cop = res.copy()
            
            for i, row in res_cop.iterrows():
                col1, col2 = st.columns([0.05, 0.95])
                with col1:
                    
                    if st.checkbox("", key=f"add{i}"):
                        st.session_state.reading_list.append(res_cop.iloc[i])
                with col2:
                    st.markdown(f"[{row['title']}]({row['link']})")
            # st.table(res)
                
            if st.sidebar.button("Add all to Reading List", type="primary"):
                for i, r in res_cop.iterrows():
                    st.session_state.reading_list.append(r)
                st.write("Added all to Reading List")
                
            # if reading_add:
            #     nums_list = reading_add.split(',')
            #     for i in nums_list:
            #         st.session_state.reading_list.append(res_cop.iloc[i])
            #     st.write("Added to Reading List") 
        except:
            if serp_api_key:
                st.write('')
            else:
                st.write('Please enter Google Scholar API')

    elif selected == 'Arxiv' and key:
        res = arxiv(key)
        # st.session_state.res = res
        res_cop = res.copy()
        for i, row in res_cop.iterrows():
            col1, col2 = st.columns([0.05, 0.95])
            with col1:
                
                if st.checkbox("", key=f"add{i}"):
                    st.session_state.reading_list.append(res_cop.iloc[i])
            with col2:
                st.markdown(f"[{row['title']}]({row['link']})")
            
        # st.table(res)
        
        if st.sidebar.button("Add all to Reading List", type="primary"):
            for i, r in res_cop.iterrows():
                st.session_state.reading_list.append(r)
            st.write("Added all to Reading List")
            
        # if reading_add:
        #         nums_list = reading_add.split(',')
        #         nums_list = [int(i) for i in nums_list]
        #         for i in nums_list:
        #             st.session_state.reading_list.append(res_cop.iloc[i])
        #         st.sidebar.write("Added to Reading List")
        
        
                
    
    elif selected == 'ASM' and key:
        st.session_state.selected = 'ASM'
        res = asm(key)
        st.session_state.res = res
        res_cop = res.copy()
        for i, row in res_cop.iterrows():
            col1, col2 = st.columns([0.05, 0.95])
            with col1:
                if st.checkbox("", key=f"add{i}"):
                    st.session_state.reading_list.append(res_cop.iloc[i])
            with col2:
                st.markdown(f"[{row['title']}]({row['link']})")
        # st.table(res)
        
        
        if st.sidebar.button("Add all to Reading List", type="primary"):
            for i, r in res_cop.iterrows():
                st.session_state.reading_list.append(r)
            st.write("Added all to Reading List")
        
        # if reading_add:
        #         nums_list = reading_add.split(',')
        #         nums_list = [int(i) for i in nums_list]
        #         for i in nums_list:
        #             st.session_state.reading_list.append(res_cop.iloc[i])
        #         st.write("Added to Reading List")
    
    elif selected == 'Semantic Scholar' and key:
        res = semantic_scholar(key)
        res_cop = res.copy()
        for i, row in res_cop.iterrows():
            col1, col2 = st.columns([0.05, 0.95])
            with col1:
                if st.checkbox("", key=f"add{i}"):
                    st.session_state.reading_list.append(res_cop.iloc[i])
            with col2:
                st.markdown(f"[{row['title']}]({row['link']})")
        # st.table(res)
        
            
        if st.sidebar.button("Add all to Reading List", type="primary"):
            for i, r in res_cop.iterrows():
                st.session_state.reading_list.append(r)
            st.write("Added all to Reading List")
        
        # if reading_add:
        #         nums_list = reading_add.split(',')
        #         nums_list = [int(i) for i in nums_list]
        #         for i in nums_list:
        #             st.session_state.reading_list.append(res_cop.iloc[i])
        #         st.write("Added to Reading List")
    
    elif selected == 'ACL (api needed)' and key:
        try:
            res = acl(key)
            res_cop = res.copy()
            for i, row in res_cop.iterrows():
                col1, col2 = st.columns([0.05, 0.95])
                with col1:
                    if st.checkbox("", key=f"add{i}"):
                        st.session_state.reading_list.append(res_cop.iloc[i])
                with col2:
                    st.markdown(f"[{row['title']}]({row['link']})")
            # st.table(res)
            
            
            if st.sidebar.button("Add all to Reading List", type="primary"):
                for i, r in res_cop.iterrows():
                    st.session_state.reading_list.append(r)
                st.write("Added all to Reading List")
            
            # if reading_add:
            #     nums_list = reading_add.split(',')
            #     nums_list = [int(i) for i in nums_list]
            #     for i in nums_list:
            #         st.session_state.reading_list.append(res_cop.iloc[i])
            #     st.write("Added to Reading List")
        except:
            if serp_api_key:
                st.write('')
            else:
                st.write('Please enter Google Scholar API')
            
            
    elif selected == 'PMLR (api needed)' and key:
        try:
            res = pmlr(key)
            res_cop = res.copy()
            for i, row in res_cop.iterrows():
                col1, col2 = st.columns([0.05, 0.95])
                with col1:
                    if st.checkbox("", key=f"add{i}"):
                        st.session_state.reading_list.append(res_cop.iloc[i])
                with col2:
                    st.markdown(f"[{row['title']}]({row['link']})")
            # st.table(res)
            
            if st.sidebar.button("Add all to Reading List", type="primary"):
                for i, r in res_cop.iterrows():
                    st.session_state.reading_list.append(r)
                st.write("Added all to Reading List")
            
            # if reading_add:
            #     nums_list = reading_add.split(',')
            #     nums_list = [int(i) for i in nums_list]
            #     for i in nums_list:
            #         st.session_state.reading_list.append(res_cop.iloc[i])
            #     st.write("Added to Reading List")
        except:
            if serp_api_key:
                st.write('')
            else:
                st.write('Please enter Google Scholar API')
            
    #Neurips 
    elif selected == 'NeurIps (api needed)' and key:
        try:
            res = neurips(key)
            res_cop = res.copy()
            for i, row in res_cop.iterrows():
                col1, col2 = st.columns([0.05, 0.95])
                with col1:
                    if st.checkbox("", key=f"add{i}"):
                        st.session_state.reading_list.append(res_cop.iloc[i])
                with col2:
                    st.markdown(f"[{row['title']}]({row['link']})")
            # st.table(res)
            
            if st.sidebar.button("Add all to Reading List", type="primary"):
                for i, r in res_cop.iterrows():
                    st.session_state.reading_list.append(r)
                st.write("Added all to Reading List")
            
            # if reading_add:
            #     nums_list = reading_add.split(',')
            #     nums_list = [int(i) for i in nums_list]
            #     for i in nums_list:
            #         st.session_state.reading_list.append(res_cop.iloc[i])
            #     st.write("Added to Reading List")
        except:
            if serp_api_key:
                st.write('')
            else:
                st.write('Please enter Google Scholar API')
    
    #Surprise me logic
    elif st.sidebar.button('Surprise Me!'):
        columns = ['title', 'link']
        temp_df = pd.DataFrame(columns=columns)
        research_topics = [
        "Artificial Intelligence","Machine Learning","Natural Language Processing","Data Science","Computer Vision","Robotics","Deep Learning","Big Data","Internet of Things","Cybersecurity","Bioinformatics","Reinforcement Learning","Quantum Computing",
        "Blockchain Technology","Cloud Computing","Human-visu Interaction","Autonomous Systems","Virtual Reality","Augmented Reality","Edge Computing","Digital Twins","Explainable AI",
        "Generative Adversarial Networks","Smart Cities","5G Technology","Biology","Neuroscience","Psychology","Genetics","Biotechnology","Cognitive Science","Evolutionary Biology",
        "Ecology","Molecular Biology","Behavioral Science","Developmental Biology","Psychopharmacology","Environmental Science","Biophysics","Microbiology","Cell Biology",
        "Psychometrics","Immunology","Neuroimaging","Cognitive Neuroscience","Psychiatry","Social Psychology","Evolutionary Psychology"
    ]

        key_s = random.choice(research_topics)
        arx = arxiv(key_s)
        temp_df = temp_df._append(arx, ignore_index=True)
        asm_v = asm(key_s)
        temp_df = temp_df._append(asm_v, ignore_index=True)
        semantic_s = semantic_scholar(key_s)
        temp_df = temp_df._append(semantic_s, ignore_index=True)
        # st.write(temp_df)
        
        sample = temp_df.sample(10)
        sample = sample.reset_index()
        # st.write(sample)
        st.success('Added to your reading!')
        for i, row in sample.iterrows():
            # col1, col2 = st.columns([0.05, 0.95])
            # with col1:
                # if st.checkbox("", key=f"select_{i}", label_visibility='collapsed'):
                    # st.write('FLAG')
            st.session_state.reading_list.append(sample.iloc[i])
            # with col2:
            st.markdown(f"[{row['title']}]({row['link']})")
                
                
        # if st.sidebar.button("Add all to Reading List", type="primary"):
        #     for i, r in sample.iterrows():
        #         st.session_state.reading_list.append(r)
        #     st.write("Added all to Reading List")
    

######
def main():
    
    PAGES = {
        "Home": app,
        "Reading List": reading_list
    }

    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]
    page()

if __name__ == "__main__":
    
    main()
    