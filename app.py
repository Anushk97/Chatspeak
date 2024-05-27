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
openai_api = st.sidebar.text_input("Enter your OpenAI API key (optional):")
if openai_api:
    prompt = st.sidebar.text_area('Custom prompt...')

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
    summarize_add = st.sidebar.text_input('Input number for summarization: ')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    container = st.container(border=True)
    
    def summary(url):
        # link = df['link'].iloc[index]
        # st.header('Summarize url')
        # url = st.text_input('Please enter your URL here:')
        try:
            if url:
                openai.api_key = openai_api
                if prompt:
                    completion = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "user",
                            "content": f'Access this url link: {url} and follow this prompt: {prompt}'
                        },
                    ],
                    )
                    # print(completion.choices[0].message.content)
                    response = completion.choices[0].message.content
                    return response
            
                else:
                    completion = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "user",
                            "content": f"Access this url link: {url} and generate a summary about it."
                        },
                    ],
                    )
                    # print(completion.choices[0].message.content)
                    response = completion.choices[0].message.content
                    return response
            
            else:
                ""
        except:
            st.error('Please enter OpenAI API', icon="🚨")

    # st.write(reading_list_df)
    if st.session_state.reading_list:
        if reading_list_df.empty:
        # st.write(st.session_state.reading_list)
            reading_list_df = pd.DataFrame(st.session_state.reading_list)
            
        reading_list_df._append(st.session_state.reading_list, ignore_index=True)
        reading_list_df.drop_duplicates('title', inplace=True)
        
        # reading_list_df.drop_duplicates()
        # st.write(reading_list_df)
        count = 0
        for i, row in reading_list_df.iterrows():
            count += 1
            st.markdown(f"{count}.[{row['title']}]({row['link']})")
        
        if summarize_add:
            for i, row in reading_list_df.iterrows():
                if i == int(summarize_add):
                    
                    url = row['link']
                    res = summary(url)
                    container.write(res)
        
        # print('test', reading_list_df.iloc[:,1])
        # if summarize_add:
        #     val = reading_list_df.iloc[int(summarize_add), 1]
        #     print('val', val)
            
        #     summary(val)
        
        csv = reading_list_df.to_csv(index=False).encode('utf-8')
        markdown_content = dataframe_to_markdown(reading_list_df)
        # st.table(reading_list_df)
        st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=f'reading_list_{timestamp}.csv',
                mime='text/csv',
            )
        
        st.download_button(
                label="Download as Markdown",
                data=markdown_content,
                file_name=f'reading_list_{timestamp}.md',
                mime='text/markdown',
            )
        
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
    
    key = st.sidebar.text_input("Enter a Keyword*:")
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
            

    if selected == 'Google Scholar (api needed)':
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

    elif selected == 'Arxiv':
        st.session_state.selected = 'Arxiv'
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
        
        
                
    
    elif selected == 'ASM':
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
    
    elif selected == 'Semantic Scholar':
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
    
    elif selected == 'ACL (api needed)':
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
            
            
    elif selected == 'PMLR (api needed)':
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
            
        
    elif selected == 'NeurIps (api needed)':
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
    