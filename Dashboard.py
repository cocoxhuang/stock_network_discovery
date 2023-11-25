import streamlit as st
from datetime import datetime
import Util

st.set_page_config(layout='wide', initial_sidebar_state='expanded')
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Header
st.sidebar.header('Stock Network Discovery Dashboard `Demo`')

# starting date
st.sidebar.subheader('Starting date:')
start = st.sidebar.text_input('Starting date (MM-DD-YYYY)', '01-01-2018')
st.sidebar.write('The current starting date is', start)

# ending date
st.sidebar.subheader('Ending date:')
end = st.sidebar.text_input('Ending date (MM-DD-YYYY)', '01-01-2023')
st.sidebar.write('The current ending date is', end)

# max number of stocks due to computational accurary of Glasso algo
start_dt = datetime.strptime(start, '%m-%d-%Y').date()
end_dt = datetime.strptime(end, '%m-%d-%Y').date()
n_days = int( (end_dt-start_dt).days * 0.69)

# number of stocks
st.sidebar.subheader('Number of stocks:')
n_stocks = st.sidebar.number_input('Select number of top weighted stocks in S&P 500:', min_value=0, max_value=n_days, value=20, step=1)
n_stocks = int(n_stocks)
st.sidebar.write(f'The number of stocks is {n_stocks}. At most {n_days} for computation accurary.')

# Correlation method
st.sidebar.subheader('Correlation method:')
st.sidebar.write('Would you like to compute a network by thresholding correlations?')
if_corr = st.sidebar.selectbox('Yes or no', ('Yes', 'No'))
threshold = st.sidebar.number_input('If so, what is the threshold: ', 0.5)
st.sidebar.write(f'There will be a connection between two stocks if their correlation is bigger than {threshold}')

st.sidebar.markdown('''
---
Created by Coco Huang and Lulu Wang.
''')

# Row A
st.markdown('### Data')
col1, col2, col3 = st.columns(3)
col1.metric("Number of top weighted stocks", n_stocks, '')
col2.metric("Start", start, '')
col3.metric("End", end, '')

X, embedding, names = Util.get_data(n_stocks,start_dt,end_dt)
binary_adj = Util.glasso_adj(X)
n_edges = binary_adj.sum()
# Row B
st.markdown('### [Glasso algorithm](%s) discovered latent stock network: %d connections' %('https://jerryfriedman.su.domains/ftp/glasso-bio.pdf',n_edges))
G = Util.create_G(binary_adj)
fig = Util.plot_network(G,embedding,names,'')
# Row C
st.plotly_chart(fig, use_container_width=True)

if if_corr == 'Yes':
    binary_adj = Util.cov_adj(X,threshold)
    n_edges = binary_adj.sum()
    # Row D
    st.markdown(f'### Correlation method discovered latent stock network: {n_edges} connections.')
    G = Util.create_G(binary_adj)
    fig = Util.plot_network(G,embedding,names,'')
    # Row E
    st.plotly_chart(fig, use_container_width=True)