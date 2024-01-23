import streamlit as st
from datetime import datetime
import Util
import plotly.express as px

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

# select stocks by number of top stocks
st.sidebar.subheader('Pick stocks:')
st.sidebar.write('Would you like to pick your stocks or select a number of top weighted stocks from S&P 500?')
if_select = st.sidebar.selectbox('Method', ('Top weighted stocks', 'Pick my own stocks'))

if if_select == 'Top weighted stocks':
    # if pick top weighted stocks
    n_stocks = st.sidebar.number_input('Select number of top weighted stocks in S&P 500:', min_value=0, max_value=n_days, value=20, step=1)
    n_stocks = int(n_stocks)
    symbols = None
else:
    # if pick your own stocks
    symbols = st.sidebar.text_input('The symbols of stocks of your choice separated by comma (e.g. MSFT, AAPL): ', 'MSFT, AAPL, AMZN, NVDA, GOOGL, META, GOOG, TSLAUNH')
    symbols = symbols.split(", ")
    n_stocks = len(symbols)

st.sidebar.write(f'The number of stocks is {n_stocks}. At most {n_days} for computation accurary.')

# Time series plot
st.sidebar.subheader('Time series:')
st.sidebar.write('Would you like to plot the time series of daily variations (close - open) of these stocks? ')
if_ts = st.sidebar.selectbox('Yes or no', ('No', 'Yes'))
if if_ts == 'Yes':
    ts_sym = st.sidebar.text_input('Choose the symbols of the stocks you would like to plot separated by comma (e.g. MSFT, AAPL): ', 'MSFT, AAPL')
    ts_sym = ts_sym.split(", ")

# Correlation method
st.sidebar.subheader('Correlation method:')
st.sidebar.write('Would you like to compute a network by thresholding correlations?')
if_corr = st.sidebar.selectbox('Yes or no', ('Yes', 'No'))
threshold = st.sidebar.number_input('If so, what is the threshold: ', value=0.5)
st.sidebar.write(f'There will be a connection between two stocks if their correlation is bigger than {threshold}')

# Comments and creators
st.sidebar.markdown('''
---
Created by Coco Huang and Lulu Wang.
''')

# Title row
st.markdown('## Discover the hidden network and relationships among stocks')

# Data board row
st.markdown('### Data')
col1, col2, col3 = st.columns(3)
col1.metric("Number of stocks", n_stocks, '')
col2.metric("Start", start, '')
col3.metric("End", end, '')
X, embedding, names = Util.get_data(n_stocks,start_dt,end_dt,symbols=symbols)

# Glasso row
binary_adj = Util.glasso_adj(X)
n_edges = binary_adj.sum()
st.markdown('### [Glasso algorithm](%s) discovered latent stock network: %d connections' %('https://jerryfriedman.su.domains/ftp/glasso-bio.pdf',n_edges))
G = Util.create_G(binary_adj)
fig = Util.plot_network(G,embedding,names,'')
# Row C
st.plotly_chart(fig, use_container_width=True)

# Correlation row
if if_corr == 'Yes':
    binary_adj = Util.cov_adj(X,threshold)
    n_edges = binary_adj.sum()
    st.markdown(f'### Correlation method discovered latent stock network: {n_edges} connections.')
    G = Util.create_G(binary_adj)
    fig = Util.plot_network(G,embedding,names,'')
    # Row E
    st.plotly_chart(fig, use_container_width=True)

# Time series row
if if_ts == 'Yes':
    ts = X[ts_sym]
    fig = px.line(ts).update_layout(
        yaxis_title="Daily variation"
    )
    st.plotly_chart(fig, use_container_width=True)
