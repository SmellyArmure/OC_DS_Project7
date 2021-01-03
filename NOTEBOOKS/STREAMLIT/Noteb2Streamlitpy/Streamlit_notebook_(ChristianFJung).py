#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[8]:


import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd


# COUCOU

# In[9]:


df_online=pd.read_csv("https://raw.githubusercontent.com/fivethirtyeight/data/master/media-mentions-2020/online_weekly.csv")
df_online.head()


# In[10]:


dfO= df_online[['name','matched_stories']]

dfO= dfO.groupby("name")["matched_stories"].sum()
dfO = pd.DataFrame(dfO)

dfToShow= dfO.sort_values("matched_stories", ascending=False)
dfToShow.head()


# import streamlit as st
# st.bar_chart(dfToShow)
