#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
# TfidfVectorizer - This is used to convert text data into numerical values
from sklearn.metrics.pairwise import cosine_similarity
import os as os


# In[6]:


display (os.getcwd())


# In[7]:


os.chdir('D:\\Intenship ML\\movie recomendation\\')
movies_data =pd.read_csv('movies.csv')
movies_data.head()


# In[8]:


display (movies_data.shape)


# In[9]:



selected_features = ['genres','keywords','tagline','cast','director']
print(selected_features)


# In[10]:


display (movies_data.info())


# In[11]:


display (movies_data.isna().sum())


# In[12]:


display (movies_data[selected_features].head())


# In[13]:


display (movies_data[selected_features].isna().sum())


# In[14]:


for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')
display (movies_data.head())


# In[15]:


combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
display (combined_features)


# In[16]:


vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
display (feature_vectors.shape)
print (feature_vectors)


# In[17]:


similarity = cosine_similarity(feature_vectors)
print  (similarity )


# In[18]:


display(similarity.shape)


# In[45]:


movie_name = input(' Enter your favourite movie name : ')


# In[46]:


list_of_all_titles = movies_data['title'].tolist()
print(list_of_all_titles)


# In[36]:


len(list_of_all_titles)


# In[47]:



find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
print(find_close_match)


# In[48]:



find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
print(find_close_match)


# In[49]:


close_match = find_close_match[0]
print(close_match)


# In[50]:


index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
print(index_of_the_movie)


# In[51]:


similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)


# In[52]:


len(similarity_score)


# In[53]:


sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
print(sorted_similar_movies)


# In[54]:


print('Movies suggested for you : \n')
i = 1
for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1


# In[ ]:




