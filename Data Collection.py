#!/usr/bin/env python
# coding: utf-8

# In[16]:


pip install chembl_webresource_client


# In[17]:


# Import necessary libraries
import pandas as pd
from chembl_webresource_client.new_client import new_client


# In[18]:


#Target searching for coronavirus
target = new_client.target
target_query = target.search('coronavirus')
targets = pd.DataFrame.from_dict(target_query)
targets


# In[19]:


#Selecting and retrieving bioactivity data 3C-like proteinase (5th entry)

selected_target =targets.target_chembl_id[4]
selected_target


# In[20]:


# Bioactivity for 3C-like proteinase are reported at as IC50 value

activity = new_client.activity
res = activity.filter(target_chembl_id = selected_target).filter(standrd_type = "IC50")

#Showing the content of the dataframe

df = pd.DataFrame.from_dict(res)
df


# In[21]:


#Only first three rows interested ,due to high volume

df.head(3)


# In[22]:


#Dataframe writing and naming

df.to_csv('bioactivity_data.csv', index = False)


# In[25]:


df2 = df[df.standard_value.notna()]
df2


# In[27]:


#Labeling compounds as active,inactive or intermediate

bioactivity_class = []
for i in df2.standard_value:
    if float(i) >= 10000:
        bioactivity_class.append("inactive")
    elif float(i) >= 1000:
        bioactivity_class.append("active")
    else:
        bioactivity_class.append("intermediate")
        


# In[28]:


#Iterating the molecule_chembl_id 

mol_cid = []
for i in df2.molecule_chembl_id:
    mol_cid.append(i)


# In[31]:


#Iterating over canonical_smiles to a list
canonical_smiles = []
for i in df2.canonical_smiles:
    canonical_smiles.append(i)


# In[32]:


#Iterating standard_value to a list
standard_value = []
for i in df2.standard_value:
    standard_value.append(i)
    


# In[33]:


#Combining the 4 lists into a dataframe

data_tuples = list(zip(mol_cid, canonical_smiles, bioactivity_class, standard_value))
df3 = pd.DataFrame( data_tuples, columns = ['molecule_chembl_id', 'canonical_smiles', 'bioactivity_class', 'standard_value'])


# In[34]:


df3


# In[39]:


#Concatinating as series than object

pd.concat([df3,pd.Series(bioactivity_class)], axis=1)


# In[40]:


#Creating a CSV file for preprocessed

df3.to_csv('bioactivity_preprocessed_data.csv', index=False)

