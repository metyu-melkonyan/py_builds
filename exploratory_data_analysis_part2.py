#!/usr/bin/env python
# coding: utf-8

# In[8]:


pip install chembl_webresource_client


# In[9]:


#Importing the libraries requried

import pandas as pd 
from chembl_webresource_client.new_client import new_client


# In[10]:


#Target search for coronavirus

target = new_client.target
target_query = target.search('coronavirus')
targets = pd.DataFrame.from_dict(target_query)
targets


# In[11]:


#Selecting the fifth entry

selected_target = targets.target_chembl_id[4]
selected_target


# In[12]:


# Bioactivity for 3C-like proteinase are reported at as IC50 value

activity = new_client.activity
res = activity.filter(target_chembl_id = selected_target).filter(standrd_type = "IC50")

#Showing the content of the dataframe

df = pd.DataFrame.from_dict(res)
df


# In[15]:


#savng results to bioactivity to CSV

df.to_csv('bioactivity_data_raw.csv', index=False)


# In[16]:


df2 = df[df.standard_value.notna()]
df2


# In[17]:


#Labeling compounds as active,inactive or intermediate

bioactivity_class = []
for i in df2.standard_value:
    if float(i) >= 10000:
        bioactivity_class.append("inactive")
    elif float(i) >= 1000:
        bioactivity_class.append("active")
    else:
        bioactivity_class.append("intermediate")
        


# In[18]:


#Combining the 3 columns of molecule_chembl_id, canonical _smiles, standard_value

selection = ['molecule_chembl_id','canonical_smiles','standard_value']
df3 = df2[selection]
df3


# In[20]:


#Combining the bioactivity_class into a DataFrame

bioactivity_class = pd.Series(bioactivity_class, name='bioactivity_class')
df4 = pd.concat([df3, bioactivity_class], axis = 1)
df4


# In[22]:


#Saving the dataframe to a CSV file

df4.to_csv('bioactivity_data_preprocessed.csv', index=False)


# In[ ]:


#Part 1 overview
downloaded the dataset including molecule names and the corresponding smiles notation about chemical structure


# In[28]:


#Installing conda and rd kit

get_ipython().system(' wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh')
get_ipython().system(' chmod +x Miniconda3-py37_4.8.2-Linux-x86_64.sh')
get_ipython().system(' bash ./Miniconda3-py37_4.8.2-Linux-x86_64.sh -b -f -p /usr/local')
get_ipython().system(' conda install -c rdkit rdkit -y')
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages/') 


# In[29]:


import pandas as od
df = pd.read_csv('bioactivity_data_preprocessed.csv')


# In[33]:


#Imprting  the Lipinski Descriptors

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski


# In[31]:


#Lipinski function

def lipinski(smiles, verbose=False):

    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem) 
        moldata.append(mol)
       
    baseData= np.arange(1,1)
    i=0  
    for mol in moldata:        
       
        desc_MolWt = Descriptors.MolWt(mol)
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_NumHDonors = Lipinski.NumHDonors(mol)
        desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)
           
        row = np.array([desc_MolWt,
                        desc_MolLogP,
                        desc_NumHDonors,
                        desc_NumHAcceptors])   
    
        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
        i=i+1      
    
    columnNames=["MW","LogP","NumHDonors","NumHAcceptors"]   
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)
    
    return descriptors


# In[32]:


#Applying the custom function 

df_lipinski = lipinski(df.canonical_smiles)


# In[ ]:


# Dataframe overview: LogP will tell about solubility
df_lipinski


# In[ ]:


#Converting IC50 to PIC50 : original IC50 has uneven distribution of the data points. To make it even weneed PIC50 than IC50


# In[ ]:


#PIC50

import numpy as np

def pIC50(input):
    pIC50 = []

    for i in input['standard_value_norm']:
        molar = i*(10**-9) # Converts nM to M
        pIC50.append(-np.log10(molar))

    input['pIC50'] = pIC50
    x = input.drop('standard_value_norm', 1)
        
    return x


# In[37]:


#Combined dataframe overview

df_combined.standard_value.describe()


# In[38]:


#Positive values are better for interpretations than
-np.log10( (10**-9)* 100000000 )


# In[39]:


#This will cause problem in interpretation due to negative value

-np.log10( (10**-9)* 10000000000 )


# In[40]:


#Defining the function

def norm_value(input):
    norm = []

    for i in input['standard_value']:
        if i > 100000000:
          i = 100000000
        norm.append(i)

    input['standard_value_norm'] = norm
    x = input.drop('standard_value', 1)
        
    return x


# In[36]:


#Performing the norm value

df_norm = norm_value(df_combined)
df_norm


# In[41]:


# Removing the 'intermediate' bioactivity class

df_2class = df_final[df_final.bioactivity_class !='intermediate']
df_2class


# In[42]:


# Explatory Data Analysis (Chemical Space Analysis) via Lipinski descriptors

import seaborn as sns
sns.set(style='ticks')
import matplotlib.pyplot as plt


# In[ ]:


#Simple frequency plot of 2 bioactivity classes

plt.figure(figsize=(5.5, 5.5))

sns.countplot(x='bioactivity_class', data=df_2class, edgecolor='black')

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')

plt.savefig('plot_bioactivity_class.pdf')


# In[ ]:


#Scatter plot of  bioactivity class

plt.figure(figsize=(5.5, 5.5))

sns.scatterplot(x='MW', y='LogP', data=df_2class, hue='bioactivity_class', size='pIC50', edgecolor='black', alpha=0.7)

plt.xlabel('MW', fontsize=14, fontweight='bold')
plt.ylabel('LogP', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
plt.savefig('plot_MW_vs_LogP.pdf')


# In[ ]:


# Box plot of bioactivity_class with pIC50 value

plt.figure(figsize=(5.5, 5.5))

sns.boxplot(x = 'bioactivity_class', y = 'pIC50', data = df_2class)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('pIC50 value', fontsize=14, fontweight='bold')

plt.savefig('plot_ic50.pdf')


# In[ ]:


Statistical analysis | Mann_Whitney U test

def mannwhitney(descriptor, verbose=False):
  # https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/
  from numpy.random import seed
  from numpy.random import randn
  from scipy.stats import mannwhitneyu

# seed the random number generator
  seed(1)

# actives and inactives
  selection = [descriptor, 'bioactivity_class']
  df = df_2class[selection]
  active = df[df.bioactivity_class == 'active']
  active = active[descriptor]

  selection = [descriptor, 'bioactivity_class']
  df = df_2class[selection]
  inactive = df[df.bioactivity_class == 'inactive']
  inactive = inactive[descriptor]

# compare samples
  stat, p = mannwhitneyu(active, inactive)
  #print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret
  alpha = 0.05
  if p > alpha:
    interpretation = 'Same distribution (fail to reject H0)'
  else:
    interpretation = 'Different distribution (reject H0)'
  
  results = pd.DataFrame({'Descriptor':descriptor,
                          'Statistics':stat,
                          'p':p,
                          'alpha':alpha,
                          'Interpretation':interpretation}, index=[0])
  filename = 'mannwhitneyu_' + descriptor + '.csv'
  results.to_csv(filename)

  return results


# In[ ]:


mannwhitney('pIC50')


# In[ ]:


# Plotting MW results

plt.figure(figsize=(5.5, 5.5))

sns.boxplot(x = 'bioactivity_class', y = 'MW', data = df_2class)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('MW', fontsize=14, fontweight='bold')

plt.savefig('plot_MW.pdf')


# In[ ]:


#Finally zipping the results

zip -r results.zip . -i *.csv *.pdf

