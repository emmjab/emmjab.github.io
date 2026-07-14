---
layout: post        # or 'page' depending on your theme setup
title: "Jupyter Notebook for Lesson 4 US Patent Phrase Similarities"
date: 2026-07-13    # optional, helpful if placing in the _posts folder
permalink: /patent-nlp-notebook/
nav_exclude: true
---


# Training Transformers to Produce Patent Phrase Similarity Scores


```python
import fastai
from pathlib import Path
```


```python
import os
import shutil
import kagglehub
```

## Step 1: Grab the data from Kaggle
Actually there was a **Step 0**: read about the [data](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/data): what files are included, what do the columns mean and what kinds of values can we find in them, and what kind of metric kaggle is using for scores.


```python
# grab the data from kaggle - works because I have ~/.kaggle/access_token file
cache_path = kagglehub.competition_download('us-patent-phrase-to-phrase-matching')
```


```python
print("Path to competition files:", cache_path)
# Path to competition files: /Users/emma/.cache/kagglehub/competitions/us-patent-phrase-to-phrase-matching
```

    Path to competition files: /Users/emma/.cache/kagglehub/competitions/us-patent-phrase-to-phrase-matching



```python
# bring the dataset from local default kaggle cache to this notebook's dir
local_dest = Path("./us-patent-phrase-to-phrase-matching")

if not os.path.exists(local_dest):
    shutil.copytree(cache_path, local_dest)
    print(f"Files copied from cache to local folder: {local_dest}")
else:
    print("Folder already exists.")

print("Files:", os.listdir(local_dest))
```

    Folder already exists.
    Files: ['test.csv', 'train.csv', 'sample_submission.csv']



```python
import pandas as pd
```


```python
df_train = pd.read_csv(local_dest/'train.csv')
```

### What have we got?


```python
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>anchor</th>
      <th>target</th>
      <th>context</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>37d61fd2272659b1</td>
      <td>abatement</td>
      <td>abatement of pollution</td>
      <td>A47</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7b9652b17b68b7a4</td>
      <td>abatement</td>
      <td>act of abating</td>
      <td>A47</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36d72442aefd8232</td>
      <td>abatement</td>
      <td>active catalyst</td>
      <td>A47</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5296b0c19e1ce60e</td>
      <td>abatement</td>
      <td>eliminating process</td>
      <td>A47</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54c1e3b9184cb5b6</td>
      <td>abatement</td>
      <td>forest region</td>
      <td>A47</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
type(df_train['score'][0])
```




    numpy.float64




```python
df_train.dtypes
```




    id          object
    anchor      object
    target      object
    context     object
    score      float64
    dtype: object




```python
df_test = pd.read_csv(local_dest/'test.csv')
```


```python
df_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>anchor</th>
      <th>target</th>
      <th>context</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4112d61851461f60</td>
      <td>opc drum</td>
      <td>inorganic photoconductor drum</td>
      <td>G02</td>
    </tr>
    <tr>
      <th>1</th>
      <td>09e418c93a776564</td>
      <td>adjust gas flow</td>
      <td>altering gas flow</td>
      <td>F23</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36baf228038e314b</td>
      <td>lower trunnion</td>
      <td>lower locating</td>
      <td>B60</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1f37ead645e7f0c8</td>
      <td>cap component</td>
      <td>upper portion</td>
      <td>D06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>71a5b6ad068d531f</td>
      <td>neural stimulation</td>
      <td>artificial neural network</td>
      <td>H04</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_sample_submission = pd.read_csv(local_dest/'sample_submission.csv')
```


```python
df_sample_submission.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4112d61851461f60</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>09e418c93a776564</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36baf228038e314b</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1f37ead645e7f0c8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>71a5b6ad068d531f</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# i saw one instance of anchor='neural stimulation' in test.csv and was curious if/how it was represented in train.csv
# and what targets it mapped to; looks like there are 55 of them (from adding .count() to the end)
df_train[df_train['anchor'].str.contains('neural stimulation', na=False)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>anchor</th>
      <th>target</th>
      <th>context</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21037</th>
      <td>baba9c4f0962c4b2</td>
      <td>neural stimulation</td>
      <td>artificial neural network</td>
      <td>A61</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>21038</th>
      <td>abc28aee01d27890</td>
      <td>neural stimulation</td>
      <td>axons stimulation</td>
      <td>A61</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>21039</th>
      <td>2dc1b13a460ca64c</td>
      <td>neural stimulation</td>
      <td>cable like bundle</td>
      <td>A61</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>21040</th>
      <td>cb727a72c1a2700b</td>
      <td>neural stimulation</td>
      <td>deliver neural stimulation</td>
      <td>A61</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>21041</th>
      <td>0df7dc5df17955ea</td>
      <td>neural stimulation</td>
      <td>electrical stimulation</td>
      <td>A61</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>21042</th>
      <td>14b8d6154c6ca5a4</td>
      <td>neural stimulation</td>
      <td>nerve stimulation</td>
      <td>A61</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>21043</th>
      <td>6bd2e81d1aed66de</td>
      <td>neural stimulation</td>
      <td>nervous systemstimulation</td>
      <td>A61</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>21044</th>
      <td>11e4b59194412e3e</td>
      <td>neural stimulation</td>
      <td>neural communication</td>
      <td>A61</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>21045</th>
      <td>a43e849b746505ea</td>
      <td>neural stimulation</td>
      <td>neural control</td>
      <td>A61</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>21046</th>
      <td>56d138ca9233d484</td>
      <td>neural stimulation</td>
      <td>neural nerve</td>
      <td>A61</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>21047</th>
      <td>7473080515892c85</td>
      <td>neural stimulation</td>
      <td>neural network</td>
      <td>A61</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>21048</th>
      <td>d7d5de1e9905c03e</td>
      <td>neural stimulation</td>
      <td>neural stimulation device</td>
      <td>A61</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>21049</th>
      <td>d5f0884fbc5b5d43</td>
      <td>neural stimulation</td>
      <td>neural stimulation signal</td>
      <td>A61</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>21050</th>
      <td>95a09d3501a21a17</td>
      <td>neural stimulation</td>
      <td>neural stimulation therapy</td>
      <td>A61</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>21051</th>
      <td>39adff29cb9327fe</td>
      <td>neural stimulation</td>
      <td>neural tube</td>
      <td>A61</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>21052</th>
      <td>e0c998653413fb1d</td>
      <td>neural stimulation</td>
      <td>neuromodulation</td>
      <td>A61</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>21053</th>
      <td>481b8dc03287bea8</td>
      <td>neural stimulation</td>
      <td>neuromuscular electrical stimulation</td>
      <td>A61</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>21054</th>
      <td>7a326eba6e9f74ea</td>
      <td>neural stimulation</td>
      <td>neurostimulator</td>
      <td>A61</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>21055</th>
      <td>5b30ced7770eecc4</td>
      <td>neural stimulation</td>
      <td>parameter</td>
      <td>A61</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>21056</th>
      <td>6f6d9f597b737713</td>
      <td>neural stimulation</td>
      <td>peripheral nervous system</td>
      <td>A61</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>21057</th>
      <td>570d31a957c3c421</td>
      <td>neural stimulation</td>
      <td>pulses</td>
      <td>A61</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>21058</th>
      <td>4e05bed83a8b9434</td>
      <td>neural stimulation</td>
      <td>simulated neural networks</td>
      <td>A61</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>21059</th>
      <td>8751d0501c35d80b</td>
      <td>neural stimulation</td>
      <td>stimulation machine</td>
      <td>A61</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>21060</th>
      <td>8add2955dc0958c7</td>
      <td>neural stimulation</td>
      <td>transmit signal</td>
      <td>A61</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>21061</th>
      <td>f8724f396b800f6f</td>
      <td>neural stimulation</td>
      <td>anti stimulant composition</td>
      <td>H04</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>21062</th>
      <td>71a5b6ad068d531f</td>
      <td>neural stimulation</td>
      <td>artificial neural network</td>
      <td>H04</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>21063</th>
      <td>466eccc3c7bac37f</td>
      <td>neural stimulation</td>
      <td>convolutional neural network</td>
      <td>H04</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>21064</th>
      <td>1c32c055a8d4c9c0</td>
      <td>neural stimulation</td>
      <td>de energize electronic components</td>
      <td>H04</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>21065</th>
      <td>625e0202be8790f5</td>
      <td>neural stimulation</td>
      <td>de energize the laser</td>
      <td>H04</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>21066</th>
      <td>8c7d208c7602ff4f</td>
      <td>neural stimulation</td>
      <td>deliver neural stimulation</td>
      <td>H04</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>21067</th>
      <td>a6c2ab570f800398</td>
      <td>neural stimulation</td>
      <td>electrical nerve stimulation</td>
      <td>H04</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>21068</th>
      <td>87664d83b5eb5961</td>
      <td>neural stimulation</td>
      <td>electrical stimulation</td>
      <td>H04</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>21069</th>
      <td>77f55e0ca2f985c2</td>
      <td>neural stimulation</td>
      <td>electrical therapy</td>
      <td>H04</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>21070</th>
      <td>d0d3e8a06e00b90a</td>
      <td>neural stimulation</td>
      <td>implant</td>
      <td>H04</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>21071</th>
      <td>cebe5aeacde29cb6</td>
      <td>neural stimulation</td>
      <td>implant control</td>
      <td>H04</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>21072</th>
      <td>e0a8e1e950106bab</td>
      <td>neural stimulation</td>
      <td>implantable control</td>
      <td>H04</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>21073</th>
      <td>9f0833059e08e87d</td>
      <td>neural stimulation</td>
      <td>medical</td>
      <td>H04</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>21074</th>
      <td>dddb4b80bf87a534</td>
      <td>neural stimulation</td>
      <td>medical stimulation</td>
      <td>H04</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>21075</th>
      <td>2034a6557e5282e0</td>
      <td>neural stimulation</td>
      <td>nerve stimulation</td>
      <td>H04</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>21076</th>
      <td>b47f34efc77c0064</td>
      <td>neural stimulation</td>
      <td>neural signal</td>
      <td>H04</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>21077</th>
      <td>85cf3a591c40b071</td>
      <td>neural stimulation</td>
      <td>neural stimulation device</td>
      <td>H04</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>21078</th>
      <td>533b2d385f9f4142</td>
      <td>neural stimulation</td>
      <td>neural stimulation parameters</td>
      <td>H04</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>21079</th>
      <td>86a6c2ead9756d9a</td>
      <td>neural stimulation</td>
      <td>neural stimulation signal</td>
      <td>H04</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>21080</th>
      <td>843784b10781b1af</td>
      <td>neural stimulation</td>
      <td>neural stimulation therapy</td>
      <td>H04</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>21081</th>
      <td>2e9a5d79196fd5a6</td>
      <td>neural stimulation</td>
      <td>neural stimulator</td>
      <td>H04</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>21082</th>
      <td>41ae9c67de2e8165</td>
      <td>neural stimulation</td>
      <td>neural stimuli</td>
      <td>H04</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>21083</th>
      <td>e9d89aba0f739cde</td>
      <td>neural stimulation</td>
      <td>neurological stimulation</td>
      <td>H04</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>21084</th>
      <td>46d1e96f5505ac52</td>
      <td>neural stimulation</td>
      <td>neuroscience</td>
      <td>H04</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>21085</th>
      <td>d60cc886efad9890</td>
      <td>neural stimulation</td>
      <td>quantum simulation</td>
      <td>H04</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>21086</th>
      <td>28501c0820eba8fb</td>
      <td>neural stimulation</td>
      <td>reinforced neural network</td>
      <td>H04</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>21087</th>
      <td>4aa804d58c6deae1</td>
      <td>neural stimulation</td>
      <td>simulation hypothesis</td>
      <td>H04</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>21088</th>
      <td>677afc6fe4e08053</td>
      <td>neural stimulation</td>
      <td>stimulation</td>
      <td>H04</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>21089</th>
      <td>7e897f9e9d27ba46</td>
      <td>neural stimulation</td>
      <td>stimulation signal</td>
      <td>H04</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>21090</th>
      <td>77413f297bfcf041</td>
      <td>neural stimulation</td>
      <td>stimulation sites</td>
      <td>H04</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>21091</th>
      <td>b30688496db8abb9</td>
      <td>neural stimulation</td>
      <td>tissue stimulation</td>
      <td>H04</td>
      <td>0.50</td>
    </tr>
  </tbody>
</table>
</div>




```python
# how many unique contexts do we have? 106 for training data.
df_train['context'].sort_values().unique()
```




    array(['A01', 'A21', 'A22', 'A23', 'A24', 'A41', 'A43', 'A44', 'A45',
           'A46', 'A47', 'A61', 'A62', 'A63', 'B01', 'B02', 'B03', 'B05',
           'B07', 'B08', 'B21', 'B22', 'B23', 'B24', 'B25', 'B27', 'B28',
           'B29', 'B31', 'B32', 'B41', 'B44', 'B60', 'B61', 'B62', 'B63',
           'B64', 'B65', 'B66', 'B67', 'B81', 'C01', 'C02', 'C03', 'C04',
           'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14',
           'C21', 'C22', 'C23', 'C25', 'D01', 'D03', 'D04', 'D05', 'D06',
           'D21', 'E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E21', 'F01',
           'F02', 'F03', 'F04', 'F15', 'F16', 'F17', 'F21', 'F22', 'F23',
           'F24', 'F25', 'F26', 'F27', 'F28', 'F41', 'F42', 'G01', 'G02',
           'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11',
           'G16', 'G21', 'H01', 'H02', 'H03', 'H04', 'H05'], dtype=object)




```python
# 29 for test
df_test['context'].sort_values().unique()
```




    array(['A44', 'A61', 'A63', 'B01', 'B21', 'B22', 'B23', 'B60', 'B61',
           'B63', 'C07', 'C10', 'C12', 'C23', 'D06', 'E04', 'F01', 'F02',
           'F04', 'F16', 'F23', 'G01', 'G02', 'G05', 'G11', 'H02', 'H03',
           'H04', 'H05'], dtype=object)




```python
import numpy as np
```


```python
contexts = set(df_test['context'].unique().tolist())
contexts.update(df_train['context'].unique().tolist())
print(len(contexts))
np.array(sorted(list(contexts)))
```

    106





    array(['A01', 'A21', 'A22', 'A23', 'A24', 'A41', 'A43', 'A44', 'A45',
           'A46', 'A47', 'A61', 'A62', 'A63', 'B01', 'B02', 'B03', 'B05',
           'B07', 'B08', 'B21', 'B22', 'B23', 'B24', 'B25', 'B27', 'B28',
           'B29', 'B31', 'B32', 'B41', 'B44', 'B60', 'B61', 'B62', 'B63',
           'B64', 'B65', 'B66', 'B67', 'B81', 'C01', 'C02', 'C03', 'C04',
           'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14',
           'C21', 'C22', 'C23', 'C25', 'D01', 'D03', 'D04', 'D05', 'D06',
           'D21', 'E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E21', 'F01',
           'F02', 'F03', 'F04', 'F15', 'F16', 'F17', 'F21', 'F22', 'F23',
           'F24', 'F25', 'F26', 'F27', 'F28', 'F41', 'F42', 'G01', 'G02',
           'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11',
           'G16', 'G21', 'H01', 'H02', 'H03', 'H04', 'H05'], dtype='<U3')




```python
# total rows: 36473
len(df_train)
```




    36473




```python
# how many unique anchors? 733 // and how many unique targets? 29,340
# in 106 unique contexts
print(len(df_train['anchor'].unique()))
print(len(df_train['target'].unique()))
print(len(df_train['context'].unique()))
```

    733
    29340
    106



```python
print(len(df_test['anchor'].unique()))
print(len(df_test['target'].unique()))
print(len(df_test['context'].unique()))
```

    34
    36
    29



```python
print(len(df_test))
print(len(df_sample_submission))
```

    36
    36



```python
# ok so the idea is to submit the scores for the ids in test.csv in the sample_submission.csv?
# all of the same ids are in test as in sample_submission
df_sample_submission[df_sample_submission['id'].str.contains('71a5b6ad068d531f', na=False)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>71a5b6ad068d531f</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### The scores are float64, which are apparentlly overkill


```python
print(df_train['score'].dtypes)
df_train['score'] = df_train['score'].astype('float32')
print(df_train['score'].dtypes)
```

    float64
    float32


## Step 2: Prepare the dds = ('train' as subset of train.csv, 'test' as validation as subset of train.csv), and test_ds = (test.csv) for the transformer model


```python
df_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>anchor</th>
      <th>target</th>
      <th>context</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>37d61fd2272659b1</td>
      <td>abatement</td>
      <td>abatement of pollution</td>
      <td>A47</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7b9652b17b68b7a4</td>
      <td>abatement</td>
      <td>act of abating</td>
      <td>A47</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36d72442aefd8232</td>
      <td>abatement</td>
      <td>active catalyst</td>
      <td>A47</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5296b0c19e1ce60e</td>
      <td>abatement</td>
      <td>eliminating process</td>
      <td>A47</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54c1e3b9184cb5b6</td>
      <td>abatement</td>
      <td>forest region</td>
      <td>A47</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>36468</th>
      <td>8e1386cbefd7f245</td>
      <td>wood article</td>
      <td>wooden article</td>
      <td>B44</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>36469</th>
      <td>42d9e032d1cd3242</td>
      <td>wood article</td>
      <td>wooden box</td>
      <td>B44</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>36470</th>
      <td>208654ccb9e14fa3</td>
      <td>wood article</td>
      <td>wooden handle</td>
      <td>B44</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>36471</th>
      <td>756ec035e694722b</td>
      <td>wood article</td>
      <td>wooden material</td>
      <td>B44</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>36472</th>
      <td>8d135da0b55b8c88</td>
      <td>wood article</td>
      <td>wooden substrate</td>
      <td>B44</td>
      <td>0.50</td>
    </tr>
  </tbody>
</table>
<p>36473 rows × 5 columns</p>
</div>




```python
df_train.describe(include='object')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>anchor</th>
      <th>target</th>
      <th>context</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>36473</td>
      <td>36473</td>
      <td>36473</td>
      <td>36473</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>36473</td>
      <td>733</td>
      <td>29340</td>
      <td>106</td>
    </tr>
    <tr>
      <th>top</th>
      <td>37d61fd2272659b1</td>
      <td>component composite coating</td>
      <td>composition</td>
      <td>H01</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>152</td>
      <td>24</td>
      <td>2186</td>
    </tr>
  </tbody>
</table>
</div>



### Add a new column called 'input' which concatenates the anchor, target, and context


```python
# putting "context" first in the text order is apparently useful for cross validation error
df_train['input'] = 'TEXT1: ' + df_train.context + '; TEXT2: ' + df_train.target + '; ANC1: ' + df_train.anchor
```


```python
### Create a Dataset object
from datasets import Dataset,DatasetDict

ds = Dataset.from_pandas(df_train)
```


```python
ds
```




    Dataset({
        features: ['id', 'anchor', 'target', 'context', 'score', 'input'],
        num_rows: 36473
    })



### Choose the model, and then set up the same tokenizer used for text that pretrained the model


```python
model_nm = 'microsoft/deberta-v3-small'
```


```python
from transformers import AutoModelForSequenceClassification,AutoTokenizer
tokz = AutoTokenizer.from_pretrained(model_nm)
```


```python
print(list(tokz.vocab.keys())[:10])
print(list(tokz.vocab.values())[:10])
```

    ['▁Thankfully', '▁919', '▁sh', '▁docks', '▁mechanism', 'evil', '▁Rubio', '▁preparedness', '▁Exchange', '▁cafeteria']
    [9651, 92279, 111719, 85442, 12501, 82185, 8349, 54165, 98976, 121420]



```python
# set up a tokenization function to 
def tok_func(x): return tokz(x["input"])
```


```python
tok_ds = ds.map(tok_func, batched=True)
```


    Map:   0%|          | 0/36473 [00:00<?, ? examples/s]



```python
tok_ds
```




    Dataset({
        features: ['id', 'anchor', 'target', 'context', 'score', 'input', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 36473
    })



### Rename the 'score' to 'labels' because the transformers expects the learned column to be called that


```python
tok_ds = tok_ds.rename_columns({'score':'labels'})
```

### Build a validation set from the training data -- which happens to be called 'test'


```python
dds = tok_ds.train_test_split(0.25, seed=42)
dds
```




    DatasetDict({
        train: Dataset({
            features: ['id', 'anchor', 'target', 'context', 'labels', 'input', 'input_ids', 'token_type_ids', 'attention_mask'],
            num_rows: 27354
        })
        test: Dataset({
            features: ['id', 'anchor', 'target', 'context', 'labels', 'input', 'input_ids', 'token_type_ids', 'attention_mask'],
            num_rows: 9119
        })
    })



### Create a dataset from the test.csv data for the actual test (not validation) data


```python
df_test['input'] = 'TEXT1: ' + df_test.context + '; TEXT2: ' + df_test.target + '; ANC1: ' + df_test.anchor
test_ds = Dataset.from_pandas(df_test).map(tok_func, batched=True)
```


    Map:   0%|          | 0/36 [00:00<?, ? examples/s]


## Step 3: Training the Model


```python
from transformers import TrainingArguments,Trainer
```


```python
# set batch size and number of epochs
# Gemini advice: "For fine-tuning or inference with microsoft/deberta-v3-small on a unified memory Mac, 
# use a per-device batch size of 8 to 16 for training, or 16 to 64 for inference. 
# Leverage Apple's Metal Performance Shaders (MPS) via the Hugging Face Transformers backend to optimize memory.
# If you encounter memory errors with longer sequences, use Gradient Accumulation 
# (e.g., gradient_accumulation_steps=4 with a batch size of 8 gives an effective batch size of 32) 
# to hit standard target batch sizes without pushing memory pressure too high."

# GPU defaults
#bs = 128
#epochs = 4
#lr = 8e-5

# ---- above the line are the values in the notebook by Jeremy Howard in the fastai course
# ---- below the line are values that gemini suggested for the TrainingArguments object

# Pass the per-device batch sizes into TrainingArguments:
# per_device_train_batch_size=8,   # Batch size per device during training
# per_device_eval_batch_size=16,   # Batch size per device during evaluation
# & 
# Optional: Combine with gradient accumulation to mimic a larger batch size
# gradient_accumulation_steps=4,   # Total effective train batch size = 8 * 4 = 32
# use_mps_device=True,             # Ensures it targets your Mac M4 GPU
```


```python
# These are the values in the notebook by Jeremy Howard in the fastai course
# args = TrainingArguments('outputs', 
#                          learning_rate=lr, 
#                          warmup_ratio=0.1, 
#                          lr_scheduler_type='cosine', 
#                          fp16=True,
#                          evaluation_strategy="epoch", 
#                          per_device_train_batch_size=bs, 
#                          per_device_eval_batch_size=bs*2,
#                          num_train_epochs=epochs, 
#                          weight_decay=0.01, 
#                          report_to='none')

# These are arguments that Gemini suggested for later libraries and Macbook Pro M4
#bs = 8       # Safe training batch size for deberta-v3-small on 24GB RAM
lr = 3e-5     # Recommended learning rate
epochs = 3    # Balanced epoch count to prevent overfitting

# ChatGPT recommendations after the kernel died on one run
# per_device_train_batch_size=2
# per_device_eval_batch_size=4
# gradient_accumulation_steps=8
# dataloader_pin_memory=False
# dataloader_num_workers=0
# fp16=False
# bf16=False

bs = 2

args = TrainingArguments('outputs', 
                         learning_rate=lr, 
                         warmup_steps=0.1, # new term; old was warmup_ratio=0.1, 
                         lr_scheduler_type='cosine', 

                         # --- MAC M4 SPECIFIC OPTIMIZATIONS ---
                         bf16=False,                         # Use Bfloat16 mixed precision (M4 native & stable) #was true
                         fp16=False,                         # # was unset, adding after kernel died on one run
                         #use_mps_device=True,               # default now to force Hugging Face to use Apple's Metal backend
                         dataloader_num_workers=0,           # Utilizes Mac's high-efficiency CPU cores for data loading (if =2)
                         dataloader_pin_memory=False,        # something for GPUs
                         # -------------------------------------

                         logging_strategy="steps",
                         logging_steps=20,
                         
                         eval_strategy="epoch", 
                         per_device_train_batch_size=bs, 
                         per_device_eval_batch_size=bs*2,
                         gradient_accumulation_steps=8, # adding after kernel died on one run
                         num_train_epochs=epochs, 
                         weight_decay=0.01, 
                         report_to='none')
```


```python
# args for "loading the best model at the end", e.g. if you have too many training steps and you start to overfit, and you want to
# submit your test data labels from the best model
    # eval_strategy="epoch",
    # save_strategy="epoch",
    # load_best_model_at_end=True,
    # metric_for_best_model="pearson",
    # greater_is_better=True,
```

### Set up compute_metrics for each run

from ChatGPT:
- Pearson is the metric that determines performance on this task.
- Spearman tells you whether the model is getting the ranking of similarities right, even if the absolute calibration is a bit off.
- RMSE helps diagnose whether two models with similar Pearson scores differ in how close their predicted values are to the targets.


```python
# setting up compute_metrics
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = np.asarray(predictions).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    return {
        "pearson": pearsonr(labels, predictions).statistic,
        "rmse": np.sqrt(mean_squared_error(labels, predictions)),
    }

# ChatGPT: the main red flags would be NaN, Pearson staying near zero, or predictions collapsing to nearly one constant value.
```


```python
import torch
# set up the model with pretrained values
# if num_labels = 1, Regression
#                    outputs single continuous number and uses MSELoss (Mean Squared Error) 
#                    to measure how close your model's prediction is to a target number.
# if num_labels > 1, Classificaton
#                    outputs multiple prediction scores (logits), uses CrossEntropyLoss 
#                    to handle discrete categories (like predicting 0, 1, or 2)

model = AutoModelForSequenceClassification.from_pretrained(model_nm, 
                                                           num_labels=1,
                                                           problem_type="regression", # this is implied by num_labels=1
                                                           dtype=torch.float32,)      # adding after kernel failure
```


    Loading weights:   0%|          | 0/102 [00:00<?, ?it/s]


    [transformers] [1mDebertaV2ForSequenceClassification LOAD REPORT[0m from: microsoft/deberta-v3-small
    Key                                     | Status     | 
    ----------------------------------------+------------+-
    mask_predictions.dense.bias             | UNEXPECTED | 
    mask_predictions.LayerNorm.weight       | UNEXPECTED | 
    mask_predictions.LayerNorm.bias         | UNEXPECTED | 
    mask_predictions.classifier.bias        | UNEXPECTED | 
    lm_predictions.lm_head.dense.weight     | UNEXPECTED | 
    lm_predictions.lm_head.LayerNorm.bias   | UNEXPECTED | 
    lm_predictions.lm_head.bias             | UNEXPECTED | 
    lm_predictions.lm_head.LayerNorm.weight | UNEXPECTED | 
    lm_predictions.lm_head.dense.bias       | UNEXPECTED | 
    mask_predictions.dense.weight           | UNEXPECTED | 
    mask_predictions.classifier.weight      | UNEXPECTED | 
    classifier.weight                       | MISSING    | 
    pooler.dense.weight                     | MISSING    | 
    pooler.dense.bias                       | MISSING    | 
    classifier.bias                         | MISSING    | 
    
    Notes:
    - UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
    - MISSING:	those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.



```python
# set up the Trainer (this is called a Learner in the fastai library)
trainer = Trainer(model, 
                  args, 
                  train_dataset=dds['train'], 
                  eval_dataset=dds['test'],
                  processing_class=tokz, # tokenizer is now called processing_class
                  compute_metrics=compute_metrics)
```


```python
# for debugging
# import torch

# batch = next(iter(trainer.get_train_dataloader()))
# batch = {
#     key: value.to(model.device)
#     for key, value in batch.items()
# }

# model.eval()
# with torch.no_grad():
#     output = model(**batch)

# print(output.logits.shape)
# print(output.loss)
# # results in
# # torch.Size([2, 1])
# # tensor(0.6705, device='mps:0')
```


```python
print(tokz.name_or_path)

print(tokz.special_tokens_map)

print(model.config)
```

    microsoft/deberta-v3-small
    {'bos_token': '[CLS]', 'eos_token': '[SEP]', 'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}
    DebertaV2Config {
      "attention_probs_dropout_prob": 0.1,
      "bos_token_id": null,
      "dtype": "float32",
      "eos_token_id": null,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 768,
      "id2label": {
        "0": "LABEL_0"
      },
      "initializer_range": 0.02,
      "intermediate_size": 3072,
      "label2id": {
        "LABEL_0": 0
      },
      "layer_norm_eps": 1e-07,
      "legacy": true,
      "max_position_embeddings": 512,
      "max_relative_positions": -1,
      "model_type": "deberta-v2",
      "norm_rel_ebd": "layer_norm",
      "num_attention_heads": 12,
      "num_hidden_layers": 6,
      "pad_token_id": 0,
      "pooler_dropout": 0.0,
      "pooler_hidden_act": "gelu",
      "pooler_hidden_size": 768,
      "pos_att_type": [
        "p2c",
        "c2p"
      ],
      "position_biased_input": false,
      "position_buckets": 256,
      "problem_type": "regression",
      "relative_attention": true,
      "share_att_key": true,
      "tie_word_embeddings": true,
      "transformers_version": "5.9.0",
      "type_vocab_size": 0,
      "use_cache": false,
      "vocab_size": 128100
    }
    



```python
print(model.config.bos_token_id)
print(model.config.eos_token_id)

print(tokz.bos_token_id)
print(tokz.eos_token_id)
```

    None
    None
    1
    2



```python
trainer.train()
```

    [transformers] The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'eos_token_id': 2, 'bos_token_id': 1}.




    <div>

      <progress value='5130' max='5130' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [5130/5130 1:05:34, Epoch 3/3]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Pearson</th>
      <th>Rmse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>0.218448</td>
      <td>0.027939</td>
      <td>0.795433</td>
      <td>0.167150</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.159067</td>
      <td>0.023375</td>
      <td>0.823309</td>
      <td>0.152890</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.117297</td>
      <td>0.022655</td>
      <td>0.829435</td>
      <td>0.150516</td>
    </tr>
  </tbody>
</table><p>



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]





    TrainOutput(global_step=5130, training_loss=0.22436229297292162, metrics={'train_runtime': 3935.5088, 'train_samples_per_second': 20.852, 'train_steps_per_second': 1.304, 'total_flos': 431993702403300.0, 'train_loss': 0.22436229297292162, 'epoch': 3.0})



### The trained model, 'trainer' is above!


```python
preds = trainer.predict(test_ds).predictions.astype(float)
preds
```








    array([[ 0.50793177],
           [ 0.69482702],
           [ 0.53050959],
           [ 0.3323034 ],
           [-0.01405059],
           [ 0.54094654],
           [ 0.49185976],
           [-0.06247338],
           [ 0.286567  ],
           [ 1.07577682],
           [ 0.24871875],
           [ 0.25795841],
           [ 0.7375955 ],
           [ 0.71554035],
           [ 0.75742549],
           [ 0.43514419],
           [ 0.28614819],
           [-0.04117407],
           [ 0.58906227],
           [ 0.27050924],
           [ 0.38998127],
           [ 0.26151669],
           [ 0.05747152],
           [ 0.23423766],
           [ 0.57260197],
           [-0.04425101],
           [-0.04549832],
           [-0.04818187],
           [-0.04342794],
           [ 0.65266436],
           [ 0.35751534],
           [ 0.02258844],
           [ 0.69498122],
           [ 0.49264497],
           [ 0.39684594],
           [ 0.23492719]])




```python
### The values have to be between 0 and 1!!!
preds = np.clip(preds, 0, 1)
```


```python
preds
```




    array([[0.50793177],
           [0.69482702],
           [0.53050959],
           [0.3323034 ],
           [0.        ],
           [0.54094654],
           [0.49185976],
           [0.        ],
           [0.286567  ],
           [1.        ],
           [0.24871875],
           [0.25795841],
           [0.7375955 ],
           [0.71554035],
           [0.75742549],
           [0.43514419],
           [0.28614819],
           [0.        ],
           [0.58906227],
           [0.27050924],
           [0.38998127],
           [0.26151669],
           [0.05747152],
           [0.23423766],
           [0.57260197],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.65266436],
           [0.35751534],
           [0.02258844],
           [0.69498122],
           [0.49264497],
           [0.39684594],
           [0.23492719]])




```python
preds.reshape(-1)
```




    array([0.50793177, 0.69482702, 0.53050959, 0.3323034 , 0.        ,
           0.54094654, 0.49185976, 0.        , 0.286567  , 1.        ,
           0.24871875, 0.25795841, 0.7375955 , 0.71554035, 0.75742549,
           0.43514419, 0.28614819, 0.        , 0.58906227, 0.27050924,
           0.38998127, 0.26151669, 0.05747152, 0.23423766, 0.57260197,
           0.        , 0.        , 0.        , 0.        , 0.65266436,
           0.35751534, 0.02258844, 0.69498122, 0.49264497, 0.39684594,
           0.23492719])




```python
submission = df_test[["id"]].copy()
submission["score"] = preds.reshape(-1)
submission.to_csv("submission_july13.csv", index=False)
```


```python
import datasets

submission = datasets.Dataset.from_dict({
    'id': test_ds['id'],
    'score': preds.reshape(-1)
})

submission.to_csv('submission_july13.csv', index=False)
```


    Creating CSV from Arrow format:   0%|          | 0/1 [00:00<?, ?ba/s]





    1192



## Trying to figure out how to submit the trained model and its predictions to kaggle competition...


```python
# ok i think this torch method isn't right for this notebook; chatgpt says you need to save tokens too
#torch.save(model.state_dict(), 'model.pth')

#the trainer has a save_model function
trainer.save_model("./patent_model.pth")
```


    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



```python
import shutil
shutil.make_archive("./patent_model", "zip", "./patent_model.pth")
```




    '/Users/emma/Documents/file_cabinet/people_projects/emma/deeplearning/nlp/patent_model.zip'



### Apparently we have to upload this on kaggle to my private model
https://www.kaggle.com/work/models

Used to be called "datasets", I think

The upload worked, it's here:  https://www.kaggle.com/models/emmjab/trained-patent-similarity-transformer-model

## We can make a notebook for the kaggle competition
If we do this from the competition page, it automatically associates the datasets for the competition with the notebook, and a little "copy" button next to the files to get the correct path.

We have to add the model to the notebook -- i think we have to first associate the model with the competition from the private workspace page (or at least my model didn't seem to show up in the search for adding data to the notebook until i did this). When the model is added, it also has the copy path button.

We need to then make a whole notebook with all of our import library to
- load in our trained model & tokens
- load in the test data from the competition
- run the prediction function on text.csv data from the competition (which we have just done locally)
- generate the submissions.csv (which we had done locally to, but we weren't allowed to just upload this data by itself)
