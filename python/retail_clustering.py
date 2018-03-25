# -*- coding: utf-8 -*-
"""
Clustering of monthly retail sales of jewelry

Created on Sat Mar 10 07:25:56 2018

@author: dli
"""

import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

os.chdir(r"C:\sync\Dropbox\training\quan\retail\python")

def read_raw_data():
    '''
    Read raw files and generate a data frame 
    
    Returns:
    df_all - a data frame with columns ['item', 'item_desc', 'unit_retail', 'color', 'pos_stores', 'pos_qty',
             'wm_month', 'months', 'interested']
    '''
    # read raw data file
    filepath = r'..\data\sales_projection.csv'
    df_all = pd.read_csv(filepath)
    df_all.columns = ['item', 'item_desc', 'unit_retail', 'color', 'pos_stores', 'pos_qty', 'wm_month']
    
    # add items interested
    df_item_interested = pd.read_csv(r'..\data\item_to_check.csv')
    df_item_interested.columns = ['item']
    df_item_interested['interested']='Yes'
    df_all = pd.merge(df_all, df_item_interested, how='left', left_on='item', right_on='item')
    df_all['interested'] = df_all['interested'].fillna('No')
    
    # check the counts of month when there is sale per item
    months_count = df_all[['item', 'wm_month']].groupby('item').agg('count')
    months_count.columns = ['months']
    months_count.reset_index(inplace=True)
    
    # merge the month count to the original data
    df_all = pd.merge(df_all, months_count, how='left', left_on='item', right_on='item')
    df_all.to_csv(r'..\data\sales_projection_months.csv', index=False)
    return df_all

#df_all = read_raw_data()
df_all = pd.read_csv(r'..\data\sales_projection_months.csv')
    
#total items:355
#df_all['item'].nunique()

def prepare_study_data(df_all):
    '''
    Select the items with at least 13 months sales, remove the first month and last month.
    
    Arguments:
        df_all - dataframe with all data
        
    Returns:
    df_study - dataframe with new columns:
               pos_diff_pct: pos changes from last month in percentage
    '''
    df_study = df_all[df_all['months']>=13]
    #items with 14 or 13 months' data: 65
    #df_study['item'].nunique()
    
    #10 of them do not have 2017/02 data, 2018/03 data is not complete, remove them
    df_study = df_study[(df_study['wm_month']!='2017/02') & (df_study['wm_month']!='2018/03')]
    df_study.set_index(['item', 'wm_month'], inplace=True)
    df_study.sort_index(inplace=True)
    
    #calculate sales change from previous month in percentage
    df_study['pos_qty'] = df_study['pos_qty'].str.strip()
    df_study['pos_qty'] = df_study['pos_qty'].str.replace(',','')
    df_study['pos_qty'] = df_study['pos_qty'].astype('float64')
    df_study['pos_diff_pct'] = df_study['pos_qty'].pct_change()
    df_study.reset_index(inplace=True)
    df_study.loc[df_study['wm_month']=='2017/03', 'pos_diff_pct']=0
                
    #calculate the sales ratio to the first month 
    df_first_month = df_study.groupby('item').nth(0)
    df_first_month.reset_index(inplace=True)
    df_first_month = df_first_month[['item','pos_qty']]
    df_first_month.columns=['item', 'pos_1st_month']
    df_study = pd.merge(df_study, df_first_month, how='left', left_on='item', right_on='item')
    df_study['pos_pct_of_1st_month']=df_study['pos_qty']/df_study['pos_1st_month']
    df_study.drop('pos_1st_month',axis=1,inplace=True)
            
    df_study.to_csv(r'..\data\sales_study.csv', index=False)
    
    return df_study

df_study=prepare_study_data(df_all)
#df_study = pd.read_csv(r'..\data\sales_study.csv')




def pos_diff_pct_cluster(df_study):
    '''
    Generate clusters based on the sales month to month difference pecentage.
    
    Arguments:
        df_study - dataframe with selected items
        
    Returns:
        df_cluster - dataframe with kmean, allgomerative, dbscan clustering
    '''
    #aggregate diff per item into lists.
    df_diff = df_study.groupby('item')['pos_diff_pct'].apply(list)
    X=np.array(df_diff.tolist())
    
    #normalize the input
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    items = df_diff.index.tolist()
    
    kmeans = KMeans(n_clusters=8, random_state=0).fit(X_scaled)
    k=kmeans.labels_.tolist()
    
    agglomerative = AgglomerativeClustering(n_clusters=8).fit(X_scaled)
    a=agglomerative.labels_.tolist()
    #X_sign =np.sign(X)
    #agg_sign = AgglomerativeClustering(n_clusters=8).fit(X_sign)
    #as=agg_sign.labels_.tolist()
    
    dbscan = DBSCAN(eps=0.4).fit(X_scaled)
    d = list(dbscan.labels_)
    df_cluster = pd.DataFrame(list(zip(items,k,a, d)),
                              columns=['item','kmean','agglomerative', 'dbscan'])
    df_cluster.to_csv(r'..\data\sales_cluster.csv', index=False)

df_cluster = pos_diff_pct_cluster(df_study)

