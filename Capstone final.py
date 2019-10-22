#!/usr/bin/env python
# coding: utf-8

# In[406]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

import json # library to handle JSON files

from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
import geocoder # to get coordinates

import requests # library to handle requests|
from bs4 import BeautifulSoup # library to parse HTML and XML documents

from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

import folium # map rendering library
import tqdm

print("Libraries imported.")


# In[407]:


data = requests.get("https://en.wikipedia.org/wiki/Category:Neighbourhoods_in_Dhaka").text


# In[408]:


soup = BeautifulSoup(data, 'html.parser')


# In[409]:


neighborhoodList = []
for row in soup.find_all("div", class_="mw-category")[0].findAll("li"):
    neighborhoodList.append(row.text)


# In[410]:


df = pd.DataFrame({"Neighborhood": neighborhoodList})
df.head()


# In[411]:


df.shape


# In[412]:


def get_latlng(neighborhood):
    # initialize your variable to None
    lat_lng_coords = None
    # loop until you get the coordinates
    while(lat_lng_coords is None):
        g = geocoder.arcgis('{}, Dhaka, Bangladesh'.format(neighborhood))
        lat_lng_coords = g.latlng
    return lat_lng_coords


# In[413]:


coords = [ get_latlng(neighborhood) for neighborhood in df["Neighborhood"].tolist() ]


# In[414]:


coords


# In[415]:


df_temp = pd.DataFrame(coords, columns=['Latitude', 'Longitude'])


# In[416]:


df['Latitude'] = df_temp['Latitude']
df['Longitude'] = df_temp['Longitude']


# In[417]:


print(df.shape)
df.head()


# In[418]:


df.to_csv("dhaka.csv", index=False)


# In[420]:


address = 'Dhaka, Bangladesh'

geolocator = Nominatim(user_agent="my-application")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Dhaka, Bangladesh {}, {}.'.format(latitude, longitude))


# In[421]:


map_dhaka = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, neighborhood in zip(df['Latitude'], df['Longitude'], df['Neighborhood']):
    label = '{}'.format(neighborhood)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7).add_to(map_dhaka)  
    
map_dhaka


# In[422]:


map_dhaka.save('map_dhaka.html')


# In[423]:


CLIENT_ID = 'client id' # your Foursquare ID
CLIENT_SECRET = 'client secret' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[424]:


def getNearbyVenues(names, latitudes, longitudes, radius=200, LIMIT = 25):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        #print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[425]:


venues = getNearbyVenues(df.Neighborhood,
                         df.Latitude,
                         df.Longitude)


# In[426]:


venues.head(10)


# In[427]:


venues.groupby(["Neighborhood"]).count()


# In[428]:


print('There are {} uniques categories.'.format(len(venues['Venue Category'].unique())))


# In[429]:


onehot = pd.get_dummies(venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
onehot['Neighborhoods'] = venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [onehot.columns[-1]] + list(onehot.columns[:-1])
onehot = onehot[fixed_columns]

print(onehot.shape)
onehot.head()


# In[430]:


onehot.drop(onehot.columns[[1,3,4,5,6]], axis = 1, inplace = True)


# In[431]:


onehot.head()


# In[432]:


onehot.drop(onehot.columns[[3,4,5,8,9]], axis = 1, inplace = True)


# In[433]:


onehot.head()


# In[434]:


onehot.drop(onehot.columns[[5,7,8,9,10,11]], axis = 1, inplace = True)


# In[435]:


onehot.head()


# In[436]:


onehot.drop(onehot.columns[[7,8,9,11]], axis = 1, inplace = True)


# In[437]:


onehot.head()


# In[438]:


onehot["Restaurants"] = onehot.sum(axis = 1)


# In[439]:


onehot.head()


# In[440]:


onehot.drop(onehot.columns[[1,2,3,4,5,6,7,8]], axis = 1, inplace = True)


# In[441]:


onehot.head()


# In[442]:


grouped = onehot.groupby(["Neighborhoods"]).mean().reset_index()
print(grouped.shape)
grouped


# In[445]:


kclusters = 3

clustering = grouped.drop(["Neighborhoods"], 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# In[447]:


merged = grouped.copy()

# add clustering labels
merged["Cluster Labels"] = kmeans.labels_


# In[448]:


merged.rename(columns={"Neighborhoods": "Neighborhood"}, inplace=True)
merged.head()


# In[449]:


merged = merged.join(df.set_index("Neighborhood"), on="Neighborhood")
print(merged.shape)
merged.head()


# In[450]:


print(merged.shape)
merged.sort_values(["Cluster Labels"], inplace=True)
merged


# In[451]:




# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i+x+(i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(merged['Latitude'], merged['Longitude'], merged['Neighborhood'], merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' - Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[458]:


map_clusters.save('D://map_clusters.html')


# In[453]:


merged.loc[merged['Cluster Labels'] == 0]


# In[454]:


merged.loc[merged['Cluster Labels'] == 1]


# In[455]:


merged.loc[merged['Cluster Labels'] == 2]


# In[ ]:




