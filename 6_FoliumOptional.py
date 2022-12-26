"""
Created on Fri Dec 23 21:38:46 2022
"""

# %% Set up

import folium
import pandas as pd
from folium.plugins import MarkerCluster
from folium.plugins import MousePosition
from folium.features import DivIcon
import webbrowser
import os

# Current directory
os.chdir('C:/Users/user/Desktop/Coursera/10_CapstoneDS/Figures')

# %% Download data, save data, select relevant data

URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_geo.csv'
spacex_df = pd.read_csv(URL)
spacex_df.to_csv('spacex_csv_file') # Save it just in case

# Select relevant sub-columns: `Launch Site`, `Lat(Latitude)`, `Long(Longitude)`, `class`
spacex_df = spacex_df[['Launch Site', 'Lat', 'Long', 'class']]
launch_sites_df = spacex_df.groupby(['Launch Site'], as_index=False).first()
launch_sites_df = launch_sites_df[['Launch Site', 'Lat', 'Long']]
launch_sites_df

# %% Task 1: Mark all launch sites on a map

output_file1 = "Map1.html" # To save map

# Start location is NASA Johnson Space Center
nasa_coordinate = [29.559684888503615, -95.0830971930759]
site_map = folium.Map(location=nasa_coordinate, zoom_start=10)
#site_map.save(output_file1)
#webbrowser.open(output_file1, new=2)

# Create a blue circle at NASA Johnson Space Center's coordinate with a popup label showing its name
circle = folium.Circle(nasa_coordinate, radius=1000, color='#d35400', fill=True).add_child(folium.Popup('NASA Johnson Space Center'))
# Create a blue circle at NASA Johnson Space Center's coordinate with a icon showing its name
marker = folium.map.Marker(
    nasa_coordinate,
    # Create an icon as a text label
    icon=DivIcon(
        icon_size=(20,20),
        icon_anchor=(0,0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % 'NASA JSC',
        )
    )
site_map.add_child(circle)
site_map.add_child(marker)

# Initial the map
site_map = folium.Map(location=nasa_coordinate, zoom_start=5)
# For each launch site, add a Circle object based on its coordinate (Lat, Long) values. In addition, add Launch site name as a popup label

def create_point_coordinates(row):
    coordinate = [row['Lat'], row['Long']]
    print('Coordinates : ',coordinate)
    circle = folium.Circle(coordinate, radius=100, color='#d35400', fill=True).add_child(folium.Popup(row['Launch Site']))
    # Create a blue circle at NASA Johnson Space Center's coordinate with a icon showing its name
    marker = folium.map.Marker(
        coordinate,
        # Create an icon as a text label
        icon=DivIcon(
            icon_size=(20,20),
            icon_anchor=(0,0),
            html='%s' % row['Launch Site'],
            )
        )
    site_map.add_child(circle)
    site_map.add_child(marker)

launch_sites_df.apply(lambda row : create_point_coordinates(row), axis = 1)

# Save and display map
site_map.save(output_file1)
webbrowser.open(output_file1, new=2)

# %% Task 2: Mark the success/failed launches for each site on the map

output_file2 = "Map2.html" # To save map

marker_cluster = MarkerCluster()

# TODO: Create a new column in launch_sites dataframe called marker_color to store the marker colors based on the class value
# Apply a function to check the value of `class` column
# If class=1, marker_color value will be green
# If class=0, marker_color value will be red
list_colors=[]
for clas in spacex_df['class']:
    if clas==1 :
        list_colors.append('green')
    else:
        list_colors.append('red')
spacex_df['marker_color'] = list_colors
spacex_df.tail(10)

# Function to assign color to launch outcome
def assign_marker_color(launch_outcome):
    if launch_outcome == 1:
        return 'green'
    else:
        return 'red'
    
spacex_df['marker_color'] = spacex_df['class'].apply(assign_marker_color)
spacex_df.tail(10)

# TODO: For each launch result in spacex_df data frame, add a folium.Marker to marker_cluster
# Add marker_cluster to current site_map
site_map.add_child(marker_cluster)

# for each row in spacex_df data frame
# create a Marker object with its coordinate
# and customize the Marker's icon property to indicate if this launch was successed or failed, 
# e.g., icon=folium.Icon(color='white', icon_color=row['marker_color']
for index, row in spacex_df.iterrows():
    # TODO: Create and add a Marker cluster to the site map
    # marker = folium.Marker(...)
    lat = row['Lat']
    lng = row['Long']
    label = row['Launch Site']
    marker = folium.Marker(
        location=[lat, lng],
        icon=folium.Icon(color='white', icon_color = row['marker_color']),
        popup= label,
    )
    marker_cluster.add_child(marker)

# Save and display map
site_map.save(output_file2)
webbrowser.open(output_file2, new=2)


# %% TASK 3: Calculate the distances between a launch site to its proximities

output_file3 = "Map3.html" # To save map

# Add Mouse Position to get the coordinate (Lat, Long) for a mouse over on the map
formatter = "function(num) {return L.Util.formatNum(num, 5);};"
mouse_position = MousePosition(
    position='topright',
    separator=' Long: ',
    empty_string='NaN',
    lng_first=False,
    num_digits=20,
    prefix='Lat:',
    lat_formatter=formatter,
    lng_formatter=formatter,
)

site_map.add_child(mouse_position)


from math import sin, cos, sqrt, atan2, radians

def calculate_distance(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

# TODO: Mark down a point on the closest coastline using MousePosition and calculate the distance between the coastline point and the launch site.
# find coordinate of the closet coastline
# e.g.,: Lat: 28.56367  Lon: -80.57163
lat_lon_launch_site = [28.56318, -80.57683]
lat_lon_coastline = [28.56317, -80.56796]

launch_site_lat = lat_lon_launch_site[0]
launch_site_lon = lat_lon_launch_site[1]
coastline_lat = lat_lon_coastline[0]
coastline_lon = lat_lon_coastline[1]

distance_coastline = calculate_distance(launch_site_lat, launch_site_lon, coastline_lat, coastline_lon)
distance_coastline

# TODO: After obtained its coordinate, create a folium.Marker to show the distance
# Create and add a folium.Marker on your selected closest coastline point on the map
# Display the distance between coastline point and launch site using the icon property 
# for example
distance_marker = folium.Marker(
   lat_lon_coastline,
   icon=DivIcon(
       icon_size=(20,20),
       icon_anchor=(0,0),
       html='%s' % "{:10.2f} KM".format(distance_coastline),
       )
   )
site_map.add_child(distance_marker)




# TODO: Draw a PolyLine between a launch site to the selected coastline point
# Create a `folium.PolyLine` object using the coastline coordinates and launch site coordinate
# lines=folium.PolyLine(locations=coordinates, weight=1)
coordinates = [lat_lon_launch_site, lat_lon_coastline]
lines = folium.PolyLine(locations=coordinates, weight=1)
site_map.add_child(lines)


# Create a marker with distance to a closest city, railway, highway, etc.
# Draw a line between the marker to the launch site
lat_lon_launch_site = [28.56318, -80.57683]
lat_lon_highway = [28.56298, -80.57073]
#lat_lon_railway = [28.57209, -80.58527]
#lat_lon_city = [28.6115, -80.8077]

launch_site_lat = lat_lon_launch_site[0]
launch_site_lon = lat_lon_launch_site[1]
highway_lat = lat_lon_highway[0]
highway_lon = lat_lon_highway[1]

distance_highway = calculate_distance(launch_site_lat, launch_site_lon, highway_lat, highway_lon)
distance_highway

distance_marker = folium.Marker(
   lat_lon_highway,
   icon=DivIcon(
       icon_size=(20,20),
       icon_anchor=(0,0),
       html='%s' % "{:10.2f} KM".format(distance_highway),
       )
   )
site_map.add_child(distance_marker)


# Save and display map
site_map.save(output_file3)
webbrowser.open(output_file3, new=2)
