import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os
from sklearn.neighbors import BallTree

# Cache the FIPS data loading
@st.cache_data
def load_fips_data():
    fips_df = pd.read_excel("US_FIPS_Codes.xls", skiprows=1).rename(columns={"County Name":"County"})
    fips_df['FIPS County'] = fips_df['FIPS State'].astype(str) + fips_df['FIPS County'].astype(str).str.zfill(3)
    return fips_df.query("State.isin(['Massachusetts','Florida','New York','Georgia','West Virginia'])")

@st.cache_data
def load_county_data(county):
    return pd.read_parquet(f"county_data/county_{county}.parquet").query("match_dir==1")

def calculate_score(state, county, lat, lon):
    """
    Calculate score based on state, county, and coordinates.
    Replace this with your actual score calculation logic.
    """
    #try:
    print("calculate_score")
    print(state, county, lat, lon)
    neighbors_data = get_neighbors(county = county,center = [[lat,lon]], radius_miles=.25)
    aggregated_volume = neighbors_data['trips_volu'].mean()

    if np.isnan(aggregated_volume):
        print("no volume")
        return 0
    
    county_data = pd.read_parquet(f"county_data_points/county_{county}_points.parquet") 
    impression_percentiles = np.linspace(1,99,99)

    print("aggregated_volume")
    print(aggregated_volume)
    volume_thresholds = np.percentile(county_data['trips_volu'],impression_percentiles)
    impression_score = interpolate_percentile(aggregated_volume, volume_thresholds, impression_percentiles)
    # Add your scoring computation here
    score = impression_score  # Replace with actual calculation
    return f"{score:.2f}"
    #except Exception as e:
    #    return "Error calculating score"

def interpolate_percentile(volume_value, volume_thresholds, impression_percentiles):
    # Handle edge cases
    if volume_value <= volume_thresholds[0]:
        return impression_percentiles[0]
    if volume_value >= volume_thresholds[-1]:
        return impression_percentiles[-1]
    
    # Find the two closest x values
    left_index = np.searchsorted(volume_thresholds, volume_value) - 1
    print(volume_thresholds)
    print(volume_value)
    print(left_index)
    # Linear interpolation
    x1, x2 = volume_thresholds[left_index], volume_thresholds[left_index + 1]
    y1, y2 = impression_percentiles[left_index], impression_percentiles[left_index + 1]
    
    percentile = y1 + (volume_value - x1) * (y2 - y1) / (x2 - x1)
    return percentile

def get_neighbors(county,center,radius_miles):
    
    '''
    use k-d tree to get all points within a certain radius of an inputted point
    specify county and radius

    returns rows of county dataframe which are within the distance from inputted point
    '''

    data = pd.read_parquet(f"county_data_points/county_{county}_points.parquet") 

    #haversine distance metric requires distance to be in radians
    MILES_RADIAN_CONVERSION = 3958.8
    radius_radians = radius_miles / MILES_RADIAN_CONVERSION

    #initialize kd tree
    rng = np.random.RandomState(0)
    tree = BallTree(data[['lat','long']].values, leaf_size=20, metric='haversine')       
    
    #perform the search
    ind,dist  = tree.query_radius(center, r=radius_radians,return_distance=True)       
    ind,dist = ind[0],dist[0]

    print(f"Number of Points Within {radius_miles} Miles: {len(ind)}")
    

    neighbors_df = data.iloc[ind].copy()
    neighbors_df['distances'] = dist * MILES_RADIAN_CONVERSION
    return neighbors_df




def deserialize_bytes(geometry_bytes):
    from shapely.wkb import loads
    return loads(geometry_bytes)

def get_lat_long(geometry):
    return geometry.xy[1][0], geometry.xy[0][0]

def generate_viz(county, viz_type='path', marker_lat=None, marker_lon=None):
    try:
        # Load only necessary columns
        lines = load_county_data(county)[['geometry', 'osm_id', 'trips_volu']]#.head(500)
        
        # Process geometry in vectorized operations
        lines['geometry'] = lines['geometry'].apply(deserialize_bytes)
        lines["lat"], lines["long"] = zip(*lines['geometry'].apply(get_lat_long))
        
        # Calculate log-transformed trips for better visualization
        log_trips = np.log1p(lines["trips_volu"])
        norm = Normalize(vmin=log_trips.min(), vmax=log_trips.max())
        colormap = ScalarMappable(norm=norm, cmap="viridis")
        colors = (colormap.to_rgba(log_trips)[:, :3] * 255).astype(int).tolist()

        # Set initial view state
        view_state = pdk.ViewState(
            latitude=float(lines["lat"].mean()),
            longitude=float(lines["long"].mean()),
            zoom=11,
            pitch=0,
            bearing=30
        )

        layers = []

        if viz_type == 'path':
            # Create line data for path visualization
            line_data = [[[float(x), float(y)] for x, y in zip(*geom.xy)] 
                        for geom in lines['geometry']]

            df = pd.DataFrame({
                'path_id': lines["osm_id"],
                'path': line_data,
                'color': colors,
                'volume': lines["trips_volu"]
            })

            path_layer = pdk.Layer(
                'PathLayer',
                df,
                get_path='path',
                get_color='color',
                width_scale=10,
                width_min_pixels=2,
                pickable=True,
                auto_highlight=True
            )
            
            layers.append(path_layer)
            tooltip = {'text': 'Path: {path_id}\nVolume: {volume}'}

        else:  # heatmap
            # Create point data for heatmap
            df = pd.DataFrame({
                'latitude': lines["lat"],
                'longitude': lines["long"],
                'weight': lines["trips_volu"]
            })

            heatmap_layer = pdk.Layer(
                'HeatmapLayer',
                df,
                opacity=0.8,
                get_position=['longitude', 'latitude'],
                get_weight='weight',
                aggregation='"SUM"',
                threshold=0.05,
                radius_pixels=30,
                color_range=[
                    [65, 182, 196],
                    [127, 205, 187],
                    [199, 233, 180],
                    [237, 248, 177],
                    [255, 255, 204],
                    [255, 237, 160],
                    [254, 217, 118],
                    [254, 178, 76],
                    [253, 141, 60],
                    [252, 78, 42],
                    [227, 26, 28],
                    [189, 0, 38]
                ]
            )
            
            layers.append(heatmap_layer)
            tooltip = None

        # Add marker layer if coordinates are provided
        if marker_lat is not None and marker_lon is not None:
            marker_data = pd.DataFrame({
                'latitude': [marker_lat],
                'longitude': [marker_lon]
            })
            
            marker_layer = pdk.Layer(
                'ScatterplotLayer',
                marker_data,
                get_position=['longitude', 'latitude'],
                get_color=[255, 0, 0, 200],  # Red marker with some transparency
                get_radius=20,
                pickable=True,
                radiusScale=6,
                radiusMinPixels=5,
                radiusMaxPixels=20
            )
            
            layers.append(marker_layer)

        # Create the deck
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_style='mapbox://styles/mapbox/satellite-streets-v11',
            tooltip=tooltip
        )

        return deck

    except Exception as e:
        st.error(f"Error processing county {county}: {str(e)}")
        return None

def main():
    st.set_page_config(layout="wide")
    st.title("Visibility and Impression Visualization")
    
    # Load FIPS data
    fips_df = load_fips_data()
    states = sorted(fips_df['State'].unique())

    # Initialize session state variables if they don't exist
    if 'score' not in st.session_state:
        st.session_state.score = ""
    if 'lat_input' not in st.session_state:
        st.session_state.lat_input = ""
    if 'lon_input' not in st.session_state:
        st.session_state.lon_input = ""
    
    # Create columns for all controls in a single row
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
    
    with col1:
        default_state_index = states.index('Massachusetts')
        selected_state = st.selectbox(
            "Select State",
            options=states,
            index=default_state_index
        )
    
    # Filter counties for selected state
    state_counties = fips_df[fips_df['State'] == selected_state]
    
    with col2:
        counties_list = state_counties['County'].tolist()
        # Set Middlesex as default county
        default_county_index = counties_list.index('Middlesex')
        selected_county_name = st.selectbox(
            "Select County",
            options=counties_list,
            index=default_county_index
        )
    
    with col3:
        viz_type = st.selectbox(
            "Visualization Type",
            options=['Path View', 'Heatmap'],
            index=0
        )

    # Get the FIPS code for the selected county
    selected_county_fips = state_counties[
        state_counties['County'] == selected_county_name
    ]['FIPS County'].iloc[0]

    def update_coordinates():
        print("update_coordinates")
        print(st.session_state.lat_input)
        print(st.session_state.lon_input)

        
        
        #try:
        if st.session_state.lat_input and st.session_state.lon_input:

            lat = float(clean_coordinate(st.session_state.lat_input.replace('−', '-')))
            lon = float(clean_coordinate(st.session_state.lon_input.replace('−', '-')))
            
            # Store the validated coordinates in session state
            st.session_state.lat = lat
            st.session_state.lon = lon

            lat = float(st.session_state.lat_input)
            lon = float(st.session_state.lon_input)
            
            # Store the validated coordinates in session state
            st.session_state.lat = lat
            st.session_state.lon = lon
            
            print(f"Updated coordinates - Lat: {lat}, Lon: {lon}")
            
            st.session_state.score = calculate_score(
                selected_state,
                selected_county_fips,
                lat,
                lon
            )
        #except ValueError as e:
        #   st.error("Please enter valid numerical coordinates")
        #    print(f"Coordinate validation error: {e}")
    
    with col4:
        lat_input = st.text_input(
            "Latitude",
            key="lat_input",
            help="Enter latitude for marker (optional)",
            on_change=update_coordinates
        )
    
    with col5:
        lon_input = st.text_input(
            "Longitude",
            key="lon_input",
            help="Enter longitude for marker (optional)",
            on_change=update_coordinates
        )
    
    with col6:
        st.text_input(
            "Score",
            value=st.session_state.score,
            disabled=True
        )
    
    # Convert coordinates to float if provided
    try:
        marker_lat = float(st.session_state.lat_input) if st.session_state.lat_input else None
        marker_lon = float(st.session_state.lon_input) if st.session_state.lon_input else None
    except ValueError:
        marker_lat, marker_lon = None, None
    
    # Generate and display visualization
    if selected_county_fips:
        with st.spinner(f'Loading data for {selected_county_name}, {selected_state}...'):
            deck = generate_viz(
                selected_county_fips, 
                'path' if viz_type == 'Path View' else 'heatmap',
                marker_lat,
                marker_lon
            )
            if deck:
                st.pydeck_chart(deck, use_container_width=True)


def clean_coordinate(coord_str):
        # Replace various types of dashes/minus signs with standard minus
        replacements = {
            '−': '-',  # replace en dash
            '—': '-',  # replace em dash
            '–': '-',  # replace figure dash
            '‐': '-',  # replace hyphen
        }
        for old, new in replacements.items():
            coord_str = coord_str.replace(old, new)
        return coord_str.strip()

if __name__ == "__main__":
    main()