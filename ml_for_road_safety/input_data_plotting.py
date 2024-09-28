from data_loaders import TrafficAccidentDataset
import plotly.express as px
import pandas as pd
dataset = TrafficAccidentDataset(state_name = "CA", data_dir="../data/Final_Graphs",
                            #node_feature_type = args.node_feature_type,
                            use_static_edge_features=True,
                            #use_dynamic_node_features=args.load_dynamic_node_features,
                            #use_dynamic_edge_features=args.load_dynamic_edge_features,
                            #train_years=args.train_years,
                            num_negative_edges=100000000) 

monthly_data = dataset.load_monthly_data(2016, 1)

mon_data = monthly_data["temporal_node_features"].cpu().detach().numpy()
df = pd.DataFrame(mon_data)
df.columns = ["node_id","lat","lon","tavg","tmin","tmax","prcp","wspd","pres"]

print(df.head())
color_scale = [(0, 'orange'), (1,'red')]

fig = px.scatter_mapbox(df, 
                        lat="lat", 
                        lon="lon", 
                        hover_name="node_id", 
                        hover_data=["tavg","tmin","tmax","prcp","wspd","pres"],
                        color="prcp",
                        color_continuous_scale=color_scale,
                        size="tavg",
                        zoom=8, 
                        height=800,
                        width=800)
#Need to plot node features
print(monthly_data['temporal_node_features'])

fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
#fig.write_html("test.html")