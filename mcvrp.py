################################################################

import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import json
import osmnx as ox
import networkx as nx
from geopy.distance import geodesic
from sklearn import *

from geopy.distance import geodesic
from datetime import timedelta

import streamlit as st
from streamlit_folium import st_folium

#Fixstars Amplify 関係のインポート
import amplify
from amplify.client import FixstarsClient
from amplify import VariableGenerator
from amplify import one_hot
from amplify import einsum
from amplify import less_equal, ConstraintList
from amplify import Poly
from amplify import Model
from amplify import FixstarsClient
from amplify import solve
import copy

st.set_page_config(
    page_title="小田原市 周辺",
    page_icon="🗾",
    layout="wide"
)
#########################################
# streamlit custom css
#########################################
st.markdown(
"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sawarabi+Gothic&display=swap');
    body{
        font-family: "Sawarabi Gothic", sans-serif;
        font-style: normal;
        font-weight: 400;
    }
    .Qheader{
        background:siliver;
    }
    .Qtitle{
        padding-left:1em;
        padding-right:3em;
        font-size:4em;
        font-weight:600;
        color:darkgray;
    }
    .Qsubheader{
        font-size:2em;
        font-weight:600;
        color:gray;
    }
    .caption{
        font-size:1.5em;
        font-weight:400:
        color:gray;
        align:right;
    }
</style>
""",unsafe_allow_html=True
)

# 色指定
_colors = [
    "green",
    "orange",
    "blue",
    "red",
    "cadetblue",
    "darkred",
    "darkblue",
    "purple",
    "pink",
    "lightred",
    "darkgreen",
    "lightgreen",
    "lightblue",
    "darkpurple",
]

# ディレクトリ設定
#dir_name = "/Q-quest2024/teamC/"
#dir_name = "/Q-Quest/"
#root_dir = "/content/drive/MyDrive/" + dir_name
root_dir="./"

#######################
#　ファイル指定
#######################

node_data =  "kyoten_geocode_Revised.json"
numOfPeople = "number_of_people.csv"
#geojson_path = root_dir + "GIS/N03-20240101_14_GML/N03-20240101_14.geojson"
geojson_path = root_dir + "N03-20240101_14.geojson"
route_file = "path_list_v2.json"
Map_Tile='https://cyberjapandata.gsi.go.jp/xyz/std/{z}/{x}/{y}.png'

GIS_HIGHT=650
GIS_WIDE=1000
GIS_ZOOM=12


########################################
# Folium を使う表示系関数
########################################

def disp_odawaraMap(odawara_district,center=[35.2646012,139.15223698], zoom_start=GIS_ZOOM):
    m = folium.Map(
        location=center,
        tiles=Map_Tile,
        attr='電子国土基本図',
        zoom_start=zoom_start
    )
    folium.GeoJson(
        odawara_district,
        style_function=lambda x: {
            'color': 'gray',
            'weight': 2,
            'dashArray': '5, 5'
        }
    ).add_to(m)
    return m


def plot_marker(m, data):
    for _, row in data.iterrows():
        if row['Node'][0] == 'K':
            icol = 'pink'
        elif row['Node'][0] == 'M':
            icol = 'blue'
        elif row['Node'][0] == 'N':
            icol = 'red'
        else:
            icol = 'green'
        folium.Marker(
            location=[row['緯度'], row['経度']],
            popup=f"{row['施設名']} / {row['住所']} ({row['拠点種類']})",
            icon=folium.Icon(color=icol)
        ).add_to(m)

def plot_select_marker(m, data,op_data):
    actve_layer = folium.FeatureGroup(name="開設")
    actve_layer.add_to(m)
    nonactive_layer = folium.FeatureGroup(name="閉鎖")
    nonactive_layer.add_to(m)

    for _, row in data.iterrows():
        if row['Node'][0] == 'K':
          if row['Node'] in (op_data['避難所']):
            icol = 'green'
            layer=actve_layer
          else:
            icol = 'lightgray'
            layer=nonactive_layer

        elif row['Node'][0] == 'M':
          if row['Node'] in (op_data['配送拠点']):
            icol = 'purple'
            layer=actve_layer
          else:
            icol = 'gray'
            layer=nonactive_layer
        else:
          continue

        folium.Marker(
            location=[row['緯度'], row['経度']],
            popup=f"{row['施設名']} / {row['住所']} ({row['拠点種類']})",
            icon=folium.Icon(color=icol)
        ).add_to(layer)


def draw_route(m, G, best_routes, path_df, node_name_list):
    for k, vehicle_route in best_routes.items():
        layer = folium.FeatureGroup(name=f"ルート {k}")
        layer.add_to(m)
        for iv in range(len(vehicle_route) - 1):
            start_node = node_name_list[vehicle_route[iv]]
            goal_node = node_name_list[vehicle_route[iv + 1]]
            route = path_df[(path_df['start_node'] == start_node) & (path_df['goal_node'] == goal_node)]['route']
            for route_nodes in route:
              route_gdf = ox.graph_to_gdfs(G.subgraph(route_nodes), nodes=False)
              route_gdf.explore(
                  m=layer,  # folium.FeatureGroupオブジェクトを指定
                  color=_colors[k % len(_colors)],
                  style_kwds={"weight": 10.0, "opacity": 0.5},
              )
    #folium.LayerControl().add_to(m)
    return m

def draw_route_v2(m, G, best_routes, path_df, node_name_list):
    for k, vehicle_route in best_routes.items():
        layer = folium.FeatureGroup(name=f"ルート {k}")
        layer.add_to(m)
        for iv in range(len(vehicle_route) - 1):
            start_node = node_name_list[vehicle_route[iv]]
            goal_node = node_name_list[vehicle_route[iv + 1]]
            route = path_df[(path_df['start_node'] == start_node) & (path_df['goal_node'] == goal_node)]['route']
            for route_nodes in route:
              route_gdf = ox.graph_to_gdfs(G.subgraph(route_nodes), nodes=False)
              route_gdf.explore(
                  m=layer,  # folium.FeatureGroupオブジェクトを指定
                  color=_colors[k % len(_colors)],
                  style_kwds={"weight": 10.0, "opacity": 0.5},
              )
    #folium.LayerControl().add_to(m)
    return 

def get_point_name(data,node):
   for i,row in data.iterrows():
      if row['Node']== node:
         return row['施設名']
      
def set_map_data():

  map_data={}

  map_data['node_d']=pd.read_json(root_dir + node_data)    #拠点データ

  administrative_district = gpd.read_file(geojson_path)
  map_data['gep_map']=administrative_district[administrative_district["N03_004"]=="小田原市"]

  map_data['path_d'] = pd.read_json(root_dir + route_file)   #道路データ

  # グラフ
  place = {'city' : 'Odawara',
         'state' : 'Kanagawa',
         'country' : 'Japan'}
  map_data['G'] = ox.graph_from_place(place, network_type='drive')

  map_data['base_map']=disp_odawaraMap(map_data['gep_map'] )

  return(map_data)

########################################
# アニーリング周り(以前の関数群)
########################################

def start_amplify():
  client = FixstarsClient()
  client.token = "AE/UnDbUvmjJ2xmrFRThrutxzPiVjxikMSk"  # 有効なトークンを設定

  return client


def process_sequence(sequence: dict[int, list]) -> dict[int, list]:
    new_seq = dict()
    for k, v in sequence.items():
        v = np.append(v, v[0])
        mask = np.concatenate(([True], np.diff(v) != 0))
        new_seq[k] = v[mask]
    return new_seq

def onehot2sequence(solution: np.ndarray) -> dict[int, list]:
    nvehicle = solution.shape[2]
    sequence = dict()
    for k in range(nvehicle):
        sequence[k] = np.where(solution[:, :, k])[1]
    return sequence

def upperbound_of_tour(capacity: int, demand: np.ndarray) -> int:
    max_tourable_bases = 0
    for w in sorted(demand):
        capacity -= w
        if capacity >= 0:
            max_tourable_bases += 1
        else:
            return max_tourable_bases
    return max_tourable_bases

def set_distance_matrix(path_df, node_list):
    distance_matrix = np.zeros((len(node_list), len(node_list)))
    for i, st_node in enumerate(node_list):
        for j, ed_node in enumerate(node_list):
            row = path_df[(path_df['start_node'] == st_node) & (path_df['goal_node'] == ed_node)]
            if row.empty:
                if st_node == ed_node:
                    dis = 0
                else:
                    dis = np.inf
            else:
                dis = row['distance'].values[0]
            distance_matrix[i, j] = dis
    return distance_matrix

def set_parameter( path_df, op_data):
    annering_param = {}
    np_df = pd.read_csv(root_dir + numOfPeople) #人数データ

    re_node_list = op_data['配送拠点'] + op_data['避難所']
    distance_matrix = set_distance_matrix(path_df, re_node_list)

    n_transport_base = len(op_data['配送拠点'])
    n_shellter = len(op_data['避難所'])
    nbase = distance_matrix.shape[0]
    nvehicle = n_transport_base

    avg_nbase_per_vehicle = (nbase - n_transport_base) // nvehicle

    demand = np.zeros(nbase)
    for i in range(nbase - n_transport_base - 1):
        demand[i + n_transport_base] = np_df.iloc[i,1]

    demand_max = np.max(demand)
    demand_mean = np.mean(demand[nvehicle:])

    capacity = int(demand_max) + int(demand_mean) * (avg_nbase_per_vehicle)

    annering_param['distance_matrix'] = distance_matrix
    annering_param['n_transport_base'] = n_transport_base
    annering_param['n_shellter'] = n_shellter
    annering_param['nbase'] = nbase
    annering_param['nvehicle'] = nvehicle
    annering_param['capacity'] = capacity
    annering_param['demand'] = demand
    annering_param['npeople'] = np_df

    return annering_param

def set_annering_model(ap):
    gen = VariableGenerator()
    max_tourable_bases = upperbound_of_tour(ap['capacity'], ap['demand'][ap['nvehicle']:])
    x = gen.array("Binary", shape=(max_tourable_bases + 2, ap['nbase'], ap['nvehicle']))

    for k in range(ap['nvehicle']):
        if k > 0:
            x[:, 0:k, k] = 0
        if k < ap['nvehicle'] - 1:
            x[:, k+1:ap['nvehicle'], k] = 0
        x[0, k, k] = 1
        x[-1, k, k] = 1
        x[0, ap['nvehicle']:, k] = 0
        x[-1, ap['nvehicle']:, k] = 0

    one_trip_constraints = one_hot(x[1:-1, :, :], axis=1)
    one_visit_constraints = one_hot(x[1:-1, ap['nvehicle']:, :], axis=(0, 2))

    weight_sums = einsum("j,ijk->ik", ap['demand'], x[1:-1, :, :])
    capacity_constraints: ConstraintList = less_equal(
        weight_sums,
        ap['capacity'],
        axis=0,
        penalty_formulation="Relaxation",
    )

    objective: Poly = einsum("pq,ipk,iqk->", ap['distance_matrix'], x[:-1], x[1:])

    constraints = one_trip_constraints + one_visit_constraints + capacity_constraints
    constraints *= np.max(ap['distance_matrix'])

    model = Model(objective, constraints)

    return model, x

def sovle_annering(model, client, num_cal, timeout):
    client.parameters.timeout = timedelta(milliseconds=timeout)
    result = solve(model, client, num_solves=num_cal)
    if len(result) == 0:
        raise RuntimeError("Constraints not satisfied.")
    return result

########################################
# ここからStreamlit本体
########################################

st.markdown('<div class="Qheader"><span class="Qtitle">Q-LOGIQ</span> <span class="caption">Quantum Logistics Intelligence & Quality Optimization  created by WINKY Force</span></div>', unsafe_allow_html=True)

gis_st, anr_st = st.columns([2, 1])

if "client" not in st.session_state:
    st.session_state["client"] =start_amplify()

client=st.session_state["client"]

if "map_data" not in st.session_state:
    st.session_state["map_data"] = set_map_data()

map_data=st.session_state["map_data"]
G=map_data['G']
df=map_data['node_d']
path_df=map_data['path_d']
base_map=map_data['base_map']
base_map_copy = copy.deepcopy(base_map)

# --- セッションステートで計算結果を保持
if "best_tour" not in st.session_state:
    st.session_state["best_tour"] = None
if "best_cost" not in st.session_state:
    st.session_state["best_cost"] = None
if "points" not in st.session_state:
    st.session_state["points"] = None
if 'num_shelter' not in st.session_state:
    st.session_state['num_shelter'] = 0
if 'num_transport' not in st.session_state:
    st.session_state['num_transport'] = 0
if 'annering_param' not in st.session_state:
    st.session_state["annering_param"] = None

st.session_state['redraw'] = False

best_tour=st.session_state['best_tour']
selected_base=st.session_state['points']


# すべての拠点のリストを取得
all_shelter= df[df['Node'].str.startswith('K')]
all_transport= df[df['Node'].str.startswith('M')]

with anr_st:
  st.markdown('<div class="Qsubheader">拠点リスト</div>',unsafe_allow_html=True)
  spinner_container = st.container()
  st.write("開設されている避難所と配送拠点を選んでください")
  selected_shelter=anr_st.pills("避難所",all_shelter['施設名'].tolist(),selection_mode="multi")
  selected_transport=anr_st.pills("配送拠点",all_transport['施設名'].tolist(),selection_mode="multi")
  st.write("選択完了後、下のボタンを押してください。")

selected_shelter_node=all_shelter[all_shelter['施設名'].isin(selected_shelter)]['Node'].tolist()
selected_transport_node=all_transport[all_transport['施設名'].isin(selected_transport)]['Node'].tolist()

num_shelter=len(selected_shelter_node)
num_transport=len(selected_transport_node)

if num_shelter != st.session_state['num_shelter'] or num_transport != st.session_state['num_transport']:
    st.session_state['num_shelter'] = num_shelter
    st.session_state['num_transport'] = num_transport
    best_tour = None
    st.session_state["best_tour"] = best_tour

selected_base={'配送拠点':selected_transport_node,'避難所':selected_shelter_node}

st.session_state['points']=selected_base
re_node_list = selected_base['配送拠点'] +selected_base['避難所']

with gis_st:
  if best_tour !=None:
    st.markdown('<div class="Qsubheader">配送最適化-計算結果</div>',unsafe_allow_html=True)
    selected_base=st.session_state['points']
    plot_select_marker(base_map_copy, df,selected_base)
    #re_node_list = selected_base['配送拠点'] +selected_base['避難所']
    base_map_copy = draw_route(base_map_copy, G, best_tour, path_df, re_node_list)

  elif selected_base !=None:
    st.markdown('<div class="Qsubheader">避難所・配送拠点の設置</div>',unsafe_allow_html=True)
    plot_select_marker(base_map_copy, df,selected_base)
  else:
    st.markdown('<div class="Qsubheader">避難所・配送拠点の設置</div>',unsafe_allow_html=True)

  folium.LayerControl().add_to(base_map_copy)
  st_folium(base_map_copy, width=GIS_WIDE, height=GIS_HIGHT)

if anr_st.button("最適経路探索開始"):
    with spinner_container:
        with st.spinner("処理中です。しばらくお待ちください..."):
        #gis_st.write(f'選択された避難所: {selected_shelter_node}//選択された配送拠点:{selected_transport_node}')
            if not selected_shelter_node or not selected_transport_node:
                anr_st.warning("避難所・配送拠点をそれぞれを1つ以上選択してください")
            else:
            # ここでアニーリング等を実行
            #annering_param = set_parameter(np_df, path_df, op_data)
                annering_param=set_parameter(path_df,selected_base)
                model, x = set_annering_model(annering_param)
                loop_max = 20
                best_tour = None
                best_obj = None

                for a in range(loop_max):
                    result = sovle_annering(model, client, 1, 5000)
                    x_values = result.best.values
                    solution = x.evaluate(x_values)
                    sequence = onehot2sequence(solution)
                    candidate_tour = process_sequence(sequence)
                    cost_val = result.solutions[0].objective

            # 条件に応じて更新(ここでは最初の解を使う例)
                    best_tour = candidate_tour
                    best_obj = cost_val
                    break

                best_obj = best_obj / 1000.0  # メートル→キロメートル
                best_obj = round(best_obj, 1)  # 小数点第1位まで

        # 結果をセッションステートに保存
                st.session_state["best_tour"] = best_tour
                st.session_state["best_cost"] = best_obj
                st.session_state["annering_param"]=annering_param
                st.session_state['redraw'] = True
            
            st.success("処理が完了しました！")
# ========== 出力 ==========
if st.session_state['best_tour'] !=None:
  annering_param=st.session_state["annering_param"]
  best_obj=st.session_state['best_cost']
  best_tour=st.session_state['best_tour']
  gis_st.write(f"#### 計算結果: 総距離: {best_obj} km")
  distance_matrix=annering_param['distance_matrix']
  demand=annering_param['demand']
  for item in best_tour.items():
     distance=0
     weight=0
     p_node=""
     for i in range(len(item[1])-1):
        it=item[1][i]
        itn=item[1][i+1]
        distance += distance_matrix[it][itn]
        weight += demand[it]
        p_node += f'{get_point_name(df,re_node_list[it])} ⇒ '
     
     it=item[1][len(item[1])-1]
     p_node += f'{get_point_name(df,re_node_list[it])}'
     r_str=f"ルート{item[0]} (走行距離:{distance/1000:.2f}km/配送量:{weight/1000*4:.2f}t)  \n【拠点】{p_node}"
     gis_st.write(r_str)
  #best_tour_markdown = "\n".join([f"{key}: {value}" for key, value in best_tour.items()])
  #gis_st.markdown(best_tour_markdown)

if st.session_state['redraw'] !=False:
  st.rerun()
