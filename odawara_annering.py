# ディレクトリ設定

root_dir = "./"

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


client = FixstarsClient()
client.token = "AE/KuJSebaDc4MXYFwp4I8YtgvDMXDBUiKE"  # 有効なトークンを設定

# データ読込
node_data =  "kyoten_geocode_Revised.json"
df = pd.read_json(root_dir + node_data)   #拠点データ

numOfPeople = "number_of_people.csv"
np_df = pd.read_csv(root_dir + numOfPeople) #人数データ

distance_matrix_file = "distance_matrix_v2.csv"
distance_matrix = np.loadtxt(root_dir + distance_matrix_file, delimiter=",")   #距離行列

route_file = "path_list_v2.json"
path_df = pd.read_json(root_dir + route_file)   #道路データ
node_name_list = path_df['start_node'].drop_duplicates().to_list()

op_data_file = "op_base8-38.json"
jf = open(root_dir + op_data_file, 'r')
op_data = json.load(jf)
jf.close()

# 神奈川県_市区町村の行政区域データ読み込み(小田原市のみ抽出)
geojson_path = root_dir + N03-20240101_14.geojson"
administrative_district = gpd.read_file(geojson_path)
odawara_district = administrative_district[administrative_district["N03_004"]=="小田原市"]

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


########################################
# Folium を使う表示系関数
########################################

def disp_odawaraMap(center=[35.2646012,139.15223698], zoom_start=12):
    m = folium.Map(
        location=center,
        tiles='https://cyberjapandata.gsi.go.jp/xyz/std/{z}/{x}/{y}.png',
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


def draw_route(m, G, best_routes, path_df, node_name_list):
    for k, vehicle_route in best_routes.items():
        layer = folium.FeatureGroup(name=f"ルート {k}")
        layer.add_to(m)
        for iv in range(len(vehicle_route) - 1):
            start_node = node_name_list[vehicle_route[iv]]
            goal_node = node_name_list[vehicle_route[iv + 1]]
            route = path_df[(path_df['start_node'] == start_node) & (path_df['goal_node'] == goal_node)]['route']
            for route_nodes in route:
                ox.plot_route_folium(
                    G,
                    route_nodes,
                    route_map=layer,
                    color=_colors[k % len(_colors)],
                    weight=10.0,
                    opacity=0.5,
                )
    folium.LayerControl().add_to(m)
    return m

########################################
# アニーリング周り(以前の関数群)
########################################

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

def set_parameter(np_df, path_df, op_data):
    annering_param = {}

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
    demand_mean = np.mean(demand[8:])

    capacity = int(demand_max) + int(demand_mean) * (avg_nbase_per_vehicle)

    annering_param['distance_matrix'] = distance_matrix
    annering_param['n_transport_base'] = n_transport_base
    annering_param['n_shellter'] = n_shellter
    annering_param['nbase'] = nbase
    annering_param['nvehicle'] = nvehicle
    annering_param['capacity'] = capacity
    annering_param['demand'] = demand

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

st.set_page_config(
    page_title="小田原市 周辺",
    page_icon="🗾",
    layout="wide"
)

# --- セッションステートで計算結果を保持
if "best_tour" not in st.session_state:
    st.session_state["best_tour"] = None
if "best_cost" not in st.session_state:
    st.session_state["best_cost"] = None

st.title("避難所・仮設救護所の選択")

# ベースマップ作成 & マーカー描画
base_map = disp_odawaraMap()
plot_marker(base_map, df)

# ボタン押下前にbase_mapを表示
st_folium(base_map, use_container_width=True, height=400)

# ユーザーが施設を選択
facility_list = df["施設名"].tolist()
selected_nodes = st.multiselect(
    "開設・使用されている場所を選んでください",
    options=facility_list,
    default=facility_list
)

st.write("選択完了後、下のボタンを押してください。")

# グラフ
G = ox.graph_from_place({'city': 'Odawara', 'state': 'Kanagawa', 'country': 'Japan'}, network_type='drive')

if st.button("最適経路探索開始"):
    if not selected_nodes:
        st.warning("場所を1つ以上選択してください")
    else:
        # ここでアニーリング等を実行
        annering_param = set_parameter(np_df, path_df, op_data)
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
        
        # best_objをキロメートルに変換
        best_obj = best_obj / 1000.0  # メートル→キロメートル
        best_obj = round(best_obj, 1)  # 小数点第1位まで

        # 結果をセッションステートに保存
        st.session_state["best_tour"] = best_tour
        st.session_state["best_cost"] = best_obj

        # ========== 出力 ==========
        st.write(f"総距離: {best_obj} km")
        st.write(*best_tour.items(), sep="\n")

        # 新しいマップを作りルートを描画して表示
        annering_map = disp_odawaraMap()
        plot_marker(annering_map, df)
        annering_map = draw_route(annering_map, G, best_tour, path_df, node_name_list)
        st_folium(annering_map, use_container_width=True, height=600)

        st.success("最適経路探索が完了しました！")

# --- ページ再実行時でも、セッションステートがあれば結果を表示
if st.session_state["best_tour"] is not None:
    st.write("\n---\n## 計算結果:")
    st.write(f"総距離: {st.session_state['best_cost']} km")
    st.write(st.session_state["best_tour"])

    # 地図を再描画
    map2 = disp_odawaraMap()
    plot_marker(map2, df)
    map2 = draw_route(map2, G, st.session_state["best_tour"], path_df, node_name_list)
    st_folium(map2, use_container_width=True, height=600)