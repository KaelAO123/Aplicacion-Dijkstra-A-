from flask import Flask, render_template, request
import osmnx as ox
import heapq
import pandas as pd
from geopy.geocoders import Nominatim
import folium

app = Flask(__name__)

# Inicializar el geocodificador
geolocator = Nominatim(user_agent="geoapiExercises")

# Funciones existentes
def get_node_coordinates(G, node_id):
    node_data = G.nodes[node_id]
    return obtener_nombre_lugar(node_data['y'], node_data['x'])

def obtener_nombre_lugar(lat, lon):
    location = geolocator.reverse((lat, lon), exactly_one=True)
    if location:
        return location.address
    else:
        return "Nombre desconocido"

def parse_maxspeed(maxspeed):
    if isinstance(maxspeed, list):
        speeds = []
        for speed in maxspeed:
            try:
                speeds.append(int(speed))
            except ValueError:
                speeds.append(int(speed.split()[0]))
        return min(speeds)
    elif isinstance(maxspeed, str):
        try:
            return int(maxspeed)
        except ValueError:
            return int(maxspeed.split()[0])
    else:
        return maxspeed

# Funciones de estilo y plot (modificadas para Folium)
def style_unvisited_edge(u, v, key, G):
    G.edges[u, v, key]["color"] = "#00a000"
    G.edges[u, v, key]["alpha"] = 0.2
    G.edges[u, v, key]["linewidth"] = 0.5

def style_visited_edge(u, v, key, G):
    G.edges[u, v, key]["color"] = "#00a000"
    G.edges[u, v, key]["alpha"] = 1
    G.edges[u, v, key]["linewidth"] = 1

def style_active_edge(u, v, key, G):
    G.edges[u, v, key]["color"] = '#00ff00'
    G.edges[u, v, key]["alpha"] = 1
    G.edges[u, v, key]["linewidth"] = 1

def style_path_edge(u, v, key, G):
    G.edges[u, v, key]["color"] = "white"
    G.edges[u, v, key]["alpha"] = 1
    G.edges[u, v, key]["linewidth"] = 1

def distance(node1, node2, G):
    x1, y1 = G.nodes[node1]["x"], G.nodes[node1]["y"]
    x2, y2 = G.nodes[node2]["x"], G.nodes[node2]["y"]
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

def get_nearest_node(G, place_name):
    geocode_result = ox.geocode(place_name)
    nearest_node = ox.distance.nearest_nodes(G, X=geocode_result[1], Y=geocode_result[0])
    return nearest_node

# Inicializar el grafo globalmente para reutilizarlo
place_name = "La Paz, Bolivia"
G = ox.graph_from_place(place_name, network_type="drive")

# Limpiar y añadir atributos al grafo
for u, v, key in G.edges(keys=True):
    maxspeed = 40
    if "maxspeed" in G.edges[u, v, key]:
        maxspeed = G.edges[u, v, key]["maxspeed"]
        maxspeed = parse_maxspeed(maxspeed)
    G.edges[u, v, key]["maxspeed"] = maxspeed
    G.edges[u, v, key]["weight"] = G.edges[u, v, key]["length"] / maxspeed

# Funciones de enrutamiento
def dijkstra(orig, dest, G):
    for node in G.nodes:
        G.nodes[node]["visited"] = False
        G.nodes[node]["distance"] = float("inf")
        G.nodes[node]["previous"] = None
        G.nodes[node]["size"] = 0

    for u, v, key in G.edges(keys=True):
        style_unvisited_edge(u, v, key, G)

    G.nodes[orig]["distance"] = 0
    G.nodes[orig]["size"] = 50
    G.nodes[dest]["size"] = 50

    pq = [(0, orig)]
    step = 0

    while pq:
        _, node = heapq.heappop(pq)

        if node == dest:
            return step  # Retorna el número de iteraciones

        if G.nodes[node]["visited"]:
            continue
        G.nodes[node]["visited"] = True

        for u, v, key in G.out_edges(node, keys=True):
            style_visited_edge(u, v, key, G)
            neighbor = v
            weight = G.edges[u, v, key]["weight"]

            if G.nodes[neighbor]["distance"] > G.nodes[node]["distance"] + weight:
                G.nodes[neighbor]["distance"] = G.nodes[node]["distance"] + weight
                G.nodes[neighbor]["previous"] = node
                heapq.heappush(pq, (G.nodes[neighbor]["distance"], neighbor))
                for u2, v2, key2 in G.out_edges(neighbor, keys=True):
                    style_active_edge(u2, v2, key2, G)
        step += 1

    return step  # Si no se encuentra el destino

def a_star(orig, dest, G):
    for node in G.nodes:
        G.nodes[node]["previous"] = None
        G.nodes[node]["size"] = 0
        G.nodes[node]["g_score"] = float("inf")
        G.nodes[node]["f_score"] = float("inf")

    for u, v, key in G.edges(keys=True):
        style_unvisited_edge(u, v, key, G)

    G.nodes[orig]["size"] = 50
    G.nodes[dest]["size"] = 50
    G.nodes[orig]["g_score"] = 0
    G.nodes[orig]["f_score"] = distance(orig, dest, G)

    pq = [(G.nodes[orig]["f_score"], orig)]
    step = 0

    while pq:
        _, node = heapq.heappop(pq)
        if node == dest:
            return step  # Retorna el número de iteraciones

        for u, v, key in G.out_edges(node, keys=True):
            style_visited_edge(u, v, key, G)
            neighbor = v
            tentative_g_score = G.nodes[node]["g_score"] + distance(node, neighbor, G)
            if tentative_g_score < G.nodes[neighbor]["g_score"]:
                G.nodes[neighbor]["previous"] = node
                G.nodes[neighbor]["g_score"] = tentative_g_score
                G.nodes[neighbor]["f_score"] = tentative_g_score + distance(neighbor, dest, G)
                heapq.heappush(pq, (G.nodes[neighbor]["f_score"], neighbor))
                for u2, v2, key2 in G.out_edges(neighbor, keys=True):
                    style_active_edge(u2, v2, key2, G)
        step += 1

    return step  # Si no se encuentra el destino

def reconstruct_path(orig, dest, G, algorithm=None):
    for u, v, key in G.edges(keys=True):
        style_unvisited_edge(u, v, key, G)

    dist = 0
    speeds = []
    curr = dest
    path = []
    while curr != orig:
        prev = G.nodes[curr]["previous"]
        if prev is None:
            break  # No hay camino
        # Obtener la primera clave disponible para la arista (prev, curr)
        keys = list(G[prev][curr].keys())
        if not keys:
            break  # No hay aristas entre prev y curr
        key = keys[0]
        path.append((prev, curr, key))
        dist += G.edges[prev, curr, key]["length"]
        speeds.append(G.edges[prev, curr, key]["maxspeed"])
        style_path_edge(prev, curr, key, G)
        if algorithm:
            G.edges[prev, curr, key][f"{algorithm}_uses"] = G.edges[prev, curr, key].get(f"{algorithm}_uses", 0) + 1
        curr = prev

    dist /= 1000
    if speeds:
        velocidad_promedio = sum(speeds) / len(speeds)
        tiempo_total = (dist / velocidad_promedio) * 60  # en minutos
    else:
        velocidad_promedio = 0
        tiempo_total = 0

    # Crear el mapa con Folium
    orig_coord = (G.nodes[orig]['y'], G.nodes[orig]['x'])
    dest_coord = (G.nodes[dest]['y'], G.nodes[dest]['x'])
    m = folium.Map(location=orig_coord, zoom_start=13)

    # Añadir el camino al mapa
    path_coords = [(G.nodes[u]['y'], G.nodes[u]['x']) for u, v, key in reversed(path)]
    path_coords.append((G.nodes[dest]['y'], G.nodes[dest]['x']))
    folium.PolyLine(path_coords, color="blue", weight=5).add_to(m)

    # Añadir marcadores de origen y destino
    folium.Marker(orig_coord, popup="Origen", icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(dest_coord, popup="Destino", icon=folium.Icon(color='red')).add_to(m)

    # Guardar el mapa en un HTML temporal
    map_html = m._repr_html_()

    return {
        'distancia': f"{dist:.2f} km",
        'velocidad_promedio': f"{velocidad_promedio:.2f} km/h",
        'tiempo_total': f"{tiempo_total:.2f} min",
        'mapa': map_html
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        start_place = request.form.get('start')
        end_place = request.form.get('end')

        # Inicializar el diccionario de comparación localmente
        comparacionDict = {
            'Algoritmo': ['Dijkstra', 'A*'],
            'Distancia': [],
            'Numero_de_Iteraciones': [],
            'Velocidad_Promedio': [],
            'Tiempo_Total': []
        }

        try:
            start_node = get_nearest_node(G, start_place)
            end_node = get_nearest_node(G, end_place)

            # Ejecutar Dijkstra
            iter_dijkstra = dijkstra(start_node, end_node, G)
            path_dijkstra = reconstruct_path(start_node, end_node, G, algorithm="Dijkstra")

            comparacionDict['Distancia'].append(path_dijkstra['distancia'])
            comparacionDict['Numero_de_Iteraciones'].append(iter_dijkstra)
            comparacionDict['Velocidad_Promedio'].append(path_dijkstra['velocidad_promedio'])
            comparacionDict['Tiempo_Total'].append(path_dijkstra['tiempo_total'])

            # Ejecutar A*
            iter_a_star = a_star(start_node, end_node, G)
            path_a_star = reconstruct_path(start_node, end_node, G, algorithm="A*")

            comparacionDict['Distancia'].append(path_a_star['distancia'])
            comparacionDict['Numero_de_Iteraciones'].append(iter_a_star)
            comparacionDict['Velocidad_Promedio'].append(path_a_star['velocidad_promedio'])
            comparacionDict['Tiempo_Total'].append(path_a_star['tiempo_total'])

            # Crear el DataFrame
            df = pd.DataFrame(comparacionDict)
            tabla_comparacion = df.to_html(classes='table table-striped', index=False)

            # Para simplificar, mostramos el mapa de A* como ejemplo
            mapa = path_a_star['mapa']

            return render_template('resultados.html', tabla=tabla_comparacion, mapa=mapa)
        except Exception as e:
            return f"Error: {e}"

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
