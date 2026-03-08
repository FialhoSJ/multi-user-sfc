import xml.etree.ElementTree as ET
import os
# import sys

# sys.path.append("..")



from scipy.spatial import cKDTree


PATH_TRACE = os.path.join("traces", "sumoTraceVehicle.xml")


def generate_trace(substrate_network):
    voronoi_kdtree = cKDTree(edges_position)
    vehicles_dict = {}
    tree = ET.parse(PATH_TRACE)
    root = tree.getroot()
    timesteps = root.findall("timestep")
    for time_obj in timesteps:
        timestamp = time_obj.attrib["time"]
        if timestamp == "100.00":
            break
        vehicles = time_obj.getchildren()
        # vehicles_attr = {'x': [],'y': [] ,'speed': [], 'id': [], 'position':[], 'region':[]}
        vehicles_attr = {"speed": [], "id": [], "position": [], "region": []}
        for vehicle in vehicles:
            vehicle_id = int(vehicle.attrib["id"])
            if vehicle_id > 400:
                break
            vehicles_attr["id"].append(vehicle_id)
            vehicles_attr["speed"].append(float(vehicle.attrib["speed"]))
            vehicles_attr["position"].append(
                [float(vehicle.attrib["x"]), float(vehicle.attrib["y"])]
            )
        test_point_dist, test_point_regions = voronoi_kdtree.query(vehicles_attr["position"])
        vehicles_attr["region"] = list(test_point_regions)
        vehicles_dict[timestamp] = vehicles_attr

    print(vehicles_dict["99.00"]["region"])

    return vehicles_dict


"""
if __name__ == "__main__":
    #topology = PaloAlto()
    #substrate_network = topology.generate_substrate_network()
    edges_position = []
    for node in substrate_network.nodes():
        #skip cloud
        if node == 0:
            continue
        position = substrate_network.get_node_position(node)
        arr = np.asarray(position)
        edges_position.append(arr)
    edges_position = np.vstack(edges_position)
    vor = Voronoi(edges_position)
    fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',
                line_width=2, line_alpha=0.6, point_size=2)
    plt.show()
    generate_trace(edges_position)
"""
