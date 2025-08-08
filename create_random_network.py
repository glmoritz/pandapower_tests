import pandapower as pp
import pandas as pd
import random
from enum import Enum
import random
import math
from shapely.geometry import Point as ShapelyPoint, LineString, MultiLineString, MultiPoint as ShapelyMultipoint
from shapely.strtree import STRtree
from shapely import affinity
from shapely import distance as shapely_distance
from geopy.distance import distance
from geopy.point import Point as GeopyPoint
from collections import deque
import networkx as nx
import numpy as np


class NetworkType(Enum):
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    RESIDENTIAL = "residential"

def generate_pandapower_net(CommercialRange, IndustrialRange, ResidencialRange,                            
                            ForkLengthRange, LineBusesRange, LineForksRange,
                            mv_bus_coordinates=(0.0, 0.0),
                            ):    
    random_net_lv = pp.create_empty_network()

    
    #random.seed(3333)

    graph = nx.DiGraph()

    # Linedata
    # UG1
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 0.162,
                 'x_ohm_per_km': 0.0832, 'max_i_ka': 1.0,
                 'type': 'cs'}
    pp.create_std_type(random_net_lv, line_data, name='UG1', element='line')

    # UG2
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 0.2647,
                 'x_ohm_per_km': 0.0823, 'max_i_ka': 1.0,
                 'type': 'cs'}
    pp.create_std_type(random_net_lv, line_data, name='UG2', element='line')

    # UG3
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 0.822,
                 'x_ohm_per_km': 0.0847, 'max_i_ka': 1.0,
                 'type': 'cs'}
    pp.create_std_type(random_net_lv, line_data, name='UG3', element='line')

    # OH1
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 0.4917,
                 'x_ohm_per_km': 0.2847, 'max_i_ka': 1.0,
                 'type': 'ol'}
    pp.create_std_type(random_net_lv, line_data, name='OH1', element='line')

    # OH2
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 1.3207,
                 'x_ohm_per_km': 0.321, 'max_i_ka': 1.0,
                 'type': 'ol'}
    pp.create_std_type(random_net_lv, line_data, name='OH2', element='line')

    # OH3
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 2.0167,
                 'x_ohm_per_km': 0.3343, 'max_i_ka': 1.0,
                 'type': 'ol'}
    pp.create_std_type(random_net_lv, line_data, name='OH3', element='line')

    # Add the main 20 kV bus (medium voltage)
    mv_bus = pp.create_bus(random_net_lv, name='MV Bus', vn_kv=20.0, type='b', zone='CIGRE_LV')
    
    # External grid
    pp.create_ext_grid(random_net_lv, mv_bus, vm_pu=1.0, va_degree=0.0, s_sc_max_mva=100.0,
                    s_sc_min_mva=100.0, rx_max=1.0, rx_min=1.0)



    def create_cigre_lv_transformer(hv_bus, lv_bus, name, transformer_type):
        if transformer_type == NetworkType.RESIDENTIAL:
            pp.create_transformer_from_parameters(random_net_lv, hv_bus, lv_bus, sn_mva=0.5, vn_hv_kv=20.0,
                                       vn_lv_kv=0.4, vkr_percent=1.0, vk_percent=4.123106,
                                       pfe_kw=0.0, i0_percent=0.0, shift_degree=30,
                                       tap_pos=0.0, name=name)
        elif transformer_type == NetworkType.INDUSTRIAL:
            pp.create_transformer_from_parameters(random_net_lv, hv_bus, lv_bus, sn_mva=0.15, vn_hv_kv=20.0,
                                            vn_lv_kv=0.4, vkr_percent=1.003125, vk_percent=4.126896,
                                            pfe_kw=0.0, i0_percent=0.0, shift_degree=30,
                                            tap_pos=0.0, name=name)
        elif transformer_type == NetworkType.COMMERCIAL:
            pp.create_transformer_from_parameters(random_net_lv, hv_bus, lv_bus, sn_mva=0.3, vn_hv_kv=20.0,
                                            vn_lv_kv=0.4, vkr_percent=0.993750, vk_percent=4.115529,
                                            pfe_kw=0.0, i0_percent=0.0, shift_degree=30,
                                            tap_pos=0.0, name=name)

    def add_transformer_branch(transformer_type, branch_index, parent_bus, current_net, graph):
        # Create LV bus (0.4 kV)        
        working_bus_name = f"{str(transformer_type)}_Branch{branch_index}_trafo_lv_bus"
        working_bus = pp.create_bus(current_net, vn_kv=0.4, name=working_bus_name)        

        # Add transformer from MV to LV
        create_cigre_lv_transformer(parent_bus, working_bus, f"{str(transformer_type)}_{branch_index}_Trafo", transformer_type)
        graph.add_edge(parent_bus, working_bus)    
        graph[parent_bus][working_bus]['type'] = 'transformer'
                
        #randomize how many forks this line will be
        Forks = random.randint(*LineForksRange)
        
        #randomize how many buses will be in this branch
        NumBuses = random.randint(*LineBusesRange)
        
        #distribute NumBuses across the forks
        fork_bus_counts = [1] * Forks  # start with 1 bus per fork
        remaining = NumBuses - Forks
        for _ in range(remaining):
            fork_bus_counts[random.randint(0, Forks - 1)] += 1
        
        added_busses = []        
        
        # Create branches/forks       
        
        for f in range(Forks):
                        
            for b in range(fork_bus_counts[f]):
                #create the bus
                new_bus_name = f"{str(transformer_type)}_Branch{branch_index}_fork{f}bus{b}"
                new_bus = pp.create_bus(current_net, name=new_bus_name, vn_kv=0.4, type='m', zone='CIGRE_LV')
                
                graph.add_edge(working_bus, new_bus)  #the lines can only be created after all buses are positioned (only then I will know its lengths)                          
                graph[working_bus][new_bus]['type'] = 'lv_line'
                added_busses.append(new_bus)    

                working_bus = new_bus
                working_bus_name = new_bus_name

            #pick a random child bus to connect the next fork
            if f < Forks - 1:
                working_bus = random.choice(added_busses)
            else:
                working_bus = parent_bus    
            
            # Retrieve the parent_bus_name from net_cigre_lv
            working_bus_name = random_net_lv.bus.at[parent_bus, 'name']
        
        return added_busses

    # Create transformers by type
    Ct = random.randint(*CommercialRange)
    It = random.randint(*IndustrialRange)
    Rt = random.randint(*ResidencialRange)

    for i in range(Ct):
        new_lines = add_transformer_branch(
            transformer_type=NetworkType.COMMERCIAL,
            branch_index=i,
            parent_bus=mv_bus,
            current_net= random_net_lv,
            graph=graph            
            )        
    for i in range(It):
        new_lines = add_transformer_branch(
            transformer_type=NetworkType.INDUSTRIAL,
            branch_index=i,
            parent_bus=mv_bus,
            current_net= random_net_lv,
            graph=graph                        
            )    
    for i in range(Rt):
        new_lines = add_transformer_branch(
            transformer_type=NetworkType.RESIDENTIAL,
            branch_index=i,
            parent_bus=mv_bus,
            current_net= random_net_lv,
            graph=graph                        
            )        

    success = generate_network_coordinates(
        net=random_net_lv,
        root_bus_index=mv_bus,        
        graph=graph,
        min_distance_m=5,  # Minimum distance between points
        max_attempts=50  # Maximum attempts to place a point
    )    
    
    # Initialize bus_geodata if it doesn't exist
    if random_net_lv.bus_geodata.empty:
        random_net_lv.bus_geodata = pd.DataFrame(columns=['x', 'y'])

    for node in graph.nodes:
        # Get XY in graph
        node_coords = graph.nodes[node]['coordinates']
        
        # Convert XY offset into lat/lon
        point_lat = distance(meters=node_coords.y).destination(mv_bus_coordinates, bearing=0).latitude
        point_lon = distance(meters=node_coords.x).destination(mv_bus_coordinates, bearing=0).longitude

        final_point = GeopyPoint(point_lat, point_lon)        

        lat, lon = final_point.latitude, final_point.longitude

        #Update bus_geodata with local XY
        random_net_lv.bus_geodata.loc[node, 'x'] = node_coords.x
        random_net_lv.bus_geodata.loc[node, 'y'] = node_coords.y

        #update bus table with lat/lon
        random_net_lv.bus.loc[node, 'lat'] = lat
        random_net_lv.bus.loc[node, 'lon'] = lon

        #update graph node with latlon
        graph.nodes[node]['latlon'] = GeopyPoint(lat, lon)

    #now, use the graph to create the lines
    for bus_u, bus_v in graph.edges:
        # Compute Euclidean distance (assuming coordinates in meters)
        length = graph.nodes[bus_u]['coordinates'].distance(graph.nodes[bus_v]['coordinates']) / 1000.0  #distance in km        

        if graph[bus_u][bus_v]['type'] == 'lv_line':
            # Create line in pandapower
            pp.create_line(
                random_net_lv,
                from_bus=bus_u,
                to_bus=bus_v,
                length_km=length,
                std_type="OH1",
                name=f"Line {random_net_lv.bus.at[bus_u, 'name']}-{random_net_lv.bus.at[bus_v, 'name']}"
            )

    return random_net_lv, graph


def arc_span(arc):
    """Calculate the angle span of an arc."""
    start, stop = normalize_arc(*arc)
    return (stop - start) % (2 * math.pi)


def normalize_arc(start, stop):
    """Normalize arc so that start in [0, 2π), and stop >= start."""
    start %= (2*math.pi)
    stop  %= (2*math.pi)
    if stop <= start:
        stop += (2*math.pi)
    return start, stop

def complement_arc(arc):
    """
    Returns the complement of an arc on the unit circle as a list of 0, 1, or 2 arcs (in radians).
    The complement is the set of points not included in the input arc.
    """
    start, stop = normalize_arc(*arc)
    if (stop - start) % (2 * math.pi) == 0:
        # Full circle, complement is empty
        return []
    elif stop > start:
        return [normalize_arc(stop % (2 * math.pi), start % (2 * math.pi))]
    else:
        # Should not happen due to normalize_arc, but handle for safety
        return [normalize_arc(stop, start)]

def angle_in_arc(angle, arc):
    """Check if angle is within arc (start, stop) on the unit circle.

    Arc is defined with start <= stop (stop may be > 2π if wrapping occurs).
    The check is inclusive of start, exclusive of stop: [start, stop).
    """
    start, stop = normalize_arc(*arc)
    angle = angle % (2 * math.pi)
    angle = angle if angle >= start else angle + 2 * math.pi
    return start <= angle < stop

def add_arc(arcs, new_arc):
    """
    Adds a new arc to a list of arcs, merging overlapping or adjacent arcs.
    All arcs are tuples (start, stop) in radians.
    Returns a new list of arcs.
    """
    if not arcs:
        return [normalize_arc(*new_arc)]
    new_arc = normalize_arc(*new_arc)
    result = []
    added = False
    for arc in arcs:
        arc = normalize_arc(*arc)
        # Check if arcs overlap or touch
        if (angle_in_arc(new_arc[0], arc) or angle_in_arc(new_arc[1] - 1e-12, arc) or
            angle_in_arc(arc[0], new_arc) or angle_in_arc(arc[1] - 1e-12, new_arc)):
            # Merge arcs
            merged = union_arcs(arc, new_arc)
            # If merged into one arc, continue merging with others
            if len(merged) == 1:
                new_arc = merged[0]
                added = True
            else:
                # If not merged, keep both
                result.extend(merged)
                added = True
        else:
            result.append(arc)
    if not added:
        result.append(new_arc)
    # After possible merges, sort and merge again if needed
    result = sorted([normalize_arc(*a) for a in result], key=lambda x: x[0])
    merged_result = []
    for arc in result:
        if not merged_result:
            merged_result.append(arc)
        else:
            last = merged_result[-1]
            # Try to merge with last
            merged = union_arcs(last, arc)
            if len(merged) == 1:
                merged_result[-1] = merged[0]
            else:
                merged_result.append(arc)
    return [normalize_arc(*arc) for arc in merged_result if arc_span(arc) > 1e-12]

def subtract_arc(arc_a, arc_b):
    """
    Subtract arc_b from arc_a on the unit circle.
    Returns a list of 0, 1, or 2 arcs (in radians) representing the result.
    """
    a_start, a_stop = normalize_arc(*arc_a)
    b_start, b_stop = normalize_arc(*arc_b)

    # Normalize arcs to [0, 2π)
    a_span = (a_start, a_stop)
    b_span = (b_start, b_stop)

    # If arc_b fully covers arc_a, result is empty
    if angle_in_arc(a_start, b_span) and angle_in_arc(a_stop - 1e-12, b_span):
        return []

    # If arc_b does not overlap arc_a, return arc_a
    if intersect_arcs(a_span, b_span) == []:
        return [normalize_arc(*a_span)]

    result = []
    # If b_start is inside a, add [a_start, b_start]
    if angle_in_arc(b_start, a_span) and not math.isclose(a_start, b_start):
        result.append((a_start, b_start))
    # If b_stop is inside a, add [b_stop, a_stop]
    if angle_in_arc(b_stop, a_span) and not math.isclose(a_stop, b_stop):
        result.append((b_stop, a_stop))
    return [normalize_arc(*arc) for arc in result if arc_span(arc) > 1e-12]

def union_arcs(a, b):
    """Return the union of two arcs as a list of 1 or 2 arcs (in radians)."""
    a_start, a_stop = normalize_arc(*a)
    b_start, b_stop = normalize_arc(*b)
    
    # Sort arcs by start angle
    arcs = sorted([(a_start, a_stop), (b_start, b_stop)])

    # Merge if overlapping or touching
    if arcs[0][1] >= arcs[1][0]:
        merged = (arcs[0][0], max(arcs[0][1], arcs[1][1]))
        return [normalize_arc(merged[0] , merged[1])]
    else:
        return [normalize_arc(arcs[0][0], arcs[0][1]), normalize_arc(arcs[1][0], arcs[1][1])]

def intersect_arcs(a, b):
    """Return the intersection of two arcs as a list of 0 or 1 arcs (in radians)."""
    a_start, a_stop = normalize_arc(*a)
    b_start, b_stop = normalize_arc(*b)

    # Unroll arcs into [a_start, a_stop) and [b_start, b_stop)
    start = max(a_start, b_start)
    stop  = min(a_stop, b_stop)

    if start < stop:
        #return [(start % (2*math.pi), stop % (2*math.pi))]
        return [normalize_arc(start, stop)]
    else:
        return []  # no overlap

def compute_nonoverlapping_subarcs(parent1, parent2, graph):    
    theta1 = graph.nodes[parent1]['angle']
    theta2 = graph.nodes[parent2]['angle']
    shared_arc = union_arcs(graph.nodes[parent1]['arc_span'],graph.nodes[parent2]['arc_span'])
    theta3 = shared_arc[0][0]
    theta4 = shared_arc[0][1]
    

    # Max possible half-widths
    w1 = min(arc_span((theta3,theta1)), arc_span((theta1,theta4)))
    w2 = min(arc_span((theta3,theta2)), arc_span((theta2,theta4)))
    
    max_alpha1 = 2 * w1
    max_alpha2 = 2 * w2
    
    available_between = 2 * arc_span((theta1,theta2))
    
    if max_alpha1 + max_alpha2 <= available_between:
        alpha1 = max_alpha1
        alpha2 = max_alpha2
    else:
        total_weight = w1 + w2
        alpha1 = available_between * (w1 / total_weight)
        alpha2 = available_between * (w2 / total_weight)
    
    arc1_start = theta1 - alpha1 / 2
    arc1_end = theta1 + alpha1 / 2
    arc2_start = theta2 - alpha2 / 2
    arc2_end = theta2 + alpha2 / 2
    
    return (
        normalize_arc(arc1_start, arc1_end),
        normalize_arc(arc2_start, arc2_end)
    )

def polar_to_cartesian(radius, angle):
    return radius * math.cos(angle), radius * math.sin(angle)

def get_vertex_coordinate(vertex, graph):
    """Get the Cartesian coordinates of a vertex in the graph."""
    if vertex not in graph.nodes:
        raise ValueError(f"Vertex {vertex} not found in the graph.")
    
    if graph.nodes[vertex]['coordinates'] is None:
        radius = graph.nodes[vertex]['radius']
        angle = graph.nodes[vertex]['angle']
        graph.nodes[vertex]['coordinates'] = polar_to_cartesian(radius, angle)
    return graph.nodes[vertex]['coordinates']

def randomize_position(vertex, graph, max_angle, max_distance):
    delta_theta = random.gauss(0, max_angle)    
    
    
        
    parent = list(graph.predecessors(vertex))[0] if list(graph.predecessors(vertex)) else None

    #node -> P2, parent -> P1
    p2 = get_vertex_coordinate(vertex, graph)
    p1 = get_vertex_coordinate(parent, graph) if parent else ShapelyPoint(0, 0)    
    r = shapely_distance(p1,p2)

    # 2. Calculate the angle alpha of the line P1 P2 relative to the positive x-axis
    # math.atan2(y, x) handles all quadrants correctly.
    theta = math.atan2(p2.y - p1.y, p2.x - p1.x)

    # 3. Calculate the new angle alpha_prime
    # The new point P2' will be at an angle dtheta relative to the original P1P2 line.
    theta_prime = theta + delta_theta

    # 4. Calculate the new distance d_prime from P1 to P2'
    r_prime = 0
    while r_prime < 50:
        delta_r = random.gauss(0, max_distance)
        r_prime = r + delta_r

    # 5. Calculate the Cartesian coordinates of the new point P2_prime
    dx = r_prime * math.cos(theta_prime)
    dy = r_prime * math.sin(theta_prime)

    p2prime = ShapelyPoint(p1.x+dx,p1.y+dy)
    dp2 = ShapelyPoint(p2prime.x - p2.x, p2prime.y - p2.y)

    # 6. Convert P2_prime back to polar coordinates (r2_prime, theta2_prime)
    r2_prime = shapely_distance(p2prime,ShapelyPoint(0,0))
    theta2_prime = math.atan2(p2prime.y, p2prime.x)

    return p2prime,dp2.x,dp2.y,delta_r,delta_theta
    
 
def build_multipoint_from_nodes(nodes, graph) -> ShapelyMultipoint:
    """Create MultiPoint geometry from graph nodes"""
    points = []
    for node in nodes:
        try:
            coord = get_vertex_coordinate(node, graph)
            if coord:
                points.append(coord)
        except KeyError:
            continue
    return ShapelyMultipoint(points) if points else ShapelyMultipoint()


def build_multilines_from_edges(edges, graph):
    lines = []
    for src, dst in edges:        
        p1 = get_vertex_coordinate(src, graph)
        p2 = get_vertex_coordinate(dst, graph)
        lines.append(LineString([p1, p2]))
    return MultiLineString(lines)


def randomize_branch_positions(graph, vertex_to_randomize, max_trials=50):
    subtree_nodes = set(nx.descendants(graph, vertex_to_randomize)) | {vertex_to_randomize}
    main_nodes = set(graph.nodes) - subtree_nodes

    main_edges = list(graph.subgraph(main_nodes).edges)
    branch_edges = list(graph.subgraph(subtree_nodes).edges)
        
    # Build geometries
    main_geom = build_multilines_from_edges(main_edges, graph)
    branch_geom = build_multilines_from_edges(branch_edges, graph)
    node_branch_geom = build_multipoint_from_nodes(subtree_nodes, graph)

    # Current root position (for connection constraint)    
    parent_pos = get_vertex_coordinate(list(graph.predecessors(vertex_to_randomize))[0], graph) if list(graph.predecessors(vertex_to_randomize)) else ShapelyPoint(0, 0)    
    original_coords = np.array([graph.nodes[node]['coordinates'] for node in subtree_nodes])

    for trial in range(max_trials):
        # New random position
        new_vertex_pos,dx,dy,dr,dtheta =  randomize_position(vertex_to_randomize, graph, (20.0*math.pi)/180, 200.0)

        # Apply transforms to geometry
        transformed_branch = affinity.rotate(branch_geom, math.degrees(dtheta), origin=parent_pos)
        transformed_branch = affinity.translate(transformed_branch, dx, dy)

        node_transformed_branch = affinity.rotate(node_branch_geom, math.degrees(dtheta), origin=parent_pos)
        node_transformed_branch = affinity.translate(node_transformed_branch, dx, dy)  # Buffer to avoid small overlaps

        #this is the new edge from parent to the randomized vertex
        connection_line = LineString([parent_pos, new_vertex_pos])

        # Check overlap
        if transformed_branch.crosses(main_geom) or connection_line.crosses(main_geom) or node_transformed_branch.distance(main_geom) < 50.0:
            continue  # Failed, try again        
        
        if len(original_coords) > 0: #if there is a subtree, rotate it
            # --- Vectorized rotation + translation ---
            # Shift to origin, rotate, shift back, then translate        
            coords_array = np.array([(p.x, p.y) for p in original_coords])

            # Center relative to parent
            parent_coord = np.array([parent_pos.x, parent_pos.y])
            centered_coords = coords_array - parent_coord  # parent_pos should be (x, y) tuple or array

            # Rotation matrix
            rot_matrix = np.array([
                [math.cos(dtheta), -math.sin(dtheta)],
                [math.sin(dtheta),  math.cos(dtheta)]
            ])

            # Apply rotation
            rotated = centered_coords @ rot_matrix.T  

            # Apply translation
            translated = rotated + parent_coord + np.array([dx, dy])

            # Convert back to Shapely Points
            new_coords = [ShapelyPoint(x, y) for x, y in translated]

            # Update graph
            for i, node in enumerate(subtree_nodes):
                graph.nodes[node]['coordinates'] = new_coords[i]

        graph.nodes[vertex_to_randomize]['coordinates'] = new_vertex_pos        
        return True
   
    return False



def generate_network_coordinates(net, root_bus_index, graph: nx.DiGraph, min_distance_m=10, max_attempts=50):    
    
    success = False    
    
    graph.nodes[root_bus_index]['coordinates'] = ShapelyPoint(0.0, 0.0)
    graph.nodes[root_bus_index]['angle'] = 0.0
    graph.nodes[root_bus_index]['radius'] = 0.0
    graph.nodes[root_bus_index]['geometric_neighbors'] = None
    
    current_breadth = [root_bus_index]
    
    radius = 100.0    

    def distribute_children_on_parent_arc(parent_node, arc, graph):                        
        
        children = list(graph.successors(parent_node))        
        arc = normalize_arc(*arc)  # Normalize the arc to ensure it is in the correct range
        if arc_span(arc) == 0: #the arc spans the whole circle:
            if len(children) == 1:
                angle_step = math.pi
            else:    
                angle_step = 2 * math.pi / len(children)
            
            arc = normalize_arc(graph.nodes[parent_node]['angle']-math.pi, graph.nodes[parent_node]['angle']+math.pi)  
        else:
            angle_step = arc_span(arc)/(len(children)+1)            

        # Compute initial angles for this parent's children
        angle_offsets = [arc[0] + (i+1) * angle_step for i in range(len(children))]
        
        radius = graph.nodes[parent_node]['radius'] + 100  # Increment radius for children                
        graph.nodes[parent_node]['arc_span'] = arc
        for child, offset in zip(children, angle_offsets):         
            x = radius * math.cos(offset)
            y = radius * math.sin(offset)            
            graph.nodes[child]['coordinates'] = ShapelyPoint(x, y)
            graph.nodes[child]['angle'] = offset
            graph.nodes[child]['radius'] = radius
            

    while len(current_breadth) > 0:        
        
        next_breadth = [child for node in current_breadth for child in graph.successors(node)]
        
        # For each parent in current_breadth, distribute its children evenly along a 360-degree arc centered on the parent direction
        if next_breadth:            
            # Determine the neighboorhood in the next_breadth            
            for vertex in next_breadth:                
                # Determine geometric neighbors: negative and positive siblings in next_breadth (wrap around if at ends)                    
                if len(next_breadth) == 1:
                    geometric_neighbors = None
                else:
                    idx = next_breadth.index(vertex)
                    negative_idx = (idx - 1) % len(next_breadth)
                    positive_idx = (idx + 1) % len(next_breadth)
                    negative_neighbor = next_breadth[negative_idx]
                    positive_neighbor = next_breadth[positive_idx]
                    geometric_neighbors = (negative_neighbor, positive_neighbor)
                
                graph.nodes[vertex]['geometric_neighbors'] = geometric_neighbors
                  
            #create base arc for all parents in current_breadth            
            for vertex in current_breadth:                                
                children = list(graph.successors(vertex))
                if len(children) > 0:                    
                    arc_span_per_child = (2*math.pi)/len(next_breadth)                                
                    base_angle = graph.nodes[vertex]['angle']                     
                    my_arc = (base_angle - ((arc_span_per_child*len(children))/2), base_angle + ((arc_span_per_child*len(children))/2))
                    graph.nodes[vertex]['arc_span'] = normalize_arc(*my_arc)                    

            #adjust all arcs to prevent overlaps with geometric neighbors            
            for vertex in current_breadth:                 
                children = list(graph.successors(vertex))
                if len(children) > 0:                    
                    if graph.nodes[vertex]['geometric_neighbors'] is not None:
                        if 'arc_span' in graph.nodes[graph.nodes[vertex]['geometric_neighbors'][1]]:
                            overlaped_arc = intersect_arcs(graph.nodes[vertex]['arc_span'],graph.nodes[graph.nodes[vertex]['geometric_neighbors'][1]]['arc_span'])
                            if overlaped_arc != []:
                                my_new_arc, neighbor_new_arc = compute_nonoverlapping_subarcs(vertex, graph.nodes[vertex]['geometric_neighbors'][1], graph) 
                                graph.nodes[vertex]['arc_span'] = my_new_arc
                                graph.nodes[graph.nodes[vertex]['geometric_neighbors'][1]]['arc_span'] = neighbor_new_arc
            
            #distribute children evenly along the calculated arcs
            for vertex in current_breadth: 
                children = list(graph.successors(vertex))
                if len(children) > 0:
                    #trim the arc to be centered on parent
                    angle = graph.nodes[vertex]['angle']
                    if angle < graph.nodes[vertex]['arc_span'][0]:
                        angle += 2 * math.pi
                    semi_arc_length = min(arc_span((graph.nodes[vertex]['arc_span'][0], angle)),arc_span((angle, graph.nodes[vertex]['arc_span'][1])))
                                        
                    graph.nodes[vertex]['arc_span'] = normalize_arc(angle - semi_arc_length, angle + semi_arc_length)                       
                    
                    distribute_children_on_parent_arc(vertex, graph.nodes[vertex]['arc_span'], graph)                               

               
        current_breadth = next_breadth
        radius = radius + 100   
    
    #now randomize the positions of the branches and remove intermediate information
    for node in nx.bfs_tree(graph, root_bus_index):
        for key in ['angle', 'radius', 'geometric_neighbors', 'arc_span']:
            graph.nodes[node].pop(key, None)
        if node == root_bus_index:
            continue  # Skip the root        
        randomize_branch_positions(graph, node, max_trials=max_attempts)
    
    success = True 
    return success, graph 