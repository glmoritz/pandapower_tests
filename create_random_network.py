import pandapower as pp
import random
from enum import Enum
import random
import math
from shapely.geometry import Point as ShapelyPoint, LineString
from shapely.strtree import STRtree
from geopy.distance import distance
from geopy.point import Point as GeopyPoint
from collections import deque


class NetworkType(Enum):
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    RESIDENTIAL = "residential"

def generate_pandapower_net(CommercialRange, IndustrialRange, ResidencialRange,                            
                            ForkLengthRange, LineBusesRange, LineForksRange,
                            mv_bus_coordinates=(0.0, 0.0),
                            ):    
    net_cigre_lv = pp.create_empty_network()

    # Linedata
    # UG1
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 0.162,
                 'x_ohm_per_km': 0.0832, 'max_i_ka': 1.0,
                 'type': 'cs'}
    pp.create_std_type(net_cigre_lv, line_data, name='UG1', element='line')

    # UG2
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 0.2647,
                 'x_ohm_per_km': 0.0823, 'max_i_ka': 1.0,
                 'type': 'cs'}
    pp.create_std_type(net_cigre_lv, line_data, name='UG2', element='line')

    # UG3
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 0.822,
                 'x_ohm_per_km': 0.0847, 'max_i_ka': 1.0,
                 'type': 'cs'}
    pp.create_std_type(net_cigre_lv, line_data, name='UG3', element='line')

    # OH1
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 0.4917,
                 'x_ohm_per_km': 0.2847, 'max_i_ka': 1.0,
                 'type': 'ol'}
    pp.create_std_type(net_cigre_lv, line_data, name='OH1', element='line')

    # OH2
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 1.3207,
                 'x_ohm_per_km': 0.321, 'max_i_ka': 1.0,
                 'type': 'ol'}
    pp.create_std_type(net_cigre_lv, line_data, name='OH2', element='line')

    # OH3
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 2.0167,
                 'x_ohm_per_km': 0.3343, 'max_i_ka': 1.0,
                 'type': 'ol'}
    pp.create_std_type(net_cigre_lv, line_data, name='OH3', element='line')

    # Add the main 20 kV bus (medium voltage)
    mv_bus = pp.create_bus(net_cigre_lv, name='Bus 0', vn_kv=20.0, type='b', zone='CIGRE_LV')
    

    def create_cigre_lv_transformer(hv_bus, lv_bus, name, transformer_type):
        if transformer_type == NetworkType.RESIDENTIAL:
            pp.create_transformer_from_parameters(net_cigre_lv, hv_bus, lv_bus, sn_mva=0.5, vn_hv_kv=20.0,
                                       vn_lv_kv=0.4, vkr_percent=1.0, vk_percent=4.123106,
                                       pfe_kw=0.0, i0_percent=0.0, shift_degree=30,
                                       tap_pos=0.0, name=name)
        elif transformer_type == NetworkType.INDUSTRIAL:
            pp.create_transformer_from_parameters(net_cigre_lv, hv_bus, lv_bus, sn_mva=0.15, vn_hv_kv=20.0,
                                            vn_lv_kv=0.4, vkr_percent=1.003125, vk_percent=4.126896,
                                            pfe_kw=0.0, i0_percent=0.0, shift_degree=30,
                                            tap_pos=0.0, name=name)
        elif transformer_type == NetworkType.COMMERCIAL:
            pp.create_transformer_from_parameters(net_cigre_lv, hv_bus, lv_bus, sn_mva=0.3, vn_hv_kv=20.0,
                                            vn_lv_kv=0.4, vkr_percent=0.993750, vk_percent=4.115529,
                                            pfe_kw=0.0, i0_percent=0.0, shift_degree=30,
                                            tap_pos=0.0, name=name)

    def add_transformer_branch(transformer_type, branch_index, parent_bus, current_net):
        # Create LV bus (0.4 kV)        
        working_bus_name = f"{str(transformer_type)}_Branch{branch_index}_trafo_lv_bus"
        working_bus = pp.create_bus(current_net, vn_kv=0.4, name=working_bus_name)        

        # Add transformer from MV to LV
        create_cigre_lv_transformer(parent_bus, working_bus, f"{str(transformer_type)}_{branch_index}_Trafo", transformer_type)
        network_edges = [
                    {
                        'vertices': (parent_bus,working_bus),
                        'length': 100.0
                    }
                    ]
                
        #randomize how many forks this line will be
        Forks = random.randint(*LineForksRange)

        #randomize the total length of the forks
        Fork_Lengths = [random.randint(*ForkLengthRange) for _ in range(Forks)]

        #randomize how many buses will be in this branch
        NumBuses = random.randint(*LineBusesRange)
        
        #distribute NumBuses across the forks
        fork_bus_counts = [1] * Forks  # start with 1 bus per fork
        remaining = NumBuses - Forks
        for _ in range(remaining):
            fork_bus_counts[random.randint(0, Forks - 1)] += 1

        # For each fork, create a vector of random lengths whose sum is Fork_Lengths[i]
        fork_hop_lengths = []
        for i in range(Forks):
            n_hops = fork_bus_counts[i] 
            total_length = Fork_Lengths[i]
            # Generate n_hops random positive numbers
            if n_hops == 1:
                fork_hop_lengths.append([total_length])
            else:
                cuts = sorted([random.uniform(0, total_length) for _ in range(n_hops - 1)])
                hop_lengths = [cuts[0]] + [cuts[j] - cuts[j-1] for j in range(1, n_hops - 1)] + [total_length - cuts[-1]]
                fork_hop_lengths.append(hop_lengths)

        added_busses = []        
        
        # Create branches/forks       
        
        for f in range(Forks):
                        
            for b in range(fork_bus_counts[f]):
                #create the bus
                new_bus_name = f"{str(transformer_type)}_Branch{branch_index}_fork{f}bus{b}"
                new_bus = pp.create_bus(current_net, name=new_bus_name, vn_kv=0.4, type='m', zone='CIGRE_LV')

                #create a line connecting the parent bus to the child bus
                pp.create_line(current_net, working_bus, new_bus, length_km=fork_hop_lengths[f][b], std_type='OH1', name=f'Line {net_cigre_lv.bus.at[working_bus, 'name']}-{net_cigre_lv.bus.at[new_bus, 'name']}')
                network_edges.append(
                    {
                        'vertices': (working_bus, new_bus),
                        'length': fork_hop_lengths[f][b]
                    }
                )
                added_busses.append(new_bus)    

                working_bus = new_bus
                working_bus_name = new_bus_name

            #pick a random child bus to connect the next fork
            if f < Forks - 1:
                working_bus = random.choice(added_busses)
            else:
                working_bus = parent_bus    
            
            # Retrieve the parent_bus_name from net_cigre_lv
            working_bus_name = net_cigre_lv.bus.at[parent_bus, 'name']
        
        return network_edges

    # Create transformers by type
    Ct = random.randint(*CommercialRange)
    It = random.randint(*IndustrialRange)
    Rt = random.randint(*ResidencialRange)
    
    total_net_lines = []

    for i in range(Ct):
        new_lines = add_transformer_branch(
            transformer_type=NetworkType.COMMERCIAL,
            branch_index=i,
            parent_bus=mv_bus,
            current_net= net_cigre_lv            
            )
        total_net_lines.extend(new_lines)
    for i in range(It):
        new_lines = add_transformer_branch(
            transformer_type=NetworkType.INDUSTRIAL,
            branch_index=i,
            parent_bus=mv_bus,
            current_net= net_cigre_lv            
            )
        total_net_lines.extend(new_lines)
    for i in range(Rt):
        new_lines = add_transformer_branch(
            transformer_type=NetworkType.RESIDENTIAL,
            branch_index=i,
            parent_bus=mv_bus,
            current_net= net_cigre_lv            
            )
        total_net_lines.extend(new_lines)

    success, points = generate_network_coordinates(
        net=net_cigre_lv,
        root_bus_index=mv_bus,        
        network_edges=total_net_lines,
        min_distance_m=5,  # Minimum distance between points
        max_attempts=50  # Maximum attempts to place a point
    )

    # Bus geo data
    origin = GeopyPoint(mv_bus_coordinates[0], mv_bus_coordinates[1])
    
    def cartesian_to_latlon(x, y, origin_point):
        # Compute bearing and distance
        angle = math.degrees(math.atan2(y, x))  # bearing in degrees
        dist = math.hypot(x, y)  # Euclidean distance in meters
        destination = distance(meters=dist).destination(origin_point, angle)
        return destination.latitude, destination.longitude

    # Convert each bus point to lat/lon GeoJSON string
    bus_geodata = [
        (
            f'{{"type":"Point", "coordinates":[{lon}, {lat}]}}'
            if idx in points else None
        )
        for idx in net_cigre_lv.bus.index
        for lat, lon in [
            cartesian_to_latlon(points[idx].x, points[idx].y, origin)
            if idx in points else (None, None)
        ]
    ]

    net_cigre_lv.bus["geo"] = bus_geodata
    return net_cigre_lv, total_net_lines


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

def compute_nonoverlapping_subarcs(parent1, parent2, points_info):
#def compute_nonoverlapping_subarcs(theta3, theta1, theta2, theta4):
    if parent1 == 26:
        breakpoint()
    theta1 = points_info[parent1]['angle']
    theta2 = points_info[parent2]['angle']
    shared_arc = union_arcs(points_info[parent1]['arc_span'],points_info[parent2]['arc_span'])
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


def generate_network_coordinates(net, root_bus_index, network_edges, min_distance_m=10, max_attempts=50):    
    connections = {}
    parent = {}
    children = {}
    success = False
    for edge in network_edges:
        connections.setdefault(edge['vertices'][0], []).append({
                                                                'destination' :edge['vertices'][1],
                                                                'length': edge['length']
                                                              })  
        
        children.setdefault(edge['vertices'][0], []).append(edge['vertices'][1]) 
        
        parent[edge['vertices'][1]] =  edge['vertices'][0]   
                                                                     
        # connections.setdefault(edge['vertices'][1], []).append(
        #                                                        {
        #                                                         'destination' :edge['vertices'][0],
        #                                                         'length': edge['length']
        #                                                        }               
        #                                                      )

    points = {root_bus_index: ShapelyPoint(0.0, 0.0)}  # Start with the root bus at (0,0)
    points_info = {root_bus_index: {
                    'angle': 0.0,
                    'radius': 100.0,
                    'geometric_neighbors': None  # No siblings for the root
                    }
                    } 

    lines = []
    visited = set()

    current_breadth = [root_bus_index]
    
    radius = 100.0    

    def distribute_children_on_parent_arc(parent_node, arc, connections, points_info):                
        if parent_node == 21:
            breakpoint()
        children = [node['destination'] for node in connections.get(parent_node, [])]
        arc = normalize_arc(*arc)  # Normalize the arc to ensure it is in the correct range
        if arc_span(arc) == 0: #the arc spans the whole circle:
            if len(children) == 1:
                angle_step = math.pi
            else:    
                angle_step = 2 * math.pi / len(children)  
            arc = normalize_arc(points_info[parent_node]['angle']-math.pi, points_info[parent_node]['angle']+math.pi)  
        else:
            angle_step = arc_span(arc)/(len(children)+1) 
            # if angle_step > math.pi/4:
            #     # Instead of distributing across the whole arc, center children around the parent's angle
            #     parent_angle = points_info[parent_node]['angle']
            #     # Limit the arc to at most pi/2 (90 degrees) centered on parent_angle
            #     arc_width = min((math.pi/4)*len(children) , arc_span(arc))
            #     # Clamp start_angle to be within the arc's start and stop
            #     unclamped_start_angle = parent_angle - arc_width / 2
            #     start_angle = max(arc[0], min(unclamped_start_angle, arc[1] - arc_width))                
            #     # Clamp the arc to stay within the original arc bounds
            #     arc_start = max(arc[0], start_angle)
            #     arc_end = min(arc[1], start_angle + arc_width)
            #     arc = normalize_arc(arc_start, arc_end)
            #     angle_step = arc_span(arc) / (len(children) + 1)

        # Compute initial angles for this parent's children
        angle_offsets = [arc[0] + (i+1) * angle_step for i in range(len(children))]

        radius = points_info[parent_node]['radius'] + 100  # Increment radius for children
        
        points_info[parent_node]['arc_span'] = arc  # Store the arc span for the parent node
        for child, offset in zip(children, angle_offsets): 
            if child == 22:
                breakpoint()
            x = radius * math.cos(offset)
            y = radius * math.sin(offset)
            points[child] = ShapelyPoint(x, y)
            points_info[child]['angle'] = offset

    while len(current_breadth) > 0:

        next_breadth = [n['destination'] for node in current_breadth for n in connections.get(node, [])]
        
        # For each parent in current_breadth, distribute its children evenly along a 360-degree arc centered on the parent direction
        if next_breadth:
            # Build a mapping from parent to its children
            # Determine the neighboorhood in the next_breadth
            parent_to_children = {}
            for child in next_breadth:
                parent_node = parent[child]
                parent_to_children.setdefault(parent_node, []).append(child)

                # Determine geometric neighbors: negative and positive siblings in next_breadth (wrap around if at ends)                    
                if len(next_breadth) == 1:
                    geometric_neighbors = None
                else:
                    idx = next_breadth.index(child)
                    negative_idx = (idx - 1) % len(next_breadth)
                    positive_idx = (idx + 1) % len(next_breadth)
                    negative_neighbor = next_breadth[negative_idx]
                    positive_neighbor = next_breadth[positive_idx]
                    geometric_neighbors = (negative_neighbor, positive_neighbor)
                points_info[child] = {
                    'angle': None,
                    'radius': radius,
                    'geometric_neighbors': geometric_neighbors,
                    'arc_span': None  # Will be calculated later
                }
                  
            #create base arc for all parents in current_breadth
            for parent_node, children in parent_to_children.items():                                
                arc_span_per_child = (2*math.pi)/len(next_breadth)                                
                base_angle = points_info[parent_node]['angle']            
                my_arc = (base_angle - ((arc_span_per_child*len(children))/2), base_angle + ((arc_span_per_child*len(children))/2))
                points_info[parent_node]['arc_span'] = normalize_arc(*my_arc)  

            #adjust all arcs to prevent overlaps with geometric neighbors
            for parent_node, children in parent_to_children.items(): 
                if parent_node == 21:
                    breakpoint()               
                if points_info[parent_node]['geometric_neighbors'] is not None:
                    if points_info[points_info[parent_node]['geometric_neighbors'][1]]['arc_span'] is not None:
                        overlaped_arc = intersect_arcs(points_info[parent_node]['arc_span'],points_info[points_info[parent_node]['geometric_neighbors'][1]]['arc_span'])
                        if overlaped_arc != []:
                            my_new_arc, neighbor_new_arc = compute_nonoverlapping_subarcs(parent_node, points_info[parent_node]['geometric_neighbors'][1], points_info)                           


                            # intersection = overlaped_arc[0]
                            # my_new_arc = subtract_arc(points_info[parent_node]['arc_span'],intersection)
                            # neighbor_new_arc = subtract_arc(points_info[points_info[parent_node]['geometric_neighbors'][1]]['arc_span'],intersection)
                            
                            # if my_new_arc != []:
                            #     my_new_arc = add_arc(my_new_arc, (intersection[0], intersection[0]+arc_span(intersection)/2))                        
                            # else:
                            #     raise ValueError(f"Parent node {parent_node} has no arc span to adjust.")
                                                
                            # if neighbor_new_arc != []:
                            #     neighbor_new_arc = add_arc(neighbor_new_arc, (intersection[1]-arc_span(intersection)/2, intersection[1]) )
                            # else:
                            #     raise ValueError(f"Neighbor node {points_info[parent_node]['geometric_neighbors'][1]} has no arc span to adjust.")                        
                            
                            # points_info[parent_node]['arc_span'] = my_new_arc[0]
                            # points_info[points_info[parent_node]['geometric_neighbors'][1]]['arc_span'] = neighbor_new_arc[0]
                            points_info[parent_node]['arc_span'] = my_new_arc
                            points_info[points_info[parent_node]['geometric_neighbors'][1]]['arc_span'] = neighbor_new_arc
            
            #distribute children evenly along the calculated arcs
            for parent_node, children in parent_to_children.items():                
                if parent_node == 58:
                    breakpoint()
                #trim the arc to be centered on parent
                angle = points_info[parent_node]['angle']
                if angle < points_info[parent_node]['arc_span'][0]:
                    angle += 2 * math.pi
                left_arc_length = arc_span((points_info[parent_node]['arc_span'][0], angle))
                right_arc_length = arc_span((angle, points_info[parent_node]['arc_span'][1]))
                if left_arc_length > right_arc_length:
                    # If left arc is longer, adjust the arc to be centered on the parent
                    points_info[parent_node]['arc_span'] = normalize_arc(angle - left_arc_length, angle + right_arc_length)                       
                elif (right_arc_length > left_arc_length):
                    points_info[parent_node]['arc_span'] = normalize_arc(angle - right_arc_length, angle + right_arc_length)                                       
                distribute_children_on_parent_arc(parent_node, points_info[parent_node]['arc_span'], connections, points_info)                               

               
        current_breadth = next_breadth
        radius = radius + 100

    def bfs(start_node): #Breadth-First Node Placement
        # The queue now stores (current_bus_index, parent_bus_index)
        queue = deque([(start_node, None)]) # Root node has no parent
        visited = set([start_node])
        point_tree = None
        line_tree = None


        while queue:
            node, parent_node_index = queue.popleft() # Pop current node and its parent
            origin = points[node]
            
            # Determine grandpa_coordinate. If it's the root node, parent_node_index will be None.
            grandpa_coordinate = points.get(parent_node_index)

            # Rebuild STRtrees after each successful placement to keep them updated
            if points:
                point_geoms = list(points.values())
                point_tree = STRtree(point_geoms)
            else:
                point_tree = None

            if lines:
                line_tree = STRtree(lines)
            else:
                line_tree = None

            for neighbor in connections.get(node, []):
                dest = neighbor['destination']
                if dest in visited:
                    continue

                placed = False
                for _ in range(max_attempts):
                    line_length_meters = neighbor['length']
                   
                    new_pos = smart_random_destination(
                                                origin_coordinate=origin,
                                                length_meters=line_length_meters, 
                                                line_index=line_tree,                                                
                                                placed_points_index=point_tree,
                                                min_distance=min(min_distance_m, line_length_meters * 0.90),
                                                grandpa_coordinate=grandpa_coordinate, # Pass grandpa_coordinate
                                                iterations=12) 

                    if new_pos is None:
                        break

                    if fast_is_valid_placement(new_pos, origin, point_tree, line_tree,
                                        min_distance=min(min_distance_m, line_length_meters * 0.90)):
                        points[dest] = new_pos
                        lines.append(LineString([origin, new_pos]))
                        
                        # Rebuild STRtrees for the next iteration of this node's neighbors
                        # (This is important to ensure the trees are always up-to-date)
                        point_geoms = list(points.values())
                        point_tree = STRtree(point_geoms)
                        line_tree = STRtree(lines)

                        visited.add(dest)
                        queue.append((dest, node)) # Store current node as parent for the destination
                        placed = True
                        break

                if not placed:
                    return False  # failed to place one neighbor

        return True
    
    success = True #bfs(root_bus_index)    
    return success, points 

def random_destination(origin_latlon, length_meters):
    angle = random.uniform(0, 360)  # bearing in degrees
    origin = GeopyPoint(origin_latlon[0], origin_latlon[1])
    destination = distance(meters=length_meters).destination(origin, angle)
    return (destination.latitude, destination.longitude)

def smart_random_destination(origin_coordinate: ShapelyPoint, length_meters: float, line_index: STRtree, placed_points_index: STRtree, min_distance: float, grandpa_coordinate: ShapelyPoint = None, iterations:int=10):            
    chosen_angle = None
    
    # Determine the central heading based on the grandpa_coordinate and origin_coordinate
    central_heading_degrees = None
    if grandpa_coordinate and not origin_coordinate.equals(grandpa_coordinate):
        dy = origin_coordinate.y - grandpa_coordinate.y
        dx = origin_coordinate.x - grandpa_coordinate.x
        central_heading_degrees = math.degrees(math.atan2(dy, dx))
    
    # Define the allowed angle range (120 degrees) or 360 degrees if no heading
    if central_heading_degrees is None:
        # If no heading (e.g., for the first node), allow 360 degrees
        angle_range_min = 0.0
        angle_range_max = 360.0
    else:
        # Define a 120-degree cone relative to the central heading
        angle_range_min = central_heading_degrees - 60.0
        angle_range_max = central_heading_degrees + 60.0

    # Helper function to navigate from an origin point given an angle and length
    def navigate_to(origin_coordinate: ShapelyPoint, angle, length_meters):
        angle_rad = math.radians(angle)
        dx = length_meters * math.cos(angle_rad)
        dy = length_meters * math.sin(angle_rad)            
        return ShapelyPoint(origin_coordinate.x + dx, origin_coordinate.y + dy)

    for i in range(iterations):
        if chosen_angle is not None:
            break
        
        num_candidates = 2 ** i
        if num_candidates == 0: # Ensure at least one candidate for i=0
            num_candidates = 1
        angle_step = (angle_range_max - angle_range_min) / num_candidates
        angle_offset = random.uniform(-angle_step / 2, angle_step / 2)  # Randomize within the step

        angles_to_try = []
        for j in range(num_candidates):
            # Linearly distribute angles within the calculated range
            angle = angle_offset + angle_range_min + j*angle_step
            angles_to_try.append(angle) # Normalize to 0-360 degrees

        random.shuffle(angles_to_try) # Randomize the order to try candidates

        for dest_angle in angles_to_try:
            dest_coordinate = navigate_to(origin_coordinate, dest_angle, length_meters)            
            if fast_is_valid_placement(dest_coordinate, origin_coordinate, placed_points_index, line_index, min_distance):
                chosen_angle = dest_angle
                break       
    
    if chosen_angle is not None:
        print(f"Angle chosen: {chosen_angle} degrees after {i+1} iterations")
        return navigate_to(origin_coordinate, chosen_angle, length_meters)
    else:
        return None  # No valid destination found after all iterations

def is_valid_placement(new_point, from_point, placed_points, lines, min_distance):
    for pt in placed_points.values():
        if new_point.distance(pt) < min_distance:
            return False
    new_line = LineString([from_point[::-1], new_point[::-1]])  # lon, lat for shapely
    for l in lines:
            if l.intersects(new_line):
                intersection = l.intersection(new_line)
                # If the intersection is a Point and it's at the endpoints of both lines
                if isinstance(intersection, ShapelyPoint):
                    if intersection.equals(ShapelyPoint(l.coords[0])) or intersection.equals(ShapelyPoint(l.coords[-1])):
                        if intersection.equals(ShapelyPoint(new_line.coords[0])) or intersection.equals(ShapelyPoint(new_line.coords[-1])):
                            # intersection is just at the origin/endpoints of both lines → allow it
                            continue
                return False
    return True

def fast_is_valid_placement(new_point: ShapelyPoint, from_point: ShapelyPoint, placed_points_index: STRtree, lines_index: STRtree , min_distance: float) -> bool:       
    """
    Checks if a new point and the line connecting it to a 'from_point' can be placed
    without violating minimum distance constraints to existing points and lines.

    Parameters:
    - new_point (ShapelyPoint): The potential new point to place.
    - from_point (ShapelyPoint): The point from which the new_point originates (forms a line).
    - placed_points_index (STRtree): Spatial index of already placed ShapelyPoint objects.
    - lines_index (STRtree): Spatial index of already placed LineString objects.
    - min_distance (float): The minimum allowable distance between geometries.

    Returns:
    - bool: True if the placement is valid, False otherwise.
    """

    # 1. Check point-to-point distance
    if placed_points_index is not None:
        # Query for points within min_distance of new_point
        nearby_points = placed_points_index.query(new_point.buffer(min_distance))
        for pt_idx in nearby_points:
            pt = placed_points_index.geometries[pt_idx]
            # Ensure the new point isn't too close to existing points, excluding the 'from_point'
            # if from_point is one of the existing points that created the STRtree.
            # We also ensure it's not the exact same point.
            if new_point.distance(pt) < min_distance and not new_point.equals(pt):
                return False

    # 2. Check new point to existing lines distance
    if lines_index is not None:
        # Create a buffer around the new_point to check for proximity to lines
        buffered_new_point = new_point.buffer(min_distance)
        
        # Query for lines within the bounding box of the buffered new point
        nearby_lines_indices = lines_index.query(buffered_new_point)
        
        for l_index in nearby_lines_indices:
            existing_line = lines_index.geometries[l_index]
            
            # Check if the buffered new point intersects with any existing line
            if buffered_new_point.intersects(existing_line):
                # We need to make sure this intersection isn't just because the 'from_point'
                # is connected to this 'existing_line'.
                # If the new point is being placed *on* the `from_point`, that's invalid if `from_point` is part of an existing line
                # (unless it's an endpoint).
                
                # Check if the existing line contains the 'from_point' AND the 'new_point' is distinct from 'from_point'
                # and not coincident with any of the existing line's endpoints.
                if existing_line.distance(new_point) < min_distance:
                    # If the existing line's buffer does not contain the from_point, then the
                    # new point is too close to an existing line that it's not directly connected to.
                    # Or, if it is connected through from_point, we need to ensure new_point
                    # is not too close to the line itself.
                    # We also want to exclude the case where the new point is just the 'from_point'
                    # and that 'from_point' is an endpoint of the existing line.
                    
                    # More robust check: if the new point is too close to the existing line,
                    # AND it's not the line that connects directly to the 'from_point',
                    # AND the new point is not an endpoint of that existing line.
                    
                    is_endpoint_of_existing_line = (new_point.equals(ShapelyPoint(existing_line.coords[0])) or new_point.equals(ShapelyPoint(existing_line.coords[-1])))
                    
                    # The line being considered for connection (from_point to new_point)
                    potential_new_line = LineString([from_point, new_point])

                    # If the existing line is the one we are creating right now (which shouldn't happen with proper indexing),
                    # or if the new point is too close to an existing line that it's not a direct endpoint of.
                    if not is_endpoint_of_existing_line and not potential_new_line.equals(existing_line):
                        # If new_point is too close to an existing line, and it's not a common endpoint
                        # and it's not the line being currently built.
                        return False

    # 3. Check new line (from_point to new_point) intersection with existing lines (not proximity)
    new_line = LineString([from_point, new_point])

    if lines_index is not None:
        nearby_lines_indices = lines_index.query(new_line)
        for l_index in nearby_lines_indices:
            existing_line = lines_index.geometries[l_index]
            if new_line.equals(existing_line):
                continue
            if new_line.intersects(existing_line):
                intersection = new_line.intersection(existing_line)
                # Allow intersection only if it's exactly at a common endpoint of the *unbuffered* lines
                if isinstance(intersection, ShapelyPoint):
                    is_endpoint_of_newline = (intersection.equals(from_point) or intersection.equals(new_point))
                    is_endpoint_of_existingline = (
                        intersection.equals(ShapelyPoint(existing_line.coords[0])) or
                        intersection.equals(ShapelyPoint(existing_line.coords[-1]))
                    )
                    if is_endpoint_of_newline and is_endpoint_of_existingline:
                        continue
                return False

    return True  # No invalid intersections found