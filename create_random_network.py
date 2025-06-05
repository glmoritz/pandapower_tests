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
        network_vertices = [
                    {
                        'edges': (parent_bus,working_bus),
                        'length': 15.0
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
                network_vertices.append(
                    {
                        'edges': (working_bus, new_bus),
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
        
        return network_vertices

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
        network_vertices=total_net_lines,
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


def generate_network_coordinates(net, root_bus_index, network_vertices, min_distance_m=10, max_attempts=50):
    connections = {}
    success = False
    for vertix in network_vertices:
        connections.setdefault(vertix['edges'][0], []).append({
                                                                'destination' :vertix['edges'][1],
                                                                'length': vertix['length']
                                                              }               
                                                             )
        connections.setdefault(vertix['edges'][1], []).append(
                                                               {
                                                                'destination' :vertix['edges'][0],
                                                                'length': vertix['length']
                                                               }               
                                                             )


    points = {root_bus_index: ShapelyPoint(0.0, 0.0)}  # Start with the root bus at the given coordinates
    lines = []
    visited = set()

    def bfs(start_node): #Breadth-First Node Placement
        queue = deque([start_node])
        visited = set([start_node])
        point_tree = None
        line_tree = None


        while queue:
            node = queue.popleft()
            origin = points[node]

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
                                                iterations=12) 

                    if new_pos is None:
                        break

                    if fast_is_valid_placement(new_pos, origin, point_tree, line_tree,
                                        min_distance=min(min_distance_m, line_length_meters * 0.90)):
                        points[dest] = new_pos
                        lines.append(LineString([origin, new_pos]))
                        point_geoms = list(points.values())
                        point_tree = STRtree(point_geoms)

                        # Create STRtree for lines
                        line_tree = STRtree(lines)

                        visited.add(dest)
                        queue.append(dest)
                        placed = True
                        break

                if not placed:
                    return False  # failed to place one neighbor

        return True

    def dfs(node): #Depth-First Node Placement
        visited.add(node)
        origin = points[node]
        for neighbor in connections.get(node, []):
            if neighbor['destination'] in visited:
                continue
            for _ in range(max_attempts):
                line_length_meters = neighbor['length']
                
                new_pos = smart_random_destination(
                                                origin_latlon=origin,
                                                length_meters=line_length_meters, 
                                                lines=lines,                                                
                                                placed_points=points,
                                                min_distance=min(min_distance_m, line_length_meters * 0.90),
                                                iterations=12) 

                if fast_is_valid_placement(new_pos, origin, points, lines,min_distance=min(min_distance_m,line_length_meters*0.90)):
                    points[neighbor['destination']] = new_pos
                    lines.append(LineString(origin, new_pos))
                    if dfs(neighbor['destination']):
                        break
                    # backtrack
                    lines.pop()
                    points.pop(neighbor['destination'])
            else:
                return False
        return True

    success = bfs(root_bus_index)    
    return success, points 

def random_destination(origin_latlon, length_meters):
    angle = random.uniform(0, 360)  # bearing in degrees
    origin = GeopyPoint(origin_latlon[0], origin_latlon[1])
    destination = distance(meters=length_meters).destination(origin, angle)
    return (destination.latitude, destination.longitude)

def smart_random_destination(origin_coordinate: ShapelyPoint, length_meters: float, line_index: STRtree, placed_points_index: STRtree, min_distance: float, iterations:int=10):            
    chosen_angle = None

    for i in range(iterations):
        if chosen_angle is not None:
            break
        
        current_step = 360.0 / (2 ** i)
        random_offset = random.uniform(0, current_step)
        angles = [random_offset + j * current_step for j in range(int(360.0 / current_step))]
        
        def navigate_to(origin_coordinate: ShapelyPoint, angle, length_meters):
            # Offset coordinate by length_meters in the direction of the angle (degrees)
            angle_rad = math.radians(angle)
            dx = length_meters * math.cos(angle_rad)
            dy = length_meters * math.sin(angle_rad)            
            return ShapelyPoint(origin_coordinate.x + dx, origin_coordinate.y + dy)

        # Calculate candidate points using shapely cartesian algorithm (treating everything as cartesian)        
        candidate_points = [navigate_to(origin_coordinate, angle, length_meters) for angle in angles]
        
        # Try candidates in random order, pick the first valid one
        candidate_indices = list(range(len(candidate_points)))
        random.shuffle(candidate_indices)
        for idx in candidate_indices:
            dest_coordinate = candidate_points[idx]            
            if fast_is_valid_placement(dest_coordinate, origin_coordinate, placed_points_index, line_index, min_distance):
                chosen_angle = angles[idx]
                break       
    
    if chosen_angle is not None:
        return navigate_to(origin_coordinate, chosen_angle, length_meters)
    else:
        return None  # No valid destination found after all iterations
        #raise RuntimeError("Could not find a valid destination after multiple attempts.")


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
                    
                    is_endpoint_of_existing_line = (new_point.equals(existing_line.coords[0]) or new_point.equals(existing_line.coords[-1]))
                    
                    # The line being considered for connection (from_point to new_point)
                    potential_new_line = LineString([from_point, new_point])

                    # If the existing line is the one we are creating right now (which shouldn't happen with proper indexing),
                    # or if the new point is too close to an existing line that it's not a direct endpoint of.
                    if not is_endpoint_of_existing_line and not potential_new_line.equals(existing_line):
                        # If new_point is too close to an existing line, and it's not a common endpoint
                        # and it's not the line being currently built.
                        return False

    # 3. Check new line (from_point to new_point) intersection/proximity with existing lines
    new_line = LineString([from_point, new_point])
    
    if lines_index is not None:
        # Buffer the new line to check for proximity to other lines
        # Using a buffer of half the min_distance ensures that the centers of lines
        # are at least min_distance apart. If you want the edges to be min_distance apart,
        # you'd use min_distance directly.
        buffered_new_line_for_clearance = new_line.buffer(min_distance, cap_style='flat', join_style='mitre')

        # Query for lines within the bounding box of the buffered new line
        nearby_lines_indices_for_clearance = lines_index.query(buffered_new_line_for_clearance)

        for l_index in nearby_lines_indices_for_clearance:
            existing_line = lines_index.geometries[l_index]
            
            # Avoid checking the line against itself if it somehow gets included
            if new_line.equals(existing_line):
                continue

            # If the buffered new line intersects with an existing line, it means they are too close
            if buffered_new_line_for_clearance.intersects(existing_line):
                intersection = buffered_new_line_for_clearance.intersection(existing_line)
                
                # Allow intersection only if it's exactly at a common endpoint of the *unbuffered* lines
                # This ensures that lines connected to the same bus are not flagged as violations
                if isinstance(intersection, ShapelyPoint):
                    # Check if the intersection point is an endpoint of both the new_line AND the existing_line
                    is_endpoint_of_newline = (intersection.equals(from_point) or intersection.equals(new_point))
                    is_endpoint_of_existingline = (intersection.equals(existing_line.coords[0]) or intersection.equals(existing_line.coords[-1]))
                    
                    if is_endpoint_of_newline and is_endpoint_of_existingline:
                        # If it's a shared endpoint, it's allowed.
                        continue
                
                # If there's an intersection not at a shared endpoint (e.g., crossing or just too close along the line)
                return False

    return True  # No invalid intersections found