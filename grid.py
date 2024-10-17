import pandapower as pp

def create_grid():
    net = pp.create_empty_network()

    # Create buses
    bus1 = pp.create_bus(net, vn_kv=0.4)
    bus2 = pp.create_bus(net, vn_kv=0.4)
    
    # Create external grid connection
    pp.create_ext_grid(net, bus1)
    
    # Create lines
    pp.create_line(net, bus1, bus2, length_km=0.1, std_type='NAYY 4x50 SE')
    
    # Create load
    pp.create_load(net, bus2, p_mw=0.1, q_mvar=0.05)
    
    # Create solar PV system
    pp.create_sgen(net, bus2, p_mw=0.05, q_mvar=0)
    
    return net
