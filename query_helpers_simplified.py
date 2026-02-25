"""
Query Helpers for the Normalized Schema
----------------------------------------
These helpers work with pre-aggregated data from:
- building_power.simulation_variable (variable definitions)
- building_power.simulation_timeseries (values)

Since data is already homogenized into fixed buckets, queries are simple JOINs.
Storage is ~10x smaller than denormalized approach.
"""

import pandas as pd
from sqlalchemy import text, Engine, Connection
from typing import Optional, List, Union


def get_last_simulation_output_id(test_name: str, conn: Union[Engine, Connection]):
    """
    Get the last simulation_output_id for a given test_name.
    
    Args:
        test_name: The test name to search for
        conn: SQLAlchemy connection or engine
        
    Returns:
        Tuple of (simulation_output_id, parameters dict) or (None, None)
    """
    query = text("""
        SELECT simulation_output_id, parameters
        FROM building_power.simulation_outputs 
        WHERE parameters ->> 'name' = :test_name
        ORDER BY started_at DESC 
        LIMIT 1
    """)
    
    result = conn.execute(query, {"test_name": test_name}).fetchone()
    
    if result:
        return result[0], result[1]
    return None, None


def get_timeseries_data(
    conn: Union[Engine, Connection],
    simulation_id: int,
    element_type: Optional[str] = None,
    element_idxs: Optional[Union[int, List[int]]] = None,
    output_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Get timeseries data from the normalized schema.
    
    Since data is pre-aggregated, this is a simple JOIN with optional filters.
    
    Args:
        conn: SQLAlchemy connection or engine
        simulation_id: The simulation to query
        element_type: Optional filter for element type (e.g., 'householdproducer')
        element_idxs: Optional filter for element indices
        output_name: Optional filter for variable name (supports SQL LIKE pattern)
        
    Returns:
        DataFrame with columns: bucket, total_quantity (sum of matching values)
    """
    if simulation_id is None:
        raise ValueError("simulation_id cannot be None")
    
    # Build WHERE clauses for variable table
    where_clauses = ["v.simulation_id = :simulation_id"]
    params = {"simulation_id": simulation_id}
    
    if element_type is not None:
        where_clauses.append("v.element_type = :element_type")
        params["element_type"] = element_type
    
    if element_idxs is not None:
        if isinstance(element_idxs, int):
            element_idxs = [str(element_idxs)]
        else:
            element_idxs = [str(idx) for idx in element_idxs]
        where_clauses.append("v.element_index = ANY(:element_idxs)")
        params["element_idxs"] = element_idxs
    
    if output_name is not None:
        where_clauses.append("v.variable_name LIKE :output_name")
        params["output_name"] = output_name
    
    where_sql = " AND ".join(where_clauses)
    
    # JOIN timeseries with variable definitions
    query = text(f"""
        SELECT 
            t.bucket,
            SUM(t.value) AS total_quantity
        FROM building_power.simulation_timeseries t
        JOIN building_power.simulation_variable v ON t.variable_id = v.variable_id
        WHERE {where_sql}
        GROUP BY t.bucket
        ORDER BY t.bucket
    """)
    
    df = pd.read_sql_query(query, conn, params=params)
    return df


def get_timeseries_by_element(
    conn: Union[Engine, Connection],
    simulation_id: int,
    element_type: Optional[str] = None,
    element_idxs: Optional[Union[int, List[int]]] = None,
    output_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Get timeseries data grouped by element (not aggregated across elements).
    
    Useful when you want to see individual bus/household data.
    
    Args:
        conn: SQLAlchemy connection or engine
        simulation_id: The simulation to query
        element_type: Optional filter for element type
        element_idxs: Optional filter for element indices
        output_name: Optional filter for variable name (supports SQL LIKE pattern)
        
    Returns:
        DataFrame with columns: bucket, element_type, element_index, variable_name, value
    """
    if simulation_id is None:
        raise ValueError("simulation_id cannot be None")
    
    where_clauses = ["v.simulation_id = :simulation_id"]
    params = {"simulation_id": simulation_id}
    
    if element_type is not None:
        where_clauses.append("v.element_type = :element_type")
        params["element_type"] = element_type
    
    if element_idxs is not None:
        if isinstance(element_idxs, int):
            element_idxs = [str(element_idxs)]
        else:
            element_idxs = [str(idx) for idx in element_idxs]
        where_clauses.append("v.element_index = ANY(:element_idxs)")
        params["element_idxs"] = element_idxs
    
    if output_name is not None:
        where_clauses.append("v.variable_name LIKE :output_name")
        params["output_name"] = output_name
    
    where_sql = " AND ".join(where_clauses)
    
    query = text(f"""
        SELECT 
            t.bucket,
            v.element_type,
            v.element_index,
            v.variable_name,
            t.value,
            v.unit
        FROM building_power.simulation_timeseries t
        JOIN building_power.simulation_variable v ON t.variable_id = v.variable_id
        WHERE {where_sql}
        ORDER BY t.bucket, v.element_type, v.element_index, v.variable_name
    """)
    
    df = pd.read_sql_query(query, conn, params=params)
    return df


def get_simulation_statistics(
    conn: Union[Engine, Connection],
    simulation_id: int,
    element_type: Optional[str] = None,
    output_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Get statistics (mean, min, max, p05, p95) for variables in a simulation.
    
    Args:
        conn: SQLAlchemy connection or engine
        simulation_id: The simulation to query
        element_type: Optional filter for element type
        output_name: Optional filter for variable name
        
    Returns:
        DataFrame with statistics per variable
    """
    if simulation_id is None:
        raise ValueError("simulation_id cannot be None")
    
    where_clauses = ["v.simulation_id = :simulation_id"]
    params = {"simulation_id": simulation_id}
    
    if element_type is not None:
        where_clauses.append("v.element_type = :element_type")
        params["element_type"] = element_type
    
    if output_name is not None:
        where_clauses.append("v.variable_name LIKE :output_name")
        params["output_name"] = output_name
    
    where_sql = " AND ".join(where_clauses)
    
    query = text(f"""
        SELECT 
            v.element_type,
            v.element_index,
            v.variable_name,
            v.unit,
            AVG(t.value) as mean_value,
            MIN(t.value) as min_value,
            MAX(t.value) as max_value,
            PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY t.value) as p05_value,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY t.value) as p95_value,
            COUNT(*) as sample_count
        FROM building_power.simulation_timeseries t
        JOIN building_power.simulation_variable v ON t.variable_id = v.variable_id
        WHERE {where_sql}
        GROUP BY v.element_type, v.element_index, v.variable_name, v.unit
        ORDER BY v.element_type, v.element_index, v.variable_name
    """)
    
    df = pd.read_sql_query(query, conn, params=params)
    return df


def get_available_variables(
    conn: Union[Engine, Connection],
    simulation_id: int
) -> pd.DataFrame:
    """
    List all unique variables available in a simulation.
    
    Args:
        conn: SQLAlchemy connection or engine
        simulation_id: The simulation to query
        
    Returns:
        DataFrame with columns: element_type, element_index, variable_name, unit
    """
    query = text("""
        SELECT
            element_type,
            element_index,
            variable_name,
            unit
        FROM building_power.simulation_variable
        WHERE simulation_id = :simulation_id
        ORDER BY element_type, element_index, variable_name
    """)
    
    df = pd.read_sql_query(query, conn, params={"simulation_id": simulation_id})
    return df


def compare_simulations(
    conn: Union[Engine, Connection],
    simulation_ids: List[int],
    variable_name: str,
    element_type: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare a specific variable across multiple simulations.
    
    Args:
        conn: SQLAlchemy connection or engine
        simulation_ids: List of simulation IDs to compare
        variable_name: The variable to compare (exact match)
        element_type: Optional filter for element type
        
    Returns:
        DataFrame with bucket as index and simulation_id columns
    """
    where_clauses = [
        "v.simulation_id = ANY(:simulation_ids)",
        "v.variable_name = :variable_name"
    ]
    params = {
        "simulation_ids": simulation_ids,
        "variable_name": variable_name
    }
    
    if element_type is not None:
        where_clauses.append("v.element_type = :element_type")
        params["element_type"] = element_type
    
    where_sql = " AND ".join(where_clauses)
    
    query = text(f"""
        SELECT 
            t.bucket,
            v.simulation_id,
            SUM(t.value) as total_value
        FROM building_power.simulation_timeseries t
        JOIN building_power.simulation_variable v ON t.variable_id = v.variable_id
        WHERE {where_sql}
        GROUP BY t.bucket, v.simulation_id
        ORDER BY t.bucket, v.simulation_id
    """)
    
    df = pd.read_sql_query(query, conn, params=params)
    
    # Pivot to have simulations as columns
    if not df.empty:
        df = df.pivot(index='bucket', columns='simulation_id', values='total_value')
    
    return df


# =============================================================================
# PLOTTING HELPERS (preserved from original)
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np


def plot_cdf_with_shadow(data_mean, data_hi, data_lo, label):
    """Plot CDF with confidence interval shadow."""
    sorted_data_mean = np.sort(data_mean)
    cdf = np.arange(1, len(sorted_data_mean) + 1) / len(sorted_data_mean) * 100
    
    sorted_data_hi = np.sort(data_hi)
    sorted_data_lo = np.sort(data_lo)
    
    plt.plot(sorted_data_mean, cdf, label=label)
    plt.fill_betweenx(cdf, sorted_data_lo, sorted_data_hi, alpha=0.2)
