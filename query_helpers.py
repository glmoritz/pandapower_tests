
import pandapower.plotting as plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd
from sqlalchemy import text, Engine, Connection


def get_last_simulation_output_id(test_name, conn):
    """Get the last simulation_output_id for a given test_name"""
   
    # Query to find the most recent simulation with matching test_name
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
    else:
        return None, None

# Função para plotar CDF com sombra
def plot_cdf_with_shadow(data_mean, data_hi, data_lo, label):
    sorted_data_mean = np.sort(data_mean)
    #cdf = 100*np.arange(len(sorted_data_mean)) / float(len(sorted_data_mean))
    cdf = np.arange(1, len(sorted_data_mean) + 1) / len(sorted_data_mean) * 100

    
    sorted_data_hi = np.sort(data_hi)
    sorted_data_lo = np.sort(data_lo)
    
    plt.plot(sorted_data_mean, cdf, label=label)
    plt.fill_betweenx(cdf, sorted_data_lo, sorted_data_hi, alpha=0.2)

def get_timeseries_statistics(conn, simulation_id, element_type=None, element_idxs=None, output_name=None):
        # Build dynamic WHERE clauses
        where_clauses = ["v.simulation_output_id = :simulation_id"]
        params = {"simulation_id": simulation_id}

        if element_type is not None:
                where_clauses.append("v.extra_info ->> 'element_type' LIKE :element_type")
                params["element_type"] = element_type

        if element_idxs is not None:
                # Ensure element_idxs is a list for ANY clause
                if isinstance(element_idxs, int):
                        element_idxs = [element_idxs]
                where_clauses.append("(v.extra_info ->> 'element_index')::int = ANY(:element_idxs)")
                params["element_idxs"] = element_idxs

        if output_name is not None:
                where_clauses.append("v.extra_info ->> 'output' LIKE :output_name")
                params["output_name"] = output_name

        where_sql = " AND ".join(where_clauses)

        query = text(f"""
                WITH bus_variables AS (
                SELECT v.variable_id, v.variable_name, v.extra_info
                FROM building_power.variable v
                WHERE {where_sql}
                ),
                per_variable AS (
                SELECT
                        ot.variable_id,
                        average(ot.quantity) AS avg_quantity,
                        percentile(ot.quantity, 0.95) AS p95_quantity,
                        percentile(ot.quantity, 0.05) AS p05_quantity
                FROM building_power.output_timeseries ot
                JOIN bus_variables bv ON ot.variable_id = bv.variable_id
                GROUP BY ot.variable_id
                )
                SELECT
                AVG(avg_quantity) AS mean_quantity,
                AVG(p95_quantity) AS mean_p95_quantity,
                AVG(p05_quantity) AS mean_p05_quantity
                FROM per_variable;""")
        
        df = pd.read_sql_query(
                query,
                conn,
                params=params
        )
        return df

def explain_get_timeseries_data(
    conn: Engine | Connection,
    simulation_id: int,
    element_type: str = None,
    element_idxs: list = None,
    output_name: str = None
):
        """
        Returns EXPLAIN ANALYZE for the optimized timeseries aggregation.
        simulation_id is required. Other filters are optional.
        """

        if simulation_id is None:
                raise ValueError("simulation_id cannot be None")

        where_clauses = ["v.simulation_output_id = %(simulation_id)s"]
        params = {"simulation_id": simulation_id}

        # Optional filter: output_name (LIKE)
        if output_name is not None:
                where_clauses.append("(v.extra_info ->> 'output') LIKE %(output_name)s")
                params["output_name"] = output_name

        # Optional filter: element_type (LIKE)
        if element_type is not None:
                where_clauses.append("(v.extra_info ->> 'element_type') LIKE %(element_type)s")
                params["element_type"] = element_type

        # Optional filter: element_idxs
        if element_idxs is not None:
                if isinstance(element_idxs, int):
                        element_idxs = [element_idxs]
                        where_clauses.append("((v.extra_info ->> 'element_index')::int = ANY(%(element_idxs)s))")
                        params["element_idxs"] = element_idxs
                        where_sql = " AND ".join(where_clauses)

        sql = f"""
                EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
                SELECT
                ot.ts AS bucket,
                SUM(ot.quantity) AS total_quantity
                FROM building_power.output_timeseries ot
                JOIN building_power.variable v
                ON v.variable_id = ot.variable_id
                WHERE {where_clauses}
                GROUP BY ot.ts
                ORDER BY ot.ts;
                """

        # Pick connection type
        if isinstance(conn, Connection):
                connection = conn
        else:
                connection = conn.connect()
        
        # Execute
        result = connection.execute(text(sql), params).fetchall()

        # If we opened the connection, close it
        if isinstance(conn, Engine):
                connection.close()

        # Format output
        return "\n".join(r[0] for r in result)




def get_timeseries_data(conn, simulation_id, element_type=None, element_idxs=None, output_name=None):
        # Build dynamic WHERE clauses
        where_clauses = ["v.simulation_output_id = :simulation_id"]
        params = {"simulation_id": simulation_id}

        if element_type is not None:
                where_clauses.append("v.extra_info ->> 'element_type' LIKE :element_type")
                params["element_type"] = element_type

        if element_idxs is not None:
                # Ensure element_idxs is a list for ANY clause
                if isinstance(element_idxs, int):
                        element_idxs = [element_idxs]
                where_clauses.append("(v.extra_info ->> 'element_index')::int = ANY(:element_idxs)")
                params["element_idxs"] = element_idxs

        if output_name is not None:
                where_clauses.append("v.extra_info ->> 'output' LIKE :output_name")
                params["output_name"] = output_name

        where_sql = " AND ".join(where_clauses)

        # LOCF = last observation carried forward
        query = text(f"""
                WITH bus_variables AS (
                SELECT v.variable_id, v.variable_name, v.extra_info
                FROM building_power.variable v
                WHERE {where_sql}
                ),
                per_variable AS (
                SELECT
                        time_bucket('15 minutes', ot.ts) AS bucket,
                        ot.variable_id,
                COALESCE(
                        average(time_weight('locf', ot.ts, ot.quantity)),
                (
                        SELECT quantity
                        FROM building_power.output_timeseries
                        WHERE variable_id = ot.variable_id
                        ORDER BY ts
                        LIMIT 1
                )
        ) AS avg_quantity
                FROM building_power.output_timeseries ot
                JOIN bus_variables bv ON ot.variable_id = bv.variable_id
                GROUP BY bucket, ot.variable_id
                )
                SELECT
                bucket,
                SUM(avg_quantity) AS total_quantity
                FROM per_variable
                GROUP BY bucket
                ORDER BY bucket;""")
        
        df = pd.read_sql_query(
                query,
                conn,
                params=params
        )
        return df



def get_timeseries_data_new(conn, simulation_id, element_type=None, element_idxs=None, output_name=None):
        """
        Retrieves aggregated timeseries data using the optimized Continuous Aggregate.
        Note: Returns SUM(power) as total_power, not SUM(AVG(power)).
        """
        # Build dynamic WHERE clauses
        
        # Filters applied to the variable table (v)
        where_clauses = ["v.simulation_output_id = :simulation_id"]
        params = {"simulation_id": simulation_id}

        if element_type is not None:
                where_clauses.append("(v.extra_info ->> 'element_type') LIKE :element_type")
                params["element_type"] = element_type

        if element_idxs is not None:
                # Ensure element_idxs is a list for ANY clause
                if isinstance(element_idxs, int):
                        element_idxs = [element_idxs]
                where_clauses.append("((v.extra_info ->> 'element_index')::int = ANY(:element_idxs))")
                params["element_idxs"] = element_idxs

        if output_name is not None:
                where_clauses.append("(v.extra_info ->> 'output') LIKE :output_name")
                params["output_name"] = output_name

        where_sql = " AND ".join(where_clauses)

        query = text(f"""
                WITH 
                bus_variables AS (
                        SELECT v.variable_id
                        FROM building_power.variable v
                        WHERE {where_sql}
                )                 
                SELECT
                        agg.bucket,
                        SUM(agg.sum_quantity) AS total_quantity
                FROM
                       building_power.power_15min_by_variable agg                
                WHERE agg.variable_id IN (SELECT variable_id FROM bus_variables)
                GROUP BY
                agg.bucket
                ORDER BY
                agg.bucket;""")
        
        df = pd.read_sql_query(
                query,
                conn,
                params=params
        )
        return df