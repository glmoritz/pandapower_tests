# Network Database Schema Update

This document explains the updated database schema for storing pandapower networks with composite keys and how to use the migration scripts.

## New Database Schema

### Key Changes

1. **Networks Table**: Central table to store multiple networks
   - `network_id`: Primary key
   - `network_name`: Unique name for each network
   - `network_description`: Optional description
   - `network_metadata`: JSON metadata about the network

2. **Network Vertices Table**: Stores bus information with composite key
   - Primary key: `(network_id, pandapower_bus_index)`
   - `pandapower_bus_index`: Corresponds directly to pandapower bus index
   - `vertix_label`: Bus name/label
   - `vertix_metadata`: JSON metadata for the bus

3. **Network Edges Table**: Stores connections between buses
   - `network_id`: Links to the network
   - `source_pandapower_bus_index`, `target_pandapower_bus_index`: Bus connections
   - Foreign key constraints ensure referential integrity

## Migration Process

### 1. SQL Migration

Run the SQL migration script to update your database structure:

```bash
# Make the script executable
chmod +x run_migration.sh

# Run the migration
./run_migration.sh
```

Or run the SQL directly:

```bash
psql -h localhost -d your_database -U your_user -f migrate_database_structure.sql
```

### 2. Python Migration

Use the Python migration utility for more advanced migration tasks:

```python
from migrate_database import NetworkDatabaseMigrator

# Database connection parameters
connection_params = {
    'host': 'localhost',
    'database': 'your_database',
    'user': 'your_user',
    'password': 'your_password',
    'port': '5432'
}

migrator = NetworkDatabaseMigrator(connection_params)
migrator.connect()
migrator.run_migration()
migrator.disconnect()
```

## Using the Updated Functions

### Saving Networks

```python
from create_random_network import save_network_to_database, generate_pandapower_net_old

# Generate a network
net, graph = generate_pandapower_net_old(
    CommercialRange=(1, 2),
    IndustrialRange=(1, 2), 
    ResidencialRange=(2, 3),
    ForkLengthRange=(50, 200),
    LineBusesRange=(3, 8),
    LineForksRange=(2, 4)
)

# Save to database
db_connection = {
    'host': 'localhost',
    'database': 'pandapower_db',
    'user': 'postgres',
    'password': 'your_password'
}

network_id, grid_id = save_network_to_database(
    graph, net, db_connection, "my_network_v1"
)
```

### Loading Networks

```python
from create_random_network import load_network_from_database

# Load from database
loaded_net, loaded_graph = load_network_from_database(
    db_connection, "my_network_v1"
)
```

### Listing Networks

```python
from create_random_network import list_networks_in_database

# Get all networks
networks_df = list_networks_in_database(db_connection)
print(networks_df[['network_id', 'network_name', 'vertex_count', 'edge_count']])
```

### Deleting Networks

```python
from create_random_network import delete_network_from_database

# Delete a network
success = delete_network_from_database(db_connection, "my_network_v1")
```

## Benefits of the New Schema

1. **Multiple Networks**: Store and manage multiple pandapower networks in one database
2. **Direct Bus Mapping**: Bus indexes in the database match pandapower bus indexes exactly
3. **No ID Conflicts**: Each network has its own vertex space
4. **Referential Integrity**: Foreign key constraints ensure data consistency
5. **Efficient Queries**: Optimized indexes for fast network retrieval
6. **Metadata Support**: Rich JSON metadata storage for buses and connections

## Migration Safety

- **Backup Creation**: The migration script automatically creates backup tables if existing data is found
- **Transaction Safety**: All operations are wrapped in transactions
- **Data Preservation**: Existing data is migrated to the new schema format
- **Rollback Support**: Failed migrations are automatically rolled back

## Database Schema Diagram

```
┌─────────────────┐
│    networks     │
├─────────────────┤
│ network_id (PK) │
│ network_name    │
│ network_desc    │
│ created_at      │
│ metadata        │
└─────────────────┘
         │
         │ 1:N
         ▼
┌─────────────────────────────┐
│     network_vertices        │
├─────────────────────────────┤
│ network_id (FK, PK)         │
│ pandapower_bus_index (PK)   │
│ vertix_label                │
│ vertix_metadata             │
└─────────────────────────────┘
         │
         │ 1:N
         ▼
┌─────────────────────────────┐
│     network_edges           │
├─────────────────────────────┤
│ network_edge_id (PK)        │
│ network_id (FK)             │
│ source_bus_index (FK)       │
│ target_bus_index (FK)       │
│ directed                    │
│ edge_metadata               │
└─────────────────────────────┘
```

## Example Usage

See `example_network_database.py` for a complete example of:
- Generating a random network
- Saving it to the database
- Loading it back
- Comparing original vs loaded data
- Managing multiple networks

## Troubleshooting

### Migration Issues

1. **Check PostgreSQL version**: Ensure you have PostgreSQL 10+ for JSON support
2. **Schema permissions**: Ensure your user has CREATE permissions on the schema
3. **Backup tables**: If migration fails, backup tables are preserved for manual recovery

### Connection Issues

1. **Port parameter**: The pandapower `to_postgresql` function may not support a `port` parameter
2. **Database name**: Use either `database` or `dbname` in connection parameters
3. **Schema access**: Ensure the `building_power` schema exists and is accessible

### Performance

1. **Indexes**: The migration creates appropriate indexes for efficient querying
2. **Large networks**: For very large networks, consider using chunked operations
3. **JSON queries**: Use GIN indexes for efficient JSON metadata queries

## Version Compatibility

- **Pandapower**: Tested with pandapower 2.x
- **PostgreSQL**: Requires PostgreSQL 10+ (for JSON support)
- **Python**: Tested with Python 3.8+
- **Dependencies**: psycopg2, pandas, networkx, shapely, geopy
