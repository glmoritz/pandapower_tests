--select loads with average power $1 and maximum deviation $2
SELECT 
    bldg_id,
    AVG(electricity_total_energy_consumption) / 24 AS avg_daily_power_kw,
    ABS((AVG(electricity_total_energy_consumption) / 24) - $1) AS deviation_kw
FROM 
    building_power.daily_energy
WHERE 
    electricity_total_energy_consumption IS NOT NULL
GROUP BY 
    bldg_id
HAVING 
    ABS((AVG(electricity_total_energy_consumption) / 24) - $1) <= $2
ORDER BY 
    ABS((AVG(electricity_total_energy_consumption) / 24) - $1) ASC
LIMIT 10;

SELECT DISTINCT time_bucket('1 day', sample_time) AS day
        FROM building_power.building_power
        WHERE bldg_id = $3
        ORDER BY day

