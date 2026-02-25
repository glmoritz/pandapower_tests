"""
Bucket Aggregator for Time-Weighted Averaging
----------------------------------------------
Handles non-uniform simulation timesteps by aggregating data into fixed-size buckets.

For power/rate variables: computes time-weighted average
For state/cumulative variables: takes the last value in the bucket

Usage:
    aggregator = BucketAggregator(bucket_size_s=900, start_time=sim_start)
    
    # In simulation step:
    aggregator.add_value(time_s, 'householdproducer', 'bus3', 'P_a_load[MW]', 0.5, 'MW')
    
    # Get completed buckets for DB insertion:
    completed = aggregator.get_completed_buckets()
    for record in completed:
        # Insert into simulation_data table
        ...
    
    # At simulation end:
    final_records = aggregator.flush_all()
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import re


@dataclass
class VariableAccumulator:
    """Tracks contributions to a single variable within a bucket."""
    total_weighted_value: float = 0.0  # Σ(value × duration)
    total_duration: float = 0.0         # Σ(duration)
    last_value: float = 0.0             # Most recent value
    last_time_s: float = 0.0            # Time of last value


@dataclass  
class BucketData:
    """Holds all variable accumulators for a single time bucket."""
    bucket_start_s: float               # Bucket start in simulation seconds
    bucket_end_s: float                 # Bucket end in simulation seconds
    variables: Dict[Tuple[str, str, str], VariableAccumulator] = field(default_factory=dict)
    # Key = (element_type, element_index, variable_name)


class BucketAggregator:
    """
    Aggregates simulation data into fixed-size time buckets with time-weighted averaging.
    All variables use time-weighted averaging.
    """
    
    def __init__(self, bucket_size_s: int, start_datetime: datetime):
        """
        Initialize the aggregator.
        
        Args:
            bucket_size_s: Size of each bucket in seconds (e.g., 900 for 15 min)
            start_datetime: Simulation start time as datetime
        """
        self.bucket_size_s = bucket_size_s
        self.start_datetime = start_datetime
        self.buckets: Dict[int, BucketData] = {}  # bucket_index -> BucketData
        self.completed_buckets: List[BucketData] = []
    
    def _get_bucket_index(self, time_s: float) -> int:
        """Get the bucket index for a given simulation time."""
        return int(time_s // self.bucket_size_s)
    
    def _get_bucket_bounds(self, bucket_index: int) -> Tuple[float, float]:
        """Get (start_s, end_s) for a bucket index."""
        start_s = bucket_index * self.bucket_size_s
        end_s = (bucket_index + 1) * self.bucket_size_s
        return start_s, end_s
    
    def _ensure_bucket(self, bucket_index: int) -> BucketData:
        """Get or create a bucket."""
        if bucket_index not in self.buckets:
            start_s, end_s = self._get_bucket_bounds(bucket_index)
            self.buckets[bucket_index] = BucketData(
                bucket_start_s=start_s,
                bucket_end_s=end_s
            )
        return self.buckets[bucket_index]
    
    def add_value(
        self,
        time_s: float,
        element_type: str,
        element_index: Optional[str],
        variable_name: str,
        value: float,
        unit: Optional[str] = None
    ):
        """
        Add a value at the given simulation time.
        
        Handles bucket transitions: if the time crosses into a new bucket,
        the contribution is split proportionally.
        
        Args:
            time_s: Current simulation time in seconds
            element_type: Type of element ('householdproducer', 'bus', etc.)
            element_index: Index/identifier of the element (can be None for global)
            variable_name: Name of the variable ('P_a_load[MW]', etc.)
            value: The value at this time
            unit: Optional unit string
        """
        bucket_index = self._get_bucket_index(time_s)
        key = (element_type, element_index or '', variable_name)
        
        bucket = self._ensure_bucket(bucket_index)
        
        if key not in bucket.variables:
            bucket.variables[key] = VariableAccumulator(
                last_time_s=bucket.bucket_start_s  # Start from bucket beginning
            )
        
        acc = bucket.variables[key]
        
        # Calculate duration since last update
        duration = time_s - acc.last_time_s
        
        if duration > 0:
            # Accumulate time-weighted contribution
            # Use the PREVIOUS value for the duration (step function assumption)
            acc.total_weighted_value += acc.last_value * duration
            acc.total_duration += duration
        
        # Update last value and time
        acc.last_value = value
        acc.last_time_s = time_s
        
        # Check if we've moved past any buckets
        self._check_bucket_completion(bucket_index)
    
    def _check_bucket_completion(self, current_bucket_index: int):
        """Move completed buckets to the completed list."""
        completed_indices = [
            idx for idx in self.buckets.keys() 
            if idx < current_bucket_index
        ]
        
        for idx in sorted(completed_indices):
            bucket = self.buckets.pop(idx)
            self._finalize_bucket(bucket)
            self.completed_buckets.append(bucket)
    
    def _finalize_bucket(self, bucket: BucketData):
        """
        Finalize all variables in a bucket by computing their time-weighted average.
        Extends the last value to the bucket end before computing.
        """
        for key, acc in bucket.variables.items():
            # Extend last value to bucket end
            remaining_duration = bucket.bucket_end_s - acc.last_time_s
            if remaining_duration > 0:
                acc.total_weighted_value += acc.last_value * remaining_duration
                acc.total_duration += remaining_duration
            
            # Compute time-weighted average
            if acc.total_duration > 0:
                acc.last_value = acc.total_weighted_value / acc.total_duration
            # else: keep last_value as is (no duration means single point)
    
    def get_completed_buckets(self) -> List[Dict]:
        """
        Get all completed buckets as a list of records ready for DB insertion.
        
        Returns:
            List of dicts with keys: bucket_datetime, element_type, element_index, 
                                      variable_name, value, unit
        """
        records = []
        
        for bucket in self.completed_buckets:
            bucket_datetime = self.start_datetime + timedelta(seconds=bucket.bucket_start_s)
            
            for (element_type, element_index, variable_name), acc in bucket.variables.items():
                # Extract unit from variable name if present
                unit = None
                unit_match = re.search(r'\[([^\]]+)\]', variable_name)
                if unit_match:
                    unit = unit_match.group(1)
                
                records.append({
                    'bucket': bucket_datetime,
                    'element_type': element_type,
                    'element_index': element_index if element_index else None,
                    'variable_name': variable_name,
                    'value': acc.last_value,
                    'unit': unit
                })
        
        # Clear completed buckets after retrieval
        self.completed_buckets = []
        
        return records
    
    def flush_all(self) -> List[Dict]:
        """
        Finalize and return all remaining buckets (call at simulation end).
        
        Returns:
            List of records for all remaining data
        """
        # Move all current buckets to completed
        for idx in sorted(self.buckets.keys()):
            bucket = self.buckets[idx]
            self._finalize_bucket(bucket)
            self.completed_buckets.append(bucket)
        
        self.buckets.clear()
        
        return self.get_completed_buckets()
    
    def get_pending_count(self) -> int:
        """Get the number of buckets currently being accumulated."""
        return len(self.buckets)
    
    def get_completed_count(self) -> int:
        """Get the number of completed buckets waiting to be retrieved."""
        return len(self.completed_buckets)


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_basic_aggregation():
    """Test basic time-weighted averaging."""
    from datetime import datetime
    
    agg = BucketAggregator(bucket_size_s=60, start_datetime=datetime(2025, 1, 1, 0, 0, 0))
    
    # Simulate non-uniform timesteps within a 60-second bucket
    agg.add_value(0, 'house', '1', 'Power[MW]', 1.0)   # 1.0 MW from 0-30s
    agg.add_value(30, 'house', '1', 'Power[MW]', 2.0)  # 2.0 MW from 30-60s
    agg.add_value(60, 'house', '1', 'Power[MW]', 3.0)  # Triggers bucket completion
    
    records = agg.get_completed_buckets()
    
    assert len(records) == 1
    # Time-weighted average: (1.0 × 30 + 2.0 × 30) / 60 = 1.5
    assert abs(records[0]['value'] - 1.5) < 0.01
    print("✓ Basic aggregation test passed")


def test_state_variable():
    """Test that SOC also uses time-weighted averaging (all variables are treated the same)."""
    from datetime import datetime
    
    agg = BucketAggregator(bucket_size_s=60, start_datetime=datetime(2025, 1, 1, 0, 0, 0))
    
    agg.add_value(0, 'house', '1', 'SOC[MWh]', 0.5)   # 0.5 from 0-30s
    agg.add_value(30, 'house', '1', 'SOC[MWh]', 0.8)  # 0.8 from 30-60s
    agg.add_value(60, 'house', '1', 'SOC[MWh]', 0.9)
    
    records = agg.get_completed_buckets()
    
    assert len(records) == 1
    # Time-weighted average: (0.5 × 30 + 0.8 × 30) / 60 = 0.65
    assert abs(records[0]['value'] - 0.65) < 0.01
    print("✓ SOC time-weighted average test passed")


def test_multiple_buckets():
    """Test aggregation across multiple buckets."""
    from datetime import datetime
    
    agg = BucketAggregator(bucket_size_s=60, start_datetime=datetime(2025, 1, 1, 0, 0, 0))
    
    agg.add_value(0, 'house', '1', 'Power[MW]', 1.0)
    agg.add_value(60, 'house', '1', 'Power[MW]', 2.0)
    agg.add_value(120, 'house', '1', 'Power[MW]', 3.0)
    
    records = agg.get_completed_buckets()
    
    assert len(records) == 2
    assert abs(records[0]['value'] - 1.0) < 0.01  # Bucket 0: constant 1.0
    assert abs(records[1]['value'] - 2.0) < 0.01  # Bucket 1: constant 2.0
    print("✓ Multiple buckets test passed")


if __name__ == '__main__':
    test_basic_aggregation()
    test_state_variable()
    test_multiple_buckets()
    print("\nAll tests passed!")
