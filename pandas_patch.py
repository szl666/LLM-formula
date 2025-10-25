#!/usr/bin/env python3
"""
Pandas compatibility patch for matminer
This script patches the pandas DataFrame.append method for older matminer versions
"""

import pandas as pd

# Check if append method exists, if not, add it back for compatibility
if not hasattr(pd.DataFrame, 'append'):
    def append_method(self, other, ignore_index=False, verify_integrity=False, sort=False):
        """
        Compatibility method to replace removed DataFrame.append
        """
        return pd.concat([self, other], ignore_index=ignore_index, 
                        verify_integrity=verify_integrity, sort=sort)
    
    # Add the method back to DataFrame
    pd.DataFrame.append = append_method
    print("Added pandas DataFrame.append compatibility method")
else:
    print("pandas DataFrame.append method already exists")



