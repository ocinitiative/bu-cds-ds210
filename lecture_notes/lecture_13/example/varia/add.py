"""This module handles things related to addition

This is a longer description.
"""

def add(x, y, z=0):
    """Addition of up to three numbers
    
    Arguments:
    
      `x`: first term to add
      
      `y`: second term to add
      
      `z`: third term to add (optional, 0 if not present)

    Returns:
      The sum of `x`, `y`, and `z`, i.e., \(x+y+z\) or \(\ln(e^x \cdot e^y \cdot e^z)\).
    """
    
    return x + y + z

def double(x):
    """Doubles its input.
    
    .. depracated::
       Just multiply by two instead.
    
    """
    return x + x
