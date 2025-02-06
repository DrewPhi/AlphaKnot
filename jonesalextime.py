import snappy
import time

# Define the PD code
pd_code = [[3, 1, 4, 26], [1, 8, 2, 9], [7, 2, 8, 3], [9, 5, 10, 4], [5, 14, 6, 15], [13, 6, 14, 7], [10, 15, 11, 16], [20, 11, 21, 12], [12, 21, 13, 22], [16, 23, 17, 24], [24, 17, 25, 18], [18, 25, 19, 26], [22, 19, 23, 20]]

def compute_polynomials(pd_code):
    try:
        # Create the link from PD code
        link = snappy.Link(pd_code)
        
        # Compute and time the Alexander polynomial
        start_time = time.time()
        alexander_poly = link.alexander_polynomial()
        alexander_time = time.time() - start_time
        
        # Compute and time the Jones polynomial
        start_time = time.time()
        jones_poly = link.jones_polynomial()
        jones_time = time.time() - start_time

        # Print results
        print(f"Alexander Polynomial: {alexander_poly}")
        print(f"Time taken: {alexander_time:.6f} seconds")
        print(f"Jones Polynomial: {jones_poly}")
        print(f"Time taken: {jones_time:.6f} seconds")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the computation
compute_polynomials(pd_code)
