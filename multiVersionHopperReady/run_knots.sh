#!/bin/bash
#SBATCH --job-name=alphazero_knots      # Job name
#SBATCH --output=alphazero_knots.out    # Output log file
#SBATCH --error=alphazero_knots.err     # Error log file
#SBATCH --time=24:00:00                 # Max run time (adjust as needed)
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Total number of tasks (1 main Python script)
#SBATCH --cpus-per-task=8               # Use 8 CPU cores
#SBATCH --mem=32G                       # Memory allocation
#SBATCH --partition=general             # Use the general partition (CPU-only)

# Check if SageMath is installed on the compute node
if ! command -v sage &> /dev/null
then
    echo "SageMath not found. Installing locally..."

    # Define installation directory
    SAGE_DIR="$HOME/sage"
    
    # Download and install SageMath (if not already installed)
    if [ ! -d "$SAGE_DIR" ]; then
        mkdir -p "$SAGE_DIR"
        cd "$SAGE_DIR"
        
        # Download the latest SageMath source (10.5)
        wget https://www-ftp.lip6.fr/pub/math/sagemath/src/sage-10.5.tar.gz

        # Extract and compile SageMath
        tar -xvf sage-10.5.tar.gz
        cd sage-10.5
        ./configure
        make -j8  # Compile using 8 CPU cores
    fi

    # Add SageMath to PATH
    export PATH="$SAGE_DIR/sage-10.5:$PATH"
    echo "SageMath installed successfully!"
else
    echo "SageMath found: $(which sage)"
fi

# Install snappy inside SageMath's Python environment
echo "Installing snappy inside SageMath..."
sage -python -m pip install --upgrade pip  # Upgrade pip inside SageMath
sage -python -m pip install snappy
# Run the Python script using SageMath
sage -python main_multi.py
