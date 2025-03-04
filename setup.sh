#!/bin/bash

# This setup script configures the development environment for a reinforcement learning project.
# It includes installing necessary Python packages and ensuring compatibility with macOS on an Apple M3 Pro chip (arm64 architecture).

# Define a function to install Python dependencies
install_dependencies() {
    echo "Upgrading pip..."
    python3 -m pip install --upgrade pip
    
    echo "Installing required Python packages: torch, simpy, plotly, dash..."
    pip3 install torch simpy plotly dash

    # Check if the installation was successful
    if [ $? -ne 0 ]; then
        echo "Installation encountered an error, please check your network connection."
    else
        echo "Python dependencies installed successfully."
    fi
}

# Function to provide instructions in case of issues
display_installation_help() {
    echo "If you encounter issues with installation, consider the following steps:"
    echo "1. Ensure your Python version is updated (Python 3.x is recommended)."
    echo "2. Verify your internet connection."
    echo "3. Check if you have the necessary permissions to install packages."
    echo "4. Try running the script with sudo permissions: sudo ./setup.sh"
}

# Main function to set up the development environment
main() {
    echo "Initializing the environment setup..."
    install_dependencies
    display_installation_help
    echo "Environment setup completed. You can now proceed with running the project."
}

# Invoke the main function
main