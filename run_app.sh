#!/bin/bash
# Script to run the Heart Disease Prediction Web Application

echo "Setting up Heart Disease Prediction App..."

# Create directories if they don't exist
mkdir -p models
mkdir -p visualizations

# Run the scikit_parallel_pipeline if models don't exist
if [ ! -f "models/random_forest_model.pkl" ]; then
    echo "Training models using GPU acceleration..."
    python scikit_parallel_pipeline.py
else
    echo "Models already exist, skipping training step."
fi

# Install React dependencies if needed
cd webapp
if [ ! -d "node_modules" ]; then
    echo "Installing React dependencies..."
    npm install
else
    echo "React dependencies already installed."
fi

# Build React app
echo "Building React frontend..."
npm run build

# Start backend server
cd ..
echo "Starting Flask backend server..."
python app.py
