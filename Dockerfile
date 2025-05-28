# Set up the base image
FROM python:3.11.11

# Set the working directory
WORKDIR /app/

# Copy the requirements file to workdir
COPY requirements-docker.txt .

# Install the requirements
RUN pip install -r requirements-docker.txt

# Copy the data files
COPY ./data/external/plot_data.csv ./data/external/plot_data.csv 
COPY ./data/processed/test.csv ./data/processed/test.csv

# Copy the models
COPY ./models/ ./models/ 

# Copy the code files
COPY ./app.py ./app.py

# Expose the port on the container
EXPOSE 8000

# Run the streamlit app
CMD [ "streamlit", "run", "app.py", "--server.port", "8000", "--server.address", "0.0.0.0"]