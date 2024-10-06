# Use the official Python 3.10 image from the Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . /app/

# Expose the port that Streamlit will run on
EXPOSE 8501

# Run Streamlit when the container launches
CMD ["streamlit", "run", "app.py"]