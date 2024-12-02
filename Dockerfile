# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY neuroinfer /app/neuroinfer
COPY requirements.txt /app/
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000
EXPOSE 8000:8010

# Define environment variable
ENV FLASK_APP=neuroinfer.server

# Run server when the container launches
CMD ["python", "-m", "neuroinfer.server"]
