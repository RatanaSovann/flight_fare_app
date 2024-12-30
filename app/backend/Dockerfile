# Use the official Python image from the Docker Hub
FROM python:3.11.4

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY app/backend/requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend directory into the container
COPY app/backend /app

# Copy the models from the host to the container
COPY models /app/models

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
