FROM python:3.12-slim

WORKDIR /app

# Update and upgrade packages to fix vulnerabilities, and install necessary tools
RUN apt-get update && \
	apt-get install -y \
		portaudio19-dev && \
	apt-get upgrade -y && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Install kitchen20 and common libraries
RUN pip install -e ./lib/kitchen20-pytorch
RUN pip install -e ./lib/common

# Run the application
CMD ["python", "main.py"]