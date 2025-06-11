FROM python:3.12-slim

WORKDIR /app

# Update and upgrade packages to fix vulnerabilities, and install necessary tools
RUN apt-get update && \
	apt-get install -y \
		ffmpeg && \
	apt-get upgrade -y && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Install kitchen20
RUN pip install -e ./lib/kitchen20
# Install additional dependencies
RUN pip install -r ./lib/kitchen20/requirements.txt

# Run the application
CMD ["python", "main.py"]