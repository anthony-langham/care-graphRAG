# Use the official AWS Lambda Python 3.11 base image
FROM public.ecr.aws/lambda/python:3.11

# Set the working directory in the container
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the function source code into the container
COPY src/ ${LAMBDA_TASK_ROOT}

# Set the command to run when the container starts
CMD [ "functions.health.handler" ]
