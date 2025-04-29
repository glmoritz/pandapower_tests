# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.13-bookworm

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN adduser -u 5678 --disabled-password --gecos "" appuser

RUN apt update

RUN apt install -y build-essential git gfortran cmake libopenblas-dev

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /pandapower/app

#COPY ./apis /sigivest/apis
#COPY ./app /sigivest/app

#COPY ./apis /sigivest/apis
COPY ./solar_data /pandapower/solar_data
COPY ./*.json /pandapower
COPY ./*.ipynb /pandapower
COPY ./*.csv /pandapower
COPY ./*.py /pandapower

#ENV GIT_HASH=$GIT_HASH
#ENV GIT_REMOTE=$GIT_REMOTE

WORKDIR /pandapower/app
#RUN rm -rf sigivest_apis
#RUN rm -rf google
#RUN mkdir sigivest_apis
#RUN python -m grpc_tools.protoc -I ../apis --pyi_out=./sigivest_apis --python_out=./sigivest_apis --grpc_python_out=./sigivest_apis $(find ../apis -iname "*.proto") 
#RUN mv sigivest_apis/google .

# Creates a non-root user with an explicit UID and adds permission to access the /sigivest folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN chown -R appuser /pandapower
#USER appuser

#EXPOSE 50001
WORKDIR /pandapower
# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "-m", "app"]