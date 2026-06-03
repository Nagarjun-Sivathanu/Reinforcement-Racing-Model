# Rl_Race — SAC trainer for DonkeySim.
#
# This image contains ONLY the training agent. DonkeySim itself runs OUTSIDE the container
# (it's a GUI simulator, typically on a Windows/desktop host). Point the container at it with
# the SIM_HOST / SIM_PORT environment variables.
#
# Build:  docker build -t rl_race .
# Run:    docker run --rm -e SIM_HOST=<sim-ip> -v "$PWD/models:/app/models" \
#                 -v "$PWD/checkpoints:/app/checkpoints" -v "$PWD/tb:/app/tb" \
#                 rl_race --timesteps 300000
FROM python:3.10-slim

# git is needed to install the pinned gym-donkeycar from source.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch (CPU) from its dedicated index, then the rest of the deps. Done before
# copying the source so dependency layers stay cached across code changes.
RUN pip install --no-cache-dir torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code + the recorded track waypoints.
COPY *.py ./
COPY tracks/ ./tracks/

# Where the sim lives (override at runtime). Empty SIM_HOST -> code falls back to gateway.
ENV SIM_HOST="" \
    SIM_PORT="9091" \
    PYTHONUNBUFFERED="1"

# Outputs persist via volume mounts (see docker-compose.yml).
VOLUME ["/app/models", "/app/checkpoints", "/app/tb"]

ENTRYPOINT ["python", "train_rl.py"]
CMD ["--timesteps", "300000"]
