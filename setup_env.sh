#!/usr/bin/env bash

# This script is intended to be sourced:
#   source ./setup_env.sh
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Run this script with: source ./setup_env.sh"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RLNAV_ROOT="$ROOT_DIR"
export ROS_DISTRO="${ROS_DISTRO:-jazzy}"
export RLNAV_MODEL_PATH="$RLNAV_ROOT/nav_core/models/finalmodel.zip"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/ros2_logs}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"
if [[ -f "$ROS_SETUP" ]]; then
  # shellcheck disable=SC1090
  source "$ROS_SETUP"
else
  echo "[setup_env] ROS setup not found: $ROS_SETUP"
fi

mkdir -p "$ROS_LOG_DIR" "$MPLCONFIGDIR"

VENV_ACTIVATE="$RLNAV_ROOT/.venv/bin/activate"
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
else
  echo "[setup_env] venv not found: $VENV_ACTIVATE"
fi

ROS2_WS_SETUP="$RLNAV_ROOT/ros2_nav/install/setup.bash"
if [[ -f "$ROS2_WS_SETUP" ]]; then
  # shellcheck disable=SC1090
  source "$ROS2_WS_SETUP"
fi

if [[ -n "${PYTHONPATH-}" ]]; then
  export PYTHONPATH="$RLNAV_ROOT:$PYTHONPATH"
else
  export PYTHONPATH="$RLNAV_ROOT"
fi

echo "[setup_env] RLNAV_ROOT=$RLNAV_ROOT"
echo "[setup_env] ROS_DISTRO=$ROS_DISTRO"
echo "[setup_env] RLNAV_MODEL_PATH=$RLNAV_MODEL_PATH"
echo "[setup_env] ROS_LOG_DIR=$ROS_LOG_DIR"
echo "[setup_env] MPLCONFIGDIR=$MPLCONFIGDIR"
