#!/bin/sh
# Writes the broker's accounts from the environment (filled from .env by Compose), then
# starts Mosquitto. Runs as root only to set file ownership; Mosquitto drops to its own user.
set -eu
umask 077

auth=/mosquitto/auth
mkdir -p "$auth"
passwd="$auth/passwd"
: > "$passwd"

account() {
    password=$(printenv "$2" || true)
    if [ -z "$password" ]; then
        echo "start-broker: $2 is empty; run: python dev_tools_scripts_runner.py dev-secrets" >&2
        exit 1
    fi
    mosquitto_passwd -b "$passwd" "$1" "$password"
}

account physics-engine MQTT_PHYSICS_ENGINE_PASSWORD
account plc-controller MQTT_PLC_CONTROLLER_PASSWORD
account alert-manager MQTT_ALERT_MANAGER_PASSWORD
account historian MQTT_HISTORIAN_PASSWORD
account api-gateway MQTT_API_GATEWAY_PASSWORD
account opcua-server MQTT_OPCUA_SERVER_PASSWORD
account monitor MQTT_MONITOR_PASSWORD

cp /cogniboiler/acl "$auth/acl"
chown -R mosquitto:mosquitto "$auth"
chmod 0700 "$auth"
chmod 0600 "$passwd" "$auth/acl"

exec /docker-entrypoint.sh mosquitto -c /mosquitto/config/mosquitto.conf
