#!/bin/sh
# Adds an HTTPS server on 8443 with the same locations when WEB_TLS_CERT and WEB_TLS_KEY
# (PEM, from .env) are set; without them the console is served over HTTP only.
set -eu

if [ -z "${WEB_TLS_CERT:-}" ] || [ -z "${WEB_TLS_KEY:-}" ]; then
    echo "cogniboiler: no TLS certificate configured; HTTP only on 8080"
    exit 0
fi

umask 077
mkdir -p /tmp/tls
printf '%s\n' "$WEB_TLS_CERT" > /tmp/tls/cert.pem
printf '%s\n' "$WEB_TLS_KEY" > /tmp/tls/key.pem

cat > /etc/nginx/conf.d/tls.conf <<'CONF'
server {
    listen 8443 ssl;
    http2 on;
    server_name _;
    ssl_certificate /tmp/tls/cert.pem;
    ssl_certificate_key /tmp/tls/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers off;
    add_header Strict-Transport-Security "max-age=31536000" always;
    include /etc/nginx/cogniboiler/site.inc;
}
CONF
echo "cogniboiler: HTTPS on 8443"
