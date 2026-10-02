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

mkdir -p /tmp/cogniboiler-nginx
cat > /tmp/cogniboiler-nginx/tls.conf <<'CONF'
server {
    listen 8443 ssl;
    http2 on;
    server_name _;
    ssl_certificate /tmp/tls/cert.pem;
    ssl_certificate_key /tmp/tls/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305;
    ssl_prefer_server_ciphers off;
    ssl_session_tickets off;
    add_header Strict-Transport-Security $cogniboiler_hsts always;
    include /etc/nginx/cogniboiler/site.inc;
}
CONF
echo "cogniboiler: HTTPS on 8443"
