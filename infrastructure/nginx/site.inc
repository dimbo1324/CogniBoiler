server_tokens off;
client_max_body_size 64k;

include /etc/nginx/cogniboiler/headers.inc;

location / {
    root /usr/share/nginx/html;
    try_files $uri /index.html;
}

location ~ ^/(api|auth)/ {
    include /etc/nginx/cogniboiler/proxy.inc;
}

location ~ ^/(health|ready)$ {
    include /etc/nginx/cogniboiler/proxy.inc;
}

# The WebSocket channels; a connection may stay open for hours.
location = /ws {
    include /etc/nginx/cogniboiler/proxy.inc;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection $connection_upgrade;
    proxy_read_timeout 1h;
    proxy_send_timeout 1h;
}

# The OpenAPI documentation loads Swagger UI from a CDN, so it gets its own policy.
location ~ ^/(docs|redoc|openapi\.json)$ {
    include /etc/nginx/cogniboiler/proxy.inc;
    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options DENY always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data: https://fastapi.tiangolo.com https://cdn.redoc.ly; worker-src 'self' blob:; frame-ancestors 'none'" always;
}
