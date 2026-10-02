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

# The API documentation. FastAPI's pages load Swagger UI and ReDoc from jsDelivr by major
# version; nginx rewrites them to exact versions with Subresource Integrity, so a new release
# on the CDN cannot run on the console's origin. If FastAPI changes those URLs, the policy in
# default.conf blocks the unpinned files and the page fails visibly. No add_header here:
# a location with its own add_header would drop the server's security headers.
location ~ ^/(docs|redoc|openapi\.json)$ {
    include /etc/nginx/cogniboiler/proxy.inc;
    proxy_set_header Accept-Encoding "";
    sub_filter 'https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"' 'https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.33.0/swagger-ui-bundle.js" integrity="sha384-YDALVcy8kj8yltLBVi1vBiBAUqdxvus673gM8XKwiy6aDUJFXivF/KCufekjYbVf" crossorigin="anonymous"';
    sub_filter 'https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css"' 'https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.33.0/swagger-ui.css" integrity="sha384-Ov4/wv3j2bmct8cDc5X4ngJZohVPzEmc6uDPH8WeljUxO5vtoykvMEfbu9Vh6RaW" crossorigin="anonymous"';
    sub_filter 'https://cdn.jsdelivr.net/npm/redoc@2/bundles/redoc.standalone.js"' 'https://cdn.jsdelivr.net/npm/redoc@2.5.4/bundles/redoc.standalone.js" integrity="sha384-w447zOpYfw/1Tv/5AK9NfHTlQIqE3RVR6KY62jCyy9zNDgO64cMwGGP1Fj0zJVf5" crossorigin="anonymous"';
}
