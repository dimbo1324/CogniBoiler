proxy_pass http://gateway;
proxy_http_version 1.1;
proxy_set_header Host $host;
# nginx is the edge: the client's address replaces anything the client claimed.
proxy_set_header X-Forwarded-For $remote_addr;
proxy_set_header X-Forwarded-Proto $scheme;
proxy_set_header X-Real-IP $remote_addr;
