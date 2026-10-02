add_header X-Content-Type-Options nosniff always;
add_header X-Frame-Options DENY always;
add_header Referrer-Policy no-referrer always;
add_header Permissions-Policy "camera=(), microphone=(), geolocation=()" always;
# The policies are in default.conf: the console's everywhere, a CDN-pinned one for the
# API documentation pages.
add_header Content-Security-Policy $cogniboiler_csp always;
