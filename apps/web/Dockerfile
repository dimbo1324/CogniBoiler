# The operator console: built with pnpm, then served by an unprivileged nginx that also
# proxies the gateway's public routes, so the browser talks to one origin.
# Built by Compose with the extra build context `nginx` = infrastructure/nginx.

FROM node:24-alpine AS build
ENV COREPACK_ENABLE_DOWNLOAD_PROMPT=0
WORKDIR /web
RUN corepack enable
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY . .
RUN pnpm run build

FROM nginxinc/nginx-unprivileged:1.29-alpine
COPY --from=nginx default.conf /etc/nginx/conf.d/default.conf
COPY --from=nginx site.inc headers.inc proxy.inc /etc/nginx/cogniboiler/
COPY --from=nginx --chmod=0755 40-cogniboiler-tls.sh /docker-entrypoint.d/40-cogniboiler-tls.sh
COPY --from=build /web/dist /usr/share/nginx/html
EXPOSE 8080 8443
