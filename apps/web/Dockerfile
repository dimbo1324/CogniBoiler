# The operator console: built with pnpm, then served by an unprivileged nginx that also
# proxies the gateway's public routes, so the browser talks to one origin.
# Built by Compose with the extra build context `nginx` = infrastructure/nginx.

# Base images are pinned by digest; rebuild on a new digest when the image scan asks for it.
FROM node:24-alpine@sha256:ebfe2f90462722a7a4de65e91990e97fe0d401c70e0e762c5b53302f905ec1c1 AS build
ENV COREPACK_ENABLE_DOWNLOAD_PROMPT=0
WORKDIR /web
RUN corepack enable
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY . .
RUN pnpm run build

FROM nginxinc/nginx-unprivileged:1.30-alpine@sha256:ed04ec1ff34502c339ee5c3ae3f855442398edc1d05591e2b98981dcbbd20b1e AS runtime
USER root
RUN apk upgrade --no-cache
USER 101
COPY --from=nginx default.conf /etc/nginx/conf.d/default.conf
COPY --from=nginx site.inc headers.inc proxy.inc /etc/nginx/cogniboiler/
COPY --from=nginx --chmod=0755 40-cogniboiler-tls.sh /docker-entrypoint.d/40-cogniboiler-tls.sh
COPY --from=build /web/dist /usr/share/nginx/html
EXPOSE 8080 8443
