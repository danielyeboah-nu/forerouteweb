FROM node:22-alpine AS deps
WORKDIR /app/web
COPY web/package.json web/package-lock.json ./
RUN npm ci

FROM node:22-alpine AS builder
WORKDIR /app/web
ENV NEXT_TELEMETRY_DISABLED=1
ARG NEXT_PUBLIC_MAPBOX_TOKEN
ENV NEXT_PUBLIC_MAPBOX_TOKEN=${NEXT_PUBLIC_MAPBOX_TOKEN}
COPY --from=deps /app/web/node_modules ./node_modules
COPY web/ ./
RUN npm run build

FROM node:22-alpine AS runner
WORKDIR /app/web

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV HOSTNAME=0.0.0.0
ENV PORT=8080

RUN addgroup --system --gid 1001 nodejs \
  && adduser --system --uid 1001 nextjs

COPY --from=builder --chown=nextjs:nodejs /app/web/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/web/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/web/package.json ./package.json
COPY --chown=nextjs:nodejs ml/data ./ml/data

USER nextjs
EXPOSE 8080

CMD ["node", "server.js"]
