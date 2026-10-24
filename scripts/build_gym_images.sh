#!/usr/bin/env bash
# Build the three Docker images the gym environment uses. Safe to re-run.
#
#   evcarry/toolathlon-gym-base:<commit>  the gym's own Dockerfile, unchanged
#   evcarry/toolathlon-gym:<commit>       + docker/gym/overlay.Dockerfile (MCP servers upstream leaves unbuilt)
#   evcarry/toolathlon-gym-pg:<commit>    postgres:15 with db/init.sql.gz loaded and the gym's
#                                         email.sent_log FK fix applied, committed as a snapshot
#
# Every run starts its own containers from these images (see src/evcarry/env/gym_docker.py),
# so the dump is loaded once here instead of once per task.
set -euo pipefail
cd "$(dirname "$0")/.."

GYM=third_party/toolathlon_gym
COMMIT=$(git -C "$GYM" rev-parse --short=7 HEAD)
LABEL=evcarry.project=evidence-subagents
BASE=evcarry/toolathlon-gym-base:$COMMIT
AGENT=evcarry/toolathlon-gym:$COMMIT
PG=evcarry/toolathlon-gym-pg:$COMMIT

docker build -t "$BASE" --label "$LABEL" --label "evcarry.gym_commit=$COMMIT" "$GYM"
docker build -f docker/gym/overlay.Dockerfile --build-arg "BASE_IMAGE=$BASE" \
    -t "$AGENT" --label "$LABEL" --label "evcarry.gym_commit=$COMMIT" docker/gym

docker pull -q postgres:15 >/dev/null
PG_DIGEST=$(docker image inspect postgres:15 --format '{{index .RepoDigests 0}}')
SEED="evcarry-pg-seed-$$"
trap 'docker rm -fv "$SEED" >/dev/null 2>&1 || true' EXIT
# PGDATA sits outside the image's declared volume so `docker commit` keeps the data.
docker run -d --name "$SEED" --label "$LABEL" \
    -e POSTGRES_DB=toolathlon_gym -e POSTGRES_USER=eigent -e POSTGRES_PASSWORD=camel -e PGDATA=/pgdata \
    -v "$PWD/$GYM/db/init.sql.gz:/docker-entrypoint-initdb.d/init.sql.gz:ro" \
    postgres:15 >/dev/null
# During init the entrypoint's temporary server listens on the unix socket only,
# so readiness is checked over TCP.
for _ in $(seq 1 300); do
    if docker exec "$SEED" pg_isready -h 127.0.0.1 -U eigent -d toolathlon_gym -q 2>/dev/null; then break; fi
    sleep 1
done
docker exec "$SEED" pg_isready -h 127.0.0.1 -U eigent -d toolathlon_gym
# Same fix run_parallel.sh applies before every task: init.sql.gz lacks ON DELETE CASCADE.
docker exec "$SEED" psql -U eigent -d toolathlon_gym -v ON_ERROR_STOP=1 -q -c \
    "ALTER TABLE email.sent_log DROP CONSTRAINT sent_log_message_id_fkey;
     ALTER TABLE email.sent_log ADD CONSTRAINT sent_log_message_id_fkey
       FOREIGN KEY (message_id) REFERENCES email.messages(id) ON DELETE CASCADE;"
docker stop "$SEED" >/dev/null
docker commit \
    --change "LABEL $LABEL" --change "LABEL evcarry.gym_commit=$COMMIT" --change "LABEL evcarry.pg_base=$PG_DIGEST" \
    "$SEED" "$PG" >/dev/null

docker image ls --filter "label=$LABEL" --format '{{.Repository}}:{{.Tag}}  {{.Size}}'
