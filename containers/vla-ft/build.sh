#!/usr/bin/env bash
# Build + push the VLA-FT training container to the platform ECR repo.
#
# Absorbed from projects/vla-ft/build.sh and generalized for the platform: the
# default repo is now the SharedBaseStack repo `pai/vla-ft` (namespaced), not the
# flat `vla-ft`. The Dockerfile and entrypoint (docker/Dockerfile, src/train.py)
# are byte-identical to the verified smoke run — only the build *target* moved.
#
#
# Default path is **CodeBuild** (per the repo Docker policy in CLAUDE.md: prefer
# AWS-side builds over local docker). CodeBuild builds the linux/amd64 CUDA image
# on an x86 host — faster than emulating amd64 on an M-series Mac, and it never
# grows the local Docker.raw. `--local` keeps the old local-docker fallback.
#
# The Dockerfile has no COPY/ADD, so the build context is just the Dockerfile;
# CodeBuild source is a one-file zip uploaded to the account CDK assets bucket.
#
# Usage:
#   ./build.sh [--region us-west-2] [--repo pai/vla-ft] [--tag latest] [--local]
# Output: prints the pushed image URI AND its resolved @sha256 digest.
#
# Reproducibility (why the digest matters): `:latest` / `:efa` are MUTABLE — a later
# build re-points them at new content, so an in-flight job that referenced the tag can
# silently switch images (a 2026-06-25 ':latest' rebuild grew the GPU-memory baseline
# and OOM'd a job that had fit on the prior image by <1 MiB). To make every build
# pin-able, this script ALSO pushes an immutable provenance tag
# (<variant>-<YYYYMMDD>-<gitsha>[-dirty]) next to the mutable tag and prints the
# resolved digest. Pin a run to the digest (drift-proof): launch.py --image-uri ...@sha256:...
set -euo pipefail

REGION="${AWS_REGION:-us-west-2}"
REPO="pai/vla-ft"    # SharedBaseStack ECR repo (namePrefix 'pai' + '/vla-ft')
TAG="latest"
MODE="codebuild"   # codebuild (default) | local
ENABLE_EFA=0       # 0 = single-node image (default); 1 = EFA/NCCL fabric overlay for Pattern C

while [ $# -gt 0 ]; do
  case "$1" in
    --region) REGION="$2"; shift 2;;
    --repo)   REPO="$2";   shift 2;;
    --tag)    TAG="$2";    shift 2;;
    --local)  MODE="local"; shift;;
    --codebuild) MODE="codebuild"; shift;;
    # --efa builds the multi-node fabric image (EFA installer 1.47.0 + GDRCopy 2.5.1 +
    # aws-ofi-nccl, gated by the Dockerfile's ENABLE_EFA ARG). Default tag flips to ':efa'
    # so the verified single-node ':latest' image is never overwritten by a fabric build.
    --efa)    ENABLE_EFA=1; [ "$TAG" = "latest" ] && TAG="efa"; shift;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
ECR="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
URI="${ECR}/${REPO}:${TAG}"
# REPO may be namespaced (pai/vla-ft). ECR repo names allow '/', but IAM role and
# CodeBuild project names do not — derive a slash-free slug for those.
REPO_SLUG="${REPO//\//-}"   # pai/vla-ft -> pai-vla-ft

# Immutable provenance tag pushed ALONGSIDE the mutable one so any build is pin-able by
# a stable name even before you read the digest. Form: <variant>-<YYYYMMDD>-<gitsha>[-dirty].
# variant = the mutable tag (latest/efa) so ':latest' and ':efa' lines stay distinguishable.
# gitsha is the build.sh repo HEAD (the image's only source is docker/Dockerfile, tracked
# here); '-dirty' marks an uncommitted Dockerfile so a pinned tag never claims a clean commit.
GIT_SHA="$(git -C "$SCRIPT_DIR" rev-parse --short=12 HEAD 2>/dev/null || echo nogit)"
if ! git -C "$SCRIPT_DIR" diff --quiet -- docker/Dockerfile 2>/dev/null; then GIT_SHA="${GIT_SHA}-dirty"; fi
BUILD_DATE="$(date -u +%Y%m%d)"
IMMUTABLE_TAG="${TAG}-${BUILD_DATE}-${GIT_SHA}"
IMMUTABLE_URI="${ECR}/${REPO}:${IMMUTABLE_TAG}"

echo "Account: ${ACCOUNT}  Region: ${REGION}  Mode: ${MODE}"
echo "Image:   ${URI}"
echo "Pin tag: ${IMMUTABLE_URI}   (immutable; pushed alongside :${TAG})"

# ECR repo: the platform's SharedBaseStack normally creates pai/vla-ft (RETAIN,
# scan-on-push). Create-if-missing here too so build.sh works before/without the
# stack (e.g. iterating on the image), idempotent if the stack already made it.
aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO" --region "$REGION" >/dev/null

# Resolve the pushed image's content digest from ECR (by the immutable tag, which the
# build just put on the same manifest) and print the pin-ready summary. ECR is the source
# of truth for the digest regardless of where the build ran (local or CodeBuild), so both
# modes call this. A digest is drift-proof: ':latest' can move, ...@sha256:<digest> cannot.
print_pushed() {
  local how="$1"
  local digest
  digest="$(aws ecr describe-images --repository-name "$REPO" --region "$REGION" \
    --image-ids imageTag="$IMMUTABLE_TAG" \
    --query 'imageDetails[0].imageDigest' --output text 2>/dev/null || true)"
  echo ""
  echo "Pushed (${how}):"
  echo "  mutable   : ${URI}"
  echo "  immutable : ${IMMUTABLE_URI}"
  if [ -n "$digest" ] && [ "$digest" != "None" ]; then
    echo "  digest    : ${ECR}/${REPO}@${digest}"
    echo ""
    echo "Pin a run to the exact image (drift-proof) with the digest:"
    echo "  python launch.py --image-uri ${ECR}/${REPO}@${digest} --policy pi05 --dataset-s3 s3://... --pretrained-path lerobot/pi05_base"
  else
    echo "  digest    : (could not resolve from ECR — pin by the immutable tag above)"
  fi
}

# ───────────────────────────── local fallback ─────────────────────────────
if [ "$MODE" = "local" ]; then
  aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$ECR"
  # --platform linux/amd64 so the image runs on SageMaker GPU hosts even when built on arm64 (M-series).
  # Tag the SAME build with both the mutable handle and the immutable provenance tag, then
  # push both (same manifest → same digest, two names pointing at it).
  docker build --platform linux/amd64 --build-arg ENABLE_EFA="${ENABLE_EFA}" \
    -t "$URI" -t "$IMMUTABLE_URI" -f "${SCRIPT_DIR}/docker/Dockerfile" "${SCRIPT_DIR}/docker"
  docker push "$URI"
  docker push "$IMMUTABLE_URI"
  print_pushed "local"
  exit 0
fi

# ─────────────────────────────── CodeBuild ────────────────────────────────
PROJECT="${REPO_SLUG}-build"
ROLE_NAME="${REPO_SLUG}-codebuild-role"
ROLE_ARN="arn:aws:iam::${ACCOUNT}:role/${ROLE_NAME}"
SRC_BUCKET="cdk-hnb659fds-assets-${ACCOUNT}-${REGION}"   # account CDK assets bucket (reused)
SRC_KEY="${REPO_SLUG}-build/source.zip"

# 1. Scoped CodeBuild service role (create-or-reuse). ECR push to this repo only,
#    GetAuthorizationToken (*-only action), source-bucket read, build logs.
if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  echo "Creating IAM role ${ROLE_NAME} ..."
  aws iam create-role --role-name "$ROLE_NAME" \
    --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{"Effect":"Allow","Principal":{"Service":"codebuild.amazonaws.com"},"Action":"sts:AssumeRole"}]
    }' >/dev/null
fi
aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name "${REPO_SLUG}-build-policy" \
  --policy-document "{
    \"Version\":\"2012-10-17\",
    \"Statement\":[
      {\"Effect\":\"Allow\",
       \"Action\":[\"ecr:BatchCheckLayerAvailability\",\"ecr:BatchGetImage\",\"ecr:CompleteLayerUpload\",\"ecr:GetDownloadUrlForLayer\",\"ecr:InitiateLayerUpload\",\"ecr:PutImage\",\"ecr:UploadLayerPart\"],
       \"Resource\":\"arn:aws:ecr:${REGION}:${ACCOUNT}:repository/${REPO}\"},
      {\"Effect\":\"Allow\",\"Action\":\"ecr:GetAuthorizationToken\",\"Resource\":\"*\"},
      {\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:GetObjectVersion\"],\"Resource\":\"arn:aws:s3:::${SRC_BUCKET}/${REPO_SLUG}-build/*\"},
      {\"Effect\":\"Allow\",
       \"Action\":[\"logs:CreateLogGroup\",\"logs:CreateLogStream\",\"logs:PutLogEvents\"],
       \"Resource\":[\"arn:aws:logs:${REGION}:${ACCOUNT}:log-group:/aws/codebuild/${PROJECT}\",\"arn:aws:logs:${REGION}:${ACCOUNT}:log-group:/aws/codebuild/${PROJECT}:*\"]}
    ]
  }" >/dev/null

# 2. Package source (Dockerfile only — no COPY in it) and upload to the CDK bucket.
TMP_ZIP="$(mktemp -d)/source.zip"
( cd "${SCRIPT_DIR}/docker" && zip -q "$TMP_ZIP" Dockerfile )
aws s3 cp "$TMP_ZIP" "s3://${SRC_BUCKET}/${SRC_KEY}" --region "$REGION" >/dev/null
echo "Source uploaded: s3://${SRC_BUCKET}/${SRC_KEY}"

# 3. Buildspec: ECR login -> buildkit build (mutable + immutable tag on one manifest) -> push both.
BUILDSPEC="{
  \"version\":\"0.2\",
  \"phases\":{
    \"pre_build\":{\"commands\":[\"aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR}\"]},
    \"build\":{\"commands\":[\"DOCKER_BUILDKIT=1 docker build --build-arg ENABLE_EFA=${ENABLE_EFA} -t ${URI} -t ${IMMUTABLE_URI} .\"]},
    \"post_build\":{\"commands\":[\"docker push ${URI}\",\"docker push ${IMMUTABLE_URI}\"]}
  }
}"

# 4. CodeBuild project (create-or-update). LARGE x86 + privileged for docker.
SOURCE_JSON="{\"type\":\"S3\",\"location\":\"${SRC_BUCKET}/${SRC_KEY}\",\"buildspec\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$BUILDSPEC")}"
ENV_JSON='{"type":"LINUX_CONTAINER","image":"aws/codebuild/standard:7.0","computeType":"BUILD_GENERAL1_LARGE","privilegedMode":true,"imagePullCredentialsType":"CODEBUILD"}'

if aws codebuild batch-get-projects --names "$PROJECT" --region "$REGION" \
     --query 'projects[0].name' --output text 2>/dev/null | grep -q "$PROJECT"; then
  aws codebuild update-project --name "$PROJECT" --region "$REGION" \
    --source "$SOURCE_JSON" --environment "$ENV_JSON" \
    --service-role "$ROLE_ARN" --artifacts '{"type":"NO_ARTIFACTS"}' \
    --timeout-in-minutes 60 >/dev/null
else
  # IAM role is eventually consistent; retry create until the role is assumable.
  for i in 1 2 3 4 5 6; do
    if aws codebuild create-project --name "$PROJECT" --region "$REGION" \
         --source "$SOURCE_JSON" --environment "$ENV_JSON" \
         --service-role "$ROLE_ARN" --artifacts '{"type":"NO_ARTIFACTS"}' \
         --timeout-in-minutes 60 >/dev/null 2>&1; then break; fi
    echo "  (waiting for IAM role to propagate, attempt ${i}) ..."; sleep 10
  done
fi

# 5. Start the build and poll to completion (streaming phase status).
BUILD_ID="$(aws codebuild start-build --project-name "$PROJECT" --region "$REGION" \
  --query 'build.id' --output text)"
echo "Started CodeBuild: ${BUILD_ID}"
echo "  Logs: https://${REGION}.console.aws.amazon.com/codesuite/codebuild/${ACCOUNT}/projects/${PROJECT}/build/${BUILD_ID//:/%3A}"

STATUS="IN_PROGRESS"; PHASE=""
while [ "$STATUS" = "IN_PROGRESS" ]; do
  sleep 15
  read -r STATUS PHASE < <(aws codebuild batch-get-builds --ids "$BUILD_ID" --region "$REGION" \
    --query 'builds[0].[buildStatus,currentPhase]' --output text)
  echo "  status=${STATUS}  phase=${PHASE}"
done

if [ "$STATUS" != "SUCCEEDED" ]; then
  echo "" >&2
  echo "CodeBuild ${STATUS}. Check logs (link above) or:" >&2
  echo "  aws codebuild batch-get-builds --ids ${BUILD_ID} --region ${REGION}" >&2
  exit 1
fi

print_pushed "CodeBuild"
