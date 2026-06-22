#!/usr/bin/env bash
# Build + push the Isaac Lab RL training container to the platform ECR repo.
#
# RL-axis counterpart of containers/vla-ft/build.sh. Default repo is the
# SharedBaseStack repo `pai/isaac-lab-rl`. The Dockerfile clones IsaacLab @ v2.3.2 on
# nvcr.io/nvidia/isaac-sim:4.5.0 and runs the official installer — a heavy build
# (multi-GB base + full RL framework install), so the CodeBuild timeout is generous.
#
# Default path is **CodeBuild** (per the repo Docker policy in CLAUDE.md: prefer
# AWS-side builds over local docker — the isaac-sim base alone is ~20 GB, which would
# balloon a local Docker.raw). `--local` keeps a local-docker fallback.
#
# The Dockerfile has no COPY/ADD (the RL entrypoint is injected at submit time via
# `python3 -c`, not baked in), so the build context is just the Dockerfile.
#
# NOTE: nvcr.io/nvidia/isaac-sim:4.5.0 may require NGC pull credentials. If the build
# fails at the FROM pull with an auth error, configure the CodeBuild project / local
# docker with an NGC API key (docker login nvcr.io -u '$oauthtoken'). Public re-pulls
# of isaac-sim are generally allowed but NGC has rate/auth gates that change.
#
# Usage:
#   ./build.sh [--region us-west-2] [--repo pai/isaac-lab-rl] [--tag latest] [--local]
# Output: prints the pushed image URI (feed to rl_launch.py via the stack image).
set -euo pipefail

REGION="${AWS_REGION:-us-west-2}"
REPO="pai/isaac-lab-rl"    # SharedBaseStack ECR repo (namePrefix 'pai' + '/isaac-lab-rl')
TAG="latest"
MODE="codebuild"   # codebuild (default) | local

while [ $# -gt 0 ]; do
  case "$1" in
    --region) REGION="$2"; shift 2;;
    --repo)   REPO="$2";   shift 2;;
    --tag)    TAG="$2";    shift 2;;
    --local)  MODE="local"; shift;;
    --codebuild) MODE="codebuild"; shift;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
ECR="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
URI="${ECR}/${REPO}:${TAG}"
# REPO may be namespaced (pai/isaac-lab-rl). ECR repo names allow '/', but IAM role
# and CodeBuild project names do not — derive a slash-free slug for those.
REPO_SLUG="${REPO//\//-}"   # pai/isaac-lab-rl -> pai-isaac-lab-rl

echo "Account: ${ACCOUNT}  Region: ${REGION}  Mode: ${MODE}"
echo "Image:   ${URI}"

# ECR repo: the platform's SharedBaseStack normally creates pai/isaac-lab-rl (RETAIN,
# scan-on-push). Create-if-missing here too so build.sh works before/without the stack.
aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO" --region "$REGION" >/dev/null

# ───────────────────────────── local fallback ─────────────────────────────
if [ "$MODE" = "local" ]; then
  aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$ECR"
  # --platform linux/amd64 so the CUDA image runs on GPU hosts even when built on arm64.
  docker build --platform linux/amd64 -t "$URI" -f "${SCRIPT_DIR}/docker/Dockerfile" "${SCRIPT_DIR}/docker"
  docker push "$URI"
  echo ""
  echo "Pushed (local): ${URI}"
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

# 3. Buildspec: ECR login -> buildkit build -> push.
BUILDSPEC="{
  \"version\":\"0.2\",
  \"phases\":{
    \"pre_build\":{\"commands\":[\"aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR}\"]},
    \"build\":{\"commands\":[\"DOCKER_BUILDKIT=1 docker build -t ${URI} .\"]},
    \"post_build\":{\"commands\":[\"docker push ${URI}\"]}
  }
}"

# 4. CodeBuild project (create-or-update). LARGE x86 + privileged for docker. The
#    isaac-sim base is large; a 250 GB build volume + 2h timeout keeps headroom.
SOURCE_JSON="{\"type\":\"S3\",\"location\":\"${SRC_BUCKET}/${SRC_KEY}\",\"buildspec\":$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$BUILDSPEC")}"
ENV_JSON='{"type":"LINUX_CONTAINER","image":"aws/codebuild/standard:7.0","computeType":"BUILD_GENERAL1_LARGE","privilegedMode":true,"imagePullCredentialsType":"CODEBUILD"}'

if aws codebuild batch-get-projects --names "$PROJECT" --region "$REGION" \
     --query 'projects[0].name' --output text 2>/dev/null | grep -q "$PROJECT"; then
  aws codebuild update-project --name "$PROJECT" --region "$REGION" \
    --source "$SOURCE_JSON" --environment "$ENV_JSON" \
    --service-role "$ROLE_ARN" --artifacts '{"type":"NO_ARTIFACTS"}' \
    --timeout-in-minutes 120 >/dev/null
else
  # IAM role is eventually consistent; retry create until the role is assumable.
  for i in 1 2 3 4 5 6; do
    if aws codebuild create-project --name "$PROJECT" --region "$REGION" \
         --source "$SOURCE_JSON" --environment "$ENV_JSON" \
         --service-role "$ROLE_ARN" --artifacts '{"type":"NO_ARTIFACTS"}' \
         --timeout-in-minutes 120 >/dev/null 2>&1; then break; fi
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

echo ""
echo "Pushed (CodeBuild): ${URI}"
echo "Next: python rl_launch.py --task Isaac-Velocity-Rough-H1-v0 --job-queue ... --job-definition ... --output-s3 ..."
