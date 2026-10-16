#!/usr/bin/env bash
# One-time setup for GCS profile artifacts bucket and IAM bindings.
# Run this manually with appropriate gcloud credentials and kubectl pointed at
# your GKE cluster. Required environment:
#   PROJECT  GCP project ID
#   BUCKET   GCS bucket name for profile artifacts (pass the same value as GCS_BUCKET)
#   REGION   bucket location, e.g. us-central1
# Optional: GCP_SA, K8S_SA (default kernel-eval), K8S_NS (default "default").
set -euo pipefail

PROJECT="${PROJECT:?set PROJECT to your GCP project ID}"
BUCKET="${BUCKET:?set BUCKET to the profile artifact bucket name}"
REGION="${REGION:?set REGION to the bucket location, e.g. us-central1}"
GCP_SA="${GCP_SA:-kernel-eval}"
K8S_SA="${K8S_SA:-kernel-eval}"
K8S_NS="${K8S_NS:-default}"

echo "=== Creating GCS bucket ==="
gcloud storage buckets create "gs://${BUCKET}" \
  --project="${PROJECT}" \
  --location="${REGION}" \
  --uniform-bucket-level-access \
  2>/dev/null || echo "Bucket already exists"

echo "=== Setting lifecycle (7-day auto-delete) ==="
LIFECYCLE_FILE=$(mktemp)
echo '{"rule": [{"action": {"type": "Delete"}, "condition": {"age": 7}}]}' > "${LIFECYCLE_FILE}"
gcloud storage buckets update "gs://${BUCKET}" --lifecycle-file="${LIFECYCLE_FILE}"
rm -f "${LIFECYCLE_FILE}"

echo "=== Creating GCP service account ==="
gcloud iam service-accounts create "${GCP_SA}" \
  --project="${PROJECT}" \
  --display-name="Kernel Eval Pod SA" \
  2>/dev/null || echo "SA already exists"

echo "=== Granting GCS objectCreator to SA ==="
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${GCP_SA}@${PROJECT}.iam.gserviceaccount.com" \
  --role="roles/storage.objectCreator"

echo "=== Creating K8s service account ==="
kubectl create serviceaccount "${K8S_SA}" -n "${K8S_NS}" \
  2>/dev/null || echo "K8s SA already exists"

echo "=== Binding K8s SA to GCP SA via Workload Identity ==="
gcloud iam service-accounts add-iam-policy-binding \
  "${GCP_SA}@${PROJECT}.iam.gserviceaccount.com" \
  --role=roles/iam.workloadIdentityUser \
  --member="serviceAccount:${PROJECT}.svc.id.goog[${K8S_NS}/${K8S_SA}]"

echo "=== Annotating K8s SA ==="
kubectl annotate serviceaccount "${K8S_SA}" -n "${K8S_NS}" \
  "iam.gke.io/gcp-service-account=${GCP_SA}@${PROJECT}.iam.gserviceaccount.com" \
  --overwrite

echo "=== Done! Verify with: ==="
echo "kubectl get sa ${K8S_SA} -n ${K8S_NS} -o yaml"
