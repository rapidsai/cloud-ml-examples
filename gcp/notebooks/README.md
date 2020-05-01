## GCP AI Platform Custom Training Container Builds
#### Reference Content: https://cloud.google.com/ai-platform/training/docs
#### Note: At this time, GCP's AI Platform notebooks do not appear to support 'docker-in-docker' functionality, so build steps should be run on a VM or local machine.

1. Ensure that your build machine has docker installed and configured
    1. https://docs.docker.com/get-started/
1. Ensure that Google's 'gcloud' sdk is installed and configured
    1. 