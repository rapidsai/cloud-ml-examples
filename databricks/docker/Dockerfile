ARG RAPIDS_IMAGE
FROM $RAPIDS_IMAGE as rapids

RUN conda list -n rapids --explicit > /rapids/rapids-spec.txt

FROM databricksruntime/gpu-conda:cuda11

COPY --from=rapids /rapids/rapids-spec.txt /tmp/spec.txt

RUN conda create --name rapids --file /tmp/spec.txt && \
    rm -f /tmp/spec.txt

# Set an environment variable used by Databricks to decide which conda environment to activate by default.
ENV DEFAULT_DATABRICKS_ROOT_CONDA_ENV=rapids
