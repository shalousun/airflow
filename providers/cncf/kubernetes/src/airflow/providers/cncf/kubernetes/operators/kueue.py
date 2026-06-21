# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Manage a Kubernetes Kueue."""

from __future__ import annotations

import json
from collections.abc import Sequence
from functools import cached_property

from kubernetes.utils import FailToCreateError

from airflow.providers.cncf.kubernetes.hooks.kubernetes import KubernetesHook
from airflow.providers.cncf.kubernetes.operators.job import KubernetesJobOperator
from airflow.providers.cncf.kubernetes.version_compat import AIRFLOW_V_3_1_PLUS
from airflow.providers.common.compat.sdk import AirflowException

if AIRFLOW_V_3_1_PLUS:
    from airflow.sdk import BaseOperator
else:
    from airflow.models import BaseOperator


class KubernetesInstallKueueOperator(BaseOperator):
    """
    Installs a Kubernetes Kueue.

    .. seealso::
        For more information on how to use this operator, take a look at the guide:
        :ref:`howto/operator:KubernetesInstallKueueOperator`

    :param kueue_version: The Kubernetes Kueue version to install.
    :param kubernetes_conn_id: The :ref:`kubernetes connection id <howto/connection:kubernetes>`
        for the Kubernetes cluster.
    """

    template_fields: Sequence[str] = (
        "kueue_version",
        "kubernetes_conn_id",
    )

    def __init__(self, kueue_version: str, kubernetes_conn_id: str = "kubernetes_default", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kubernetes_conn_id = kubernetes_conn_id
        self.kueue_version = kueue_version
        self._kueue_yaml_url = (
            f"https://github.com/kubernetes-sigs/kueue/releases/download/{self.kueue_version}/manifests.yaml"
        )

    @cached_property
    def hook(self) -> KubernetesHook:
        return KubernetesHook(conn_id=self.kubernetes_conn_id)

    def execute(self, context):
        yaml_objects = self.hook.get_yaml_content_from_file(kueue_yaml_url=self._kueue_yaml_url)
        try:
            self.hook.apply_from_yaml_file(yaml_objects=yaml_objects)
        except FailToCreateError as ex:
            error_bodies = []
            for e in ex.api_exceptions:
                try:
                    if e.body:
                        error_bodies.append(json.loads(e.body))
                    else:
                        # If no body content, use reason as the message
                        reason = getattr(e, "reason", "Unknown")
                        error_bodies.append({"message": reason, "reason": reason})
                except (json.JSONDecodeError, ValueError, TypeError):
                    # If the body is a string (e.g., in a 429 error), it can't be parsed as JSON.
                    # Use the body directly as the message instead.
                    error_bodies.append({"message": e.body, "reason": getattr(e, "reason", "Unknown")})
            if next((e for e in error_bodies if e.get("reason") == "AlreadyExists"), None):
                self.log.info("Kueue is already enabled for the cluster")

            if errors := [e for e in error_bodies if e.get("reason") != "AlreadyExists"]:
                error_message = "\n".join(
                    e.get("message") or e.get("body") or f"Unknown error: {e.get('reason', 'Unknown')}"
                    for e in errors
                )
                raise AirflowException(error_message)
            return

        self.hook.check_kueue_deployment_running(name="kueue-controller-manager", namespace="kueue-system")
        self.log.info("Kueue installed successfully!")


class KubernetesStartKueueJobOperator(KubernetesJobOperator):
    """
    Executes a Kubernetes Job in Kueue.

    .. seealso::
        For more information on how to use this operator, take a look at the guide:
        :ref:`howto/operator:KubernetesStartKueueJobOperator`

    :param queue_name: The name of the Queue in the cluster
    """

    template_fields = tuple({"queue_name"} | set(KubernetesJobOperator.template_fields))

    def __init__(self, queue_name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.queue_name = queue_name

        self.suspend: bool
        if self.suspend is False:
            raise AirflowException(
                "The `suspend` parameter can't be False. If you want to use Kueue for running Job"
                " in a Kubernetes cluster, set the `suspend` parameter to True.",
            )
        if self.suspend is None:
            self.log.info(
                "You have not set parameter `suspend` in class %s. "
                "For running a Job in Kueue the `suspend` parameter has been set to True.",
                self.__class__.__name__,
            )
            self.suspend = True
        self.labels.update({"kueue.x-k8s.io/queue-name": self.queue_name})
        self.annotations.update({"kueue.x-k8s.io/queue-name": self.queue_name})

    def execute(self, context):
        import warnings

        from airflow.exceptions import AirflowProviderDeprecationWarning

        self.name = self._set_name(self.name)
        if self.parallelism is None:
            warnings.warn(
                "parallelism should be set explicitly. Defaulting to 1.",
                AirflowProviderDeprecationWarning,
                stacklevel=2,
            )
            self.parallelism = 1
        elif self.wait_until_job_complete and self.parallelism < 1:
            raise AirflowException("parallelism cannot be less than 1 with `wait_until_job_complete=True`.")
        self.job_request_obj = self.build_job_request_obj(context)
        self.job = self.create_job(job_request_obj=self.job_request_obj)

        ti = context["ti"]
        ti.xcom_push(key="job_name", value=self.job.metadata.name)
        ti.xcom_push(key="job_namespace", value=self.job.metadata.namespace)

        try:
            if not self.wait_until_job_complete:
                ti.xcom_push(key="job", value=self.job.to_dict())
                return

            if self.hook.is_job_suspended(job=self.job):
                self._defer_when_suspended(ti)

            return self._wait_for_job_completion(context)
        finally:
            self._cleanup_monitoring_pods(context)

    def _defer_when_suspended(self, ti):
        """Query the Kueue Workload for the suspension reason, log it, and defer."""
        job_name = self.job.metadata.name
        job_namespace = self.job.metadata.namespace

        reason = self.hook.get_kueue_workload_info(job_name=job_name, namespace=job_namespace)
        if reason:
            self.log.warning(
                "Job '%s' is suspended by Kueue: %s. Deferring poll for %s seconds.",
                job_name,
                reason,
                self.job_poll_interval,
            )
            ti.xcom_push(key="kueue_suspend_reason", value=reason)
        else:
            self.log.info(
                "Job '%s' is suspended by Kueue, deferring poll for %s seconds.",
                job_name,
                self.job_poll_interval,
            )

        from airflow.providers.cncf.kubernetes.triggers.job import (
            KueueSuspendTrigger,
        )

        self.defer(
            trigger=KueueSuspendTrigger(
                job_name=job_name,
                job_namespace=job_namespace,
                poll_interval=self.job_poll_interval,
            ),
            method_name="execute_complete",
        )

    def _wait_for_job_completion(self, context):
        """Discover pods (if any) and wait for the K8s Job to finish."""
        ti = context["ti"]
        ti.xcom_push(key="kueue_suspend_reason", value="")  # clear the suspend reason
        self.log.info(
            "Job '%s' has been admitted by Kueue, waiting for pods and completion.",
            self.job.metadata.name,
        )
        self.pods = self.get_pods(pod_request_obj=self.pod_request_obj, context=context)

        if self.do_xcom_push:
            xcom_result = []
            if self.pods:
                for pod in self.pods:
                    self.pod_manager.await_container_completion(
                        pod=pod, container_name=self.base_container_name
                    )
                    self.pod_manager.await_xcom_sidecar_container_start(pod=pod)
                    xcom_result.append(self.extract_xcom(pod=pod))

        self.job = self.hook.wait_until_job_complete(
            job_name=self.job.metadata.name,
            namespace=self.job.metadata.namespace,
            job_poll_interval=self.job_poll_interval,
        )

        if self.get_logs:
            if not self.hook.is_job_suspended(job=self.job) and self.pods:
                for pod in self.pods:
                    self.pod_manager.fetch_requested_container_logs(
                        pod=pod,
                        containers=self.container_logs,
                        follow_logs=True,
                    )

        ti.xcom_push(key="job", value=self.job.to_dict())

        if error_message := self.hook.is_job_failed(job=self.job):
            raise AirflowException(
                f"Kubernetes job '{self.job.metadata.name}' failed with error '{error_message}'"
            )

        if self.do_xcom_push:
            return xcom_result[0] if self.unwrap_single and len(xcom_result) == 1 else xcom_result

    def execute_complete(self, context, event):
        # Re-fetch job status using the sync hook in the worker process.
        # The trigger (KueueSuspendTrigger) does NOT query K8s — it only
        # sleeps, so the actual API call happens here in the worker.
        try:
            self.job = self.hook.get_job_status(
                job_name=self.job.metadata.name,
                namespace=self.job.metadata.namespace,
            )

            if self.hook.is_job_suspended(self.job):
                self._defer_when_suspended(context["ti"])

            return self._wait_for_job_completion(context)
        finally:
            self._cleanup_monitoring_pods(context)
