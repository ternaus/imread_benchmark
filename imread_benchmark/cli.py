from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import typer

from imread_benchmark.analysis.publication import publish
from imread_benchmark.artifacts import hydrate_committed_runs, validate_run_bundle
from imread_benchmark.datasets.fodb import prepare_fodb
from imread_benchmark.datasets.gcs import GcloudObjectStore
from imread_benchmark.datasets.materializer import materialize_dataset_package, publish_dataset_package
from imread_benchmark.datasets.package import build_dataset_package
from imread_benchmark.decoders import REGISTRY
from imread_benchmark.decoders.capabilities import describe_decoder
from imread_benchmark.environments import EnvironmentRequest, provision_environment
from imread_benchmark.environments.cache import materialize_environment_cache, publish_environment_cache
from imread_benchmark.execution.campaign import CampaignConfig, run_campaign
from imread_benchmark.execution.coordinator import RemoteCheckpoint
from imread_benchmark.plans import expand_experiment_plan, instantiate_experiment_plans, load_experiment_plan
from imread_benchmark.platforms import capture_current_platform, write_platform_descriptor

app = typer.Typer(add_completion=False, help="Reproducible JPEG decoder benchmark campaigns.")
dataset_app = typer.Typer(help="Build, publish, and materialize immutable dataset packages.")
plan_app = typer.Typer(help="Instantiate, validate, and expand schema-2 experiment plans.")
environment_app = typer.Typer(help="Provision immutable lock-backed worker environments.")
platform_app = typer.Typer(help="Capture platform provenance.")
artifacts_app = typer.Typer(help="Hydrate and validate committed run bundles.")
campaign_app = typer.Typer(help="Run or resume an isolated benchmark campaign.")
app.add_typer(dataset_app, name="dataset")
app.add_typer(plan_app, name="plan")
app.add_typer(environment_app, name="environment")
app.add_typer(platform_app, name="platform")
app.add_typer(artifacts_app, name="artifacts")
app.add_typer(campaign_app, name="campaign")


@app.command("list-decoders")
def list_decoders() -> None:
    """Print the registered decoder capability contracts as JSON."""
    _emit_json([describe_decoder(REGISTRY[name]).to_dict() for name in sorted(REGISTRY)])


@dataset_app.command("package")
def dataset_package(
    package_name: str = typer.Option(..., "--name", help="Stable logical package name."),
    workload: list[str] = typer.Option(
        ...,
        "--workload",
        help="Repeat NAME=JPEG_DIRECTORY for every workload view.",
    ),
    output_root: Path = typer.Option(..., "--output-root", help="Content-addressed package cache."),
    provenance_source: str = typer.Option(..., "--provenance-source"),
) -> None:
    """Build one deterministic uncompressed-tar dataset package."""
    workloads = _parse_workloads(workload)
    descriptor = build_dataset_package(
        package_name=package_name,
        workloads=workloads,
        output_root=output_root,
        provenance={"source": provenance_source},
    )
    _emit_json({"descriptor": str(descriptor), "schema_version": "2.0"})


@dataset_app.command("publish")
def dataset_publish(
    descriptor: Path = typer.Argument(..., exists=True, dir_okay=False),
    store_uri: str = typer.Option(..., "--store", help="GCS base URI, for example gs://bucket/benchmark."),
    prefix: str = typer.Option("datasets", "--prefix"),
) -> None:
    """Create-only publish every package component to GCS."""
    remote_descriptor = publish_dataset_package(
        descriptor,
        store=GcloudObjectStore(store_uri),
        prefix=prefix,
    )
    _emit_json({"remote_descriptor": remote_descriptor, "schema_version": "2.0", "store": store_uri})


@dataset_app.command("fodb-package")
def dataset_fodb_package(
    archive: list[Path] = typer.Option(..., "--archive", exists=True, dir_okay=False),
    output_root: Path = typer.Option(..., "--output-root", file_okay=False),
    scene_count: int = typer.Option(12, "--scene-count", min=1),
    seed: int = typer.Option(20260729, "--seed"),
    compressed_byte_limit: int = typer.Option(2 * 1024**3, "--compressed-byte-limit", min=1),
) -> None:
    """Build canonical native/mixed FODB workloads directly from downloaded ZIP parts."""
    descriptor = prepare_fodb(
        archive,
        output_root,
        scene_count=scene_count,
        seed=seed,
        compressed_byte_limit=compressed_byte_limit,
    )
    _emit_json({"descriptor": str(descriptor), "schema_version": "2.0"})


@dataset_app.command("controlled-package")
def dataset_controlled_package(
    source_dir: Path = typer.Option(..., "--source-dir", exists=True, file_okay=False),
    output_root: Path = typer.Option(..., "--output-root", file_okay=False),
    source_name: str = typer.Option(..., "--source-name"),
    source_release: str = typer.Option(..., "--source-release"),
    source_license: str = typer.Option(..., "--source-license"),
    source_url: str | None = typer.Option(None, "--source-url"),
    long_edge: list[int] = typer.Option([512, 1024, 2048], "--long-edge", min=1),
    quality: list[int] = typer.Option([50, 75, 90, 95], "--quality", min=1, max=95),
    include_native: bool = typer.Option(True, "--include-native/--no-native"),
    subsampling: str = typer.Option("4:2:0", "--subsampling"),
    seed: int = typer.Option(20260729, "--seed"),
    compressed_byte_limit: int = typer.Option(2 * 1024**3, "--compressed-byte-limit", min=1),
) -> None:
    """Build matched resolution-by-quality workloads from pinned lossless PNGs."""
    from imread_benchmark.datasets.controlled import prepare_controlled_ablation

    descriptor = prepare_controlled_ablation(
        source_dir,
        output_root,
        source_name=source_name,
        source_release=source_release,
        source_license=source_license,
        source_url=source_url,
        long_edges=long_edge,
        qualities=quality,
        include_native=include_native,
        subsampling=subsampling,
        seed=seed,
        compressed_byte_limit=compressed_byte_limit,
    )
    _emit_json({"descriptor": str(descriptor), "schema_version": "2.0"})


@dataset_app.command("materialize")
def dataset_materialize(
    remote_descriptor: str = typer.Argument(..., help="Object key returned by dataset publish."),
    store_uri: str = typer.Option(..., "--store", help="GCS base URI used for publication."),
    cache_root: Path = typer.Option(..., "--cache-root"),
) -> None:
    """Download, fully verify, and atomically publish a local package cache entry."""
    descriptor = materialize_dataset_package(
        remote_descriptor,
        store=GcloudObjectStore(store_uri),
        cache_root=cache_root,
    )
    _emit_json({"descriptor": str(descriptor), "schema_version": "2.0"})


@plan_app.command("instantiate")
def plan_instantiate(
    template_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    package_descriptor: Path = typer.Option(..., "--package-descriptor", exists=True, dir_okay=False),
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False),
    workload: list[str] = typer.Option([], "--workload"),
) -> None:
    """Fill and validate one plan per selected package workload."""
    plans = instantiate_experiment_plans(
        template_path=template_path,
        package_descriptor=package_descriptor,
        output_dir=output_dir,
        workload_ids=tuple(workload),
    )
    _emit_json({"plans": [plan.to_dict() for plan in plans], "schema_version": "2.0"})


@plan_app.command("validate")
def plan_validate(
    plan_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    package_descriptor: Path | None = typer.Option(None, "--package-descriptor", dir_okay=False),
) -> None:
    """Validate plan semantics and its exact pinned dataset identity."""
    plan = load_experiment_plan(plan_path, dataset_descriptor=package_descriptor)
    templates = expand_experiment_plan(plan)
    _emit_json(
        {
            "configuration_count": len({template.configuration.config_id for template in templates}),
            "plan_id": templates[0].plan_id,
            "run_count_per_platform": len(templates),
            "schema_version": "2.0",
        },
    )


@plan_app.command("expand")
def plan_expand(
    plan_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", dir_okay=False),
    package_descriptor: Path | None = typer.Option(None, "--package-descriptor", dir_okay=False),
) -> None:
    """Write the deterministic randomized run-template matrix."""
    plan = load_experiment_plan(plan_path, dataset_descriptor=package_descriptor)
    templates = expand_experiment_plan(plan)
    document = {
        "plan_id": templates[0].plan_id,
        "runs": [
            {
                "block_position": template.position,
                "config_id": template.configuration.config_id,
                "configuration": asdict(template.configuration),
                "repetition": template.repetition,
                "template_id": template.template_id,
            }
            for template in templates
        ],
        "schema_version": "2.0",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _emit_json({"output": str(output.resolve()), "plan_id": templates[0].plan_id, "schema_version": "2.0"})


@environment_app.command("provision")
def environment_provision(
    dependency_group: str = typer.Option(..., "--group"),
    runner_revision: str = typer.Option(..., "--runner-revision"),
    project_root: Path = typer.Option(Path(), "--project-root", file_okay=False),
    cache_root: Path = typer.Option(..., "--cache-root", file_okay=False),
    python_executable: Path = typer.Option(Path(sys.executable), "--python", dir_okay=False),
    native_backend: list[str] = typer.Option([], "--native-backend", help="Repeat NAME=VERSION for native libraries."),
    remote_store_uri: str | None = typer.Option(None, "--remote-store"),
    remote_prefix: str = typer.Option("environments", "--remote-prefix"),
) -> None:
    """Provision or reuse an atomic `uv sync --frozen --no-editable` environment."""
    request = EnvironmentRequest(
        project_root=project_root,
        cache_root=cache_root,
        dependency_group=dependency_group,
        runner_revision=runner_revision,
        python_executable=python_executable,
        native_backends=_parse_name_values(native_backend, option="--native-backend"),
    )
    store = GcloudObjectStore(remote_store_uri) if remote_store_uri is not None else None
    result = materialize_environment_cache(request, store=store, prefix=remote_prefix) if store is not None else None
    if result is None:
        result = provision_environment(request)
        if store is not None:
            publish_environment_cache(result.root, store=store, prefix=remote_prefix)
    _emit_json(
        {
            "cache_hit": result.cache_hit,
            "descriptor": str(result.descriptor_path),
            "environment_id": result.environment_id,
            "environment_key": result.environment_key,
            "python": str(result.python_executable),
            "root": str(result.root),
            "schema_version": "2.0",
        },
    )


@platform_app.command("capture")
def platform_capture(
    output: Path = typer.Option(..., "--output", dir_okay=False),
    cloud_provider: str = typer.Option("local", "--cloud-provider"),
    machine_type: str = typer.Option(..., "--machine-type"),
    location: str = typer.Option(..., "--location"),
) -> None:
    """Capture stable machine identity and separate dynamic runtime metadata."""
    descriptor = capture_current_platform(
        cloud_provider=cloud_provider,
        machine_type=machine_type,
        location=location,
    )
    path = write_platform_descriptor(output, descriptor)
    _emit_json({"descriptor": str(path.resolve()), "platform_id": descriptor.platform_id, "schema_version": "2.0"})


@campaign_app.command("run")
def campaign_run(
    plan_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    package_descriptor: Path = typer.Option(..., "--package-descriptor", exists=True, dir_okay=False),
    environment_descriptor: Path = typer.Option(..., "--environment-descriptor", exists=True, dir_okay=False),
    platform_descriptor: Path = typer.Option(..., "--platform-descriptor", exists=True, dir_okay=False),
    artifact_root: Path = typer.Option(..., "--artifact-root", file_okay=False),
    attempts_root: Path = typer.Option(..., "--attempts-root", file_okay=False),
    runner_revision: str = typer.Option(..., "--runner-revision"),
    remote_store_uri: str | None = typer.Option(None, "--remote-store"),
    worker_python: Path = typer.Option(Path(sys.executable), "--worker-python", dir_okay=False),
) -> None:
    """Run support audits and missing run specs, checkpointing each bundle."""
    remote = RemoteCheckpoint(GcloudObjectStore(remote_store_uri)) if remote_store_uri is not None else None
    result = run_campaign(
        CampaignConfig(
            plan_path=plan_path,
            package_descriptor=package_descriptor,
            environment_descriptor=environment_descriptor,
            platform_descriptor=platform_descriptor,
            artifact_root=artifact_root,
            attempts_root=attempts_root,
            runner_revision=runner_revision,
            worker_python=worker_python,
            remote=remote,
        ),
    )
    _emit_json(result.to_dict())
    if not result.complete:
        raise typer.Exit(1)


@artifacts_app.command("validate")
def artifacts_validate(
    artifact_root: Path = typer.Argument(..., exists=True, file_okay=False),
) -> None:
    """Validate every committed run bundle below ARTIFACT_ROOT/runs."""
    run_root = artifact_root / "runs"
    bundles = tuple(sorted(path for path in run_root.iterdir() if path.is_dir())) if run_root.is_dir() else ()
    if not bundles:
        raise typer.BadParameter("artifact root contains no run bundles")
    for bundle in bundles:
        validate_run_bundle(bundle, expected_run_key=bundle.name)
    _emit_json({"bundle_count": len(bundles), "schema_version": "2.0", "status": "valid"})


@artifacts_app.command("hydrate")
def artifacts_hydrate(
    source_artifact_root: Path = typer.Argument(..., exists=True, file_okay=False),
    output_root: Path = typer.Option(..., "--output-root", file_okay=False),
) -> None:
    """Rebuild local run bundles from a downloaded remote artifact layout."""
    bundles = hydrate_committed_runs(
        source_artifact_root=source_artifact_root,
        destination_artifact_root=output_root,
    )
    _emit_json(
        {
            "bundle_count": len(bundles),
            "output_root": str(output_root.resolve()),
            "schema_version": "2.0",
        },
    )


@app.command("publish")
def publication_publish(
    spec_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    artifact_root: Path = typer.Option(..., "--artifact-root", exists=True, file_okay=False),
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False),
    check: bool = typer.Option(False, "--check"),
) -> None:
    """Generate or verify deterministic claim-scoped publication artifacts."""
    publish(artifact_root=artifact_root, spec_path=spec_path, output_dir=output_dir, check=check)
    _emit_json({"check": check, "output_dir": str(output_dir.resolve()), "schema_version": "2.0"})


def _parse_workloads(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise typer.BadParameter("each --workload must be NAME=JPEG_DIRECTORY")
        if name in result:
            raise typer.BadParameter(f"duplicate workload name: {name}")
        path = Path(raw_path).resolve()
        if not path.is_dir():
            raise typer.BadParameter(f"workload directory does not exist: {path}")
        result[name] = path
    return result


def _parse_name_values(values: list[str], *, option: str) -> tuple[tuple[str, str], ...]:
    result: dict[str, str] = {}
    for value in values:
        name, separator, raw_value = value.partition("=")
        if not separator or not name or not raw_value:
            raise typer.BadParameter(f"each {option} must be NAME=VALUE")
        if name in result:
            raise typer.BadParameter(f"duplicate {option} name: {name}")
        result[name] = raw_value
    return tuple(sorted(result.items()))


def _emit_json(payload: object) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
