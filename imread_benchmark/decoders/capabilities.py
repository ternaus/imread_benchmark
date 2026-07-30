from __future__ import annotations

from dataclasses import dataclass

from imread_benchmark.contracts import OutputContract
from imread_benchmark.decoders import BaseDecoder

CAPABILITY_SCHEMA_VERSION = "2.0"


@dataclass(frozen=True, slots=True)
class DecoderCapabilities:
    decoder_id: str
    package_name: str
    dependency_group: str
    input_sources: tuple[str, ...]
    output_contracts: tuple[OutputContract, ...]
    native_output_available: bool
    normalization_location: str
    thread_control: str
    process_compatible: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "decoder_id": self.decoder_id,
            "dependency_group": self.dependency_group,
            "input_sources": list(self.input_sources),
            "native_output_available": self.native_output_available,
            "normalization_location": self.normalization_location,
            "output_contracts": [contract.to_dict() for contract in self.output_contracts],
            "package_name": self.package_name,
            "process_compatible": self.process_compatible,
            "schema_version": CAPABILITY_SCHEMA_VERSION,
            "thread_control": self.thread_control,
        }


def describe_decoder(decoder_cls: type[BaseDecoder]) -> DecoderCapabilities:
    thread_control = "settable" if decoder_cls.set_num_threads is not BaseDecoder.set_num_threads else "fixed"
    return DecoderCapabilities(
        decoder_id=decoder_cls.name,
        package_name=decoder_cls.package_name,
        dependency_group=decoder_cls.group,
        input_sources=("bytes", "path"),
        output_contracts=(OutputContract.normalized_rgb(),),
        native_output_available=decoder_cls.native_output_available,
        normalization_location="decoder-adapter",
        thread_control=thread_control,
        process_compatible=decoder_cls.in_dataloader,
    )
