# Independent resident refiner

`DGXMonarchRefinerUNETLoader` describes a separate `refiner` slot. The existing
primary (`cond`) and CFG-negative (`uncond`) loaders retain their meanings and
saved input order. The optional model-request `slot` field is emitted only for
this new loader; absent still means `cond`. Unknown slot names never alias the
primary slot. Both hosts must run this exact source before using the new node.

The first supported execution scope is serial pure Ulysses, CFG degree one,
without FSDP or a dual-checkpoint guider. One actor keeps the current primary
base and the finishing model; changing the primary base need not release the
finisher. This does not promise that several primary checkpoints remain loaded.

All store snapshots, per-rank capacity facts, model adoption, sampling and
explicit unload include the finishing slot. Already resident memory stays in
MemAvailable; a second pending load is charged against the remaining budget.
The existing capacity formulas, floor and low-RSS slab relocation fix remain.
No capacity failure silently unloads a peer resident to continue the request.
The old-model automatic fleet-clear offer is suppressed for refiner loads or
a held refiner; the original capacity refusal stands. An explicit operator
reset or stop still releases the mesh.

ComfyUI's public `free_memory(..., keep_loaded=...)` seam protects peer models
only during a serialized GPU load or sample. The context is reset on failure
or interruption. An explicit store-wide unload remains able to release all
slots. The existing ComfyUI implementation and attention kernels are unchanged.

This is a residency mechanism, not a model-conversion implementation. Latent
converters, upscalers, VAEs, conditioning and frame/audio handling belong to the
calling graph. A finishing model receives its own conditioning and latent
format; a foreign model's raw latent cannot be wired directly into it.

Sources: the repository's DESIGN sections 5.2 and 5.5, and ComfyUI commit
2f8943324e1eae89a5ee6c7db93602299246b5f1 `comfy/model_management.py` public
`free_memory` / `load_models_gpu` APIs. No dependency implementation is vendored.

Validation status: CPU regressions are in `tests/test_refiner_residency.py`.
Hardware co-residency, accuracy and memory stability remain NOT RUN until the
operator acceptance is recorded against the final source manifest.
