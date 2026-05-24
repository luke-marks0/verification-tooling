"""D6 Phase 2 PP=2 smoke — distributed inference over a 2-node Ray cluster.

Triggers modules/inference/runner/vllm_runner.py's `pp_size > 1` branch, which pins the
cross-node NCCL environment. Prints TOKEN_IDS on stdout for diff.
"""
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"

# Manually apply the NCCL pinning that modules/inference/runner/vllm_runner.py would set
# for pp_size > 1 / VLLM_MULTI_NODE runs. Required because this smoke script
# calls LLM() directly instead of going through the runner.
os.environ["NCCL_ALGO"] = "Ring"
os.environ["NCCL_PROTO"] = "Simple"
os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "WARN")
os.environ["NCCL_NET"] = "Socket"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["NCCL_BUFFSIZE"] = "8388608"
os.environ.setdefault("NCCL_SOCKET_IFNAME", "eno1")
# Disable vLLM's Ray-wrapped PP communicator (vllm 0.17.1 + ray 2.54 bug:
# ValueError "cuda_stream other than the current stream is not supported").
os.environ["VLLM_USE_RAY_WRAPPED_PP_COMM"] = "0"

# Patch vllm's RayPPCommunicator to accept Ray 2.54's cross-stream call.
# vllm 0.17.1 raises on any cuda_stream != current_stream(); Ray 2.54 passes
# a communication stream. We drop the check — NCCL ops will still run on the
# passed stream, which is consistent across actors.
import vllm.distributed.device_communicators.ray_communicator as _rc  # noqa: E402
_orig_init = _rc.RayPPCommunicator.__init__
def _patched_init(self, world_size, comm_id, rank, actor_handles,
                  cuda_stream=None, use_communication_streams=False):
    import ray
    self._world_size = world_size
    self._rank = None
    self._actor_handles = actor_handles
    if rank is not None:
        from vllm.distributed.parallel_state import get_pp_group
        assert ray.get_gpu_ids(), "RayPPCommunicator has no GPUs assigned"
        self._comm = get_pp_group().device_communicator
        assert self._comm is not None
        self._rank = self._comm.rank_in_group
        self._build_actor_rank_mapping()
    else:
        self._comm = None
    self._closed = False
_rc.RayPPCommunicator.__init__ = _patched_init

from vllm import LLM, SamplingParams  # noqa: E402

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    seed=42,
    enforce_eager=True,
    max_model_len=512,
    pipeline_parallel_size=2,
    tensor_parallel_size=1,
    distributed_executor_backend="ray",
    attention_backend="FLASH_ATTN",
)
params = SamplingParams(temperature=0, max_tokens=20)
out = llm.generate(["The meaning of life is"], params)
gen = out[0].outputs[0]
print("TOKEN_IDS:", list(gen.token_ids))
print("TEXT:", gen.text)
