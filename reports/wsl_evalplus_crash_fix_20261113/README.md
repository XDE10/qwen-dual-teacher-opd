# WSL EvalPlus crash diagnosis and scoped repair (2026-11-13)

## Scope
Only this repository's future EvalPlus Docker evaluations are changed. No historical samples, scores, training manifests, checkpoints, or approval fields were edited. Docker image remains pinned to sha256:79a6508d6b82dd8319c1769e87e346d02b4a4b80a9847ebec691ba453fd67204 (Python 3.10.22, EvalPlus 0.3.1). No candidate code was executed on the host.

## Cause and evidence
The two November 12 WSL ELF cores identify `python -m evalplus.evaluate --dataset mbpp`. Both have SIGSEGV in libpython3.10, the stack pointer 96 bytes below the mapped stack, and 4.037 GiB inherited virtual mappings. The adapter imposed an absolute 1 GiB RLIMIT_AS through EvalPlus reliability_guard, below already inherited mappings. This prevents stack growth.

The checkpoint-50 MBPP/392 candidate has no n=0 recursion base case. The original result marked base and plus as timeout; the whole 378-task evaluation nevertheless completed with exit code 0. MBPP/392 is the strongly correlated triggering candidate; the original core was not symbolically unwound to its Python function name.

A controlled test in the exact pinned Docker image reproduced SIGSEGV (child exit -11) with the original 1 GiB limit. With 4 GiB and a low baseline, it raised RecursionError normally. A simulated >4 GiB virtual baseline demonstrated that a fixed 4 GiB remedy is not sufficient. Baseline-plus-1-GiB headroom caught RecursionError normally.

## Repair
- Default resource protocol: `vms-headroom-v1`.
- Each evaluation child reads its own current `/proc/self/statm` immediately before the guard. It receives 1 GiB additional virtual address space, not a 1 GiB absolute cap below its inherited state.
- Existing Docker 6 GiB memory, 2 CPU, PID limit, read-only filesystem, non-root user, no-network and capability restrictions remain.
- `--memory-policy legacy-fixed-1g` retains the old absolute memory-limit behavior for explicit investigation. It is not recommended for normal runs. The command is still recorded in official_protocol.json.
- `resource_protocol`, memory budget, core limit and evaluator dumpability are recorded. Scores summaries include resource protocol. Historical resource constraints are not directly equivalent: do not silently pool new and legacy scores.
- EvalPlus task definitions, reference tests, candidate solutions and pass/fail scoring rules are unchanged.
- Imported module `cli.main()` is used so ProcessPoolExecutor callables remain pickleable. An intermediate runpy entry failed end-to-end and was replaced before completion.

## WSL core-dump detail
`--ulimit core=0:0` alone was experimentally insufficient: WSL's piped CaptureCrash still wrote three new test dumps. The final entry also calls Linux PR_SET_DUMPABLE=0 for only the evaluator process, inherited by fork children. No global WSL core_pattern or system security setting was changed.

A controlled fork child with this flag produced exit -11 while the WSL crash directory count/bytes stayed unchanged (5 files / 8560148480 bytes). Old and new dumps were not deleted during investigation.

New test dumps (not user workload):
- wsl-crash-1794538087-8-_usr_local_bin_python3.10-11.dmp: 1031446528 bytes
- wsl-crash-1794538118-7-_usr_local_bin_python3.10-11.dmp: 1209028608 bytes
- wsl-crash-1794538194-8-_usr_local_bin_python3.10-11.dmp: 1463508992 bytes
Total 3703984128 bytes. These must be disclosed before cleanup.

## Verification
- tests/test_code_evaluation_v2.py: 10 passed (including two new policy regression tests).
- Changed Python source files: AST syntax checks passed.
- Patched entry under >4 GiB virtual baseline: correct function passes, infinite recursion fails normally, oversized 8 GiB allocation fails; all worker exit codes 0.
- Actual MBPP/2 and original MBPP/392 samples + official data: end-to-end through EvalPlus 0.3.1; good sample pass/pass, recursive sample fail/fail. Historical output directories untouched; temporary outputs stay inside the removed container.
- Full project suite was NOT verified: tests/test_eval_v3.py collection on the available Windows Python failed because transformers is absent. No dependency was installed to hide this limitation.
- Entire 378-task historical workload was not rescored; no claims of unchanged aggregate scores or universal future crash immunity.

## Operational issue
Docker Desktop started successfully and Windows Docker CLI worked. Ubuntu WSL integration intermittently returned WSAETIMEDOUT. Repository edits used the functioning Windows WSL file share without restarting WSL or stopping user sessions. This Docker integration issue is separate from the reproduced EvalPlus resource-limit bug.
## Confirmed cleanup after owner approval
After the owner explicitly approved permanent deletion of the two original dumps and the three test dumps, all five exact filenames and byte sizes were revalidated. Only those five regular files were deleted. The directory is now empty.
- Deleted bytes: 8,560,148,480 (7.972 GiB).
- C: free before: 68,257,992,704 bytes.
- C: free after: 76,818,149,376 bytes.
- Observed net free-space increase: 8,560,156,672 bytes (minor concurrent filesystem activity explains the 8,192-byte difference).
No other file was deleted by this cleanup step.
