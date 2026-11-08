# provenance/ — 冻结发布的哈希锚定证据

本目录保存**冻结数据发布清单里按 sha256 锁定的历史证据文件**。这些文件原本散落在
`data/`、`reports/`、`src/` 的旧路径中，重排时原目录被删除，但发布清单仍要求其字节
可校验，因此**原样镜像**到此处（内容不变，sha256 不变）。

- 映射表：`docs/migration/path_mapping.json`（旧路径 → 新位置，含期望/实际 sha256）。
- 校验：`python tools/verify_code_repair_v4.py` 与 `python tools/verify_repository.py`
  会先按原路径查找，缺失时回退到本目录，再比对 `expected_sha256`。
- 本目录只作为证据存档，不参与训练/评测的运行时读取。

## 已知缺口（重排前即存在）

- `data/project3_audit/geometry3k_intergps_links.jsonl` 在 `data/portable_release/manifest.json`
  的 `inputs` 中有哈希记录，但该文件在重排前已不在仓库中；此项在映射表里标记为
  `missing_before_reorg`，不是本次重排造成的删除。
