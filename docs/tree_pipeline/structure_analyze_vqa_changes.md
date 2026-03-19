# Structure Analyze VQA Changes

## Summary

This update extends `structure_analyze`-related logic so markdown input can be split into modality-aware components for the tree pipeline:

- `text`: keeps the original title-based segmentation behavior.
- `table`: detects HTML `<table>...</table>` blocks and tries to attach the nearest caption-like line above the table.
- `image`: detects markdown image syntax and attaches caption text below the image while preserving nearby `Note:` text.

The goal is to let downstream tree/VQA operators consume structured `text`, `table`, and `image` components instead of treating everything as plain text.

## Main Changes

### 1. Markdown modality parsing

File: `graphgen/operators/tree_pipeline/tree_utils.py`

- Added markdown-aware parsing in `normalize_components(...)`.
- Added image-path extraction for markdown and HTML image tags.
- Added table-caption detection using a simple `Table 1...` style heuristic.
- Added support for image notes between image blocks and captions.
- Preserved compatibility with old pure-text title parsing.

### 2. StructureAnalyze-compatible output shape

File: `graphgen/operators/tree_pipeline/structure_analyze_service.py`

- `StructureAnalyzeService.process()` continues to return `component_pack`.
- Each component now carries modality-specific metadata when applicable.

Expected component shapes:

- `text`
  - `type`
  - `title`
  - `content`
  - `title_level`
- `table`
  - `type`
  - `title`
  - `content`
  - `title_level`
  - `metadata.table_body`
  - `metadata.table_caption`
- `image`
  - `type`
  - `title`
  - `content`
  - `title_level`
  - `metadata.img_path`
  - `metadata.image_caption`
  - `metadata.note_text`

### 3. Test fixture

File: `tests/fixtures/tree_vqa_demo.md`

The fixture covers:

- normal headings and paragraphs
- a table with caption
- a table without caption
- an image with caption below
- an image with note text between image and caption
- an image without caption

## Verification

Tests were run in the `graphgen` conda environment using `conda run -n graphgen`.

### Direct parser check

Command:

```bash
conda run --no-capture-output -n graphgen env PYTHONPATH=/home/lukashe/data/projects/GraphGen python - <<'PY'
from pathlib import Path
from graphgen.operators.tree_pipeline.tree_utils import normalize_components

fixture = Path('/home/lukashe/data/projects/GraphGen/tests/fixtures/tree_vqa_demo.md').read_text(encoding='utf-8')
components = normalize_components({'type': 'text', 'content': fixture})
print('types =', [c['type'] for c in components])
print('table_caption =', components[2]['metadata']['table_caption'])
print('table_without_caption =', components[4]['metadata']['table_caption'])
print('image_caption =', components[5]['metadata']['image_caption'])
print('image_note =', components[5]['metadata']['note_text'])
print('image_without_caption =', components[6]['metadata']['image_caption'])
PY
```

Observed result:

```text
types = ['text', 'text', 'table', 'text', 'table', 'image', 'image', 'text']
table_caption = ['Table 1. Accuracy across baselines.']
table_without_caption = []
image_caption = ['Figure 1. The microscope image highlights the reactive region after treatment.']
image_note = Note: arrows mark the highlighted tissue.
image_without_caption = []
```

### StructureAnalyzeService process check

Command:

```bash
conda run --no-capture-output -n graphgen env PYTHONPATH=/home/lukashe/data/projects/GraphGen python - <<'PY'
from pathlib import Path
from unittest.mock import patch

from graphgen.operators.tree_pipeline.structure_analyze_service import StructureAnalyzeService

class DummyKV:
    def get_by_id(self, key):
        return None
    def get_by_ids(self, ids):
        return []
    def upsert(self, batch):
        return None
    def update(self, batch):
        return None
    def reload(self):
        return None
    def index_done_callback(self):
        return None

fixture = Path('/home/lukashe/data/projects/GraphGen/tests/fixtures/tree_vqa_demo.md').read_text(encoding='utf-8')
with patch('graphgen.common.init_storage.init_storage', return_value=DummyKV()):
    service = StructureAnalyzeService(working_dir='cache', kv_backend='json_kv')
    rows, meta = service.process([{'_trace_id': 'read-md', 'type': 'text', 'content': fixture}])

components = rows[0]['components']
print('row_type =', rows[0]['type'])
print('component_types =', [c['type'] for c in components])
print('first_table_caption =', components[2]['metadata']['table_caption'])
print('first_image_caption =', components[5]['metadata']['image_caption'])
print('second_image_caption =', components[6]['metadata']['image_caption'])
print('meta_keys =', list(meta.keys()))
PY
```

Observed result:

```text
row_type = component_pack
component_types = ['text', 'text', 'table', 'text', 'table', 'image', 'image', 'text']
first_table_caption = ['Table 1. Accuracy across baselines.']
first_image_caption = ['Figure 1. The microscope image highlights the reactive region after treatment.']
second_image_caption = []
meta_keys = ['read-md']
```

### Note on pytest

Running the integration pytest directly in the sandbox hit a Ray socket permission issue during operator initialization. The parsing and `process()` logic itself was still verified by direct execution in the `graphgen` environment, with KV storage mocked to avoid unrelated runtime constraints.

## Git Notes

Suggested commit title:

```text
feat(tree_pipeline): split markdown text table image blocks for structure analyze
```

Suggested commit body:

```text
- parse markdown into text/table/image components in tree_utils
- attach table captions above html tables when available
- attach image captions below markdown images and preserve note text
- add markdown fixture and focused structure_analyze verification notes
```
