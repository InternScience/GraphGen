# GraphGen fork release and tagging policy

The fork is an independently versioned Python distribution (`graphg`) maintained at `Leon-Algo/GraphGen`. Upstream merge status does not control fork releases. Release tags are immutable identifiers for exact commits; never move or reuse a published tag.

## Tag format

Use `graphgen-fork-vMAJOR.MINOR.PATCH`:

- **MAJOR**: breaking config/profile/schema or graph identity changes requiring consumer migration.
- **MINOR**: backward-compatible capability or new profile/API.
- **PATCH**: backward-compatible bug fix or documentation/packaging correction.

PMS domain profiles are independently versioned by their profile IDs and are not implied by a package release. Profiles referenced by production configs must not be edited after release; changes get a new profile ID.

## Release gate

Before tagging:

1. Review the fork diff and working tree; ensure no unrelated user changes are included.
2. Run deterministic profile/contract/builder/partitioner tests, relevant upstream GraphGen tests, and PMS compatibility tests.
3. Run a non-PMS extraction → star graph → event_join smoke and a PMS compatibility smoke; confirm default LightRAG + ECE behavior is unchanged.
4. Build wheel and sdist in a disposable directory; inspect both for prompt/profile assets and install the wheel into a clean environment to load each built-in profile.
5. Run `git diff --check`, record exact commit, test commands/results, profile hashes and upstream base.

If any required gate fails, do not tag or push the release.

## Release sequence

1. Merge only reviewed release files into the intended fork branch; preserve unrelated local changes.
2. Commit with a concise message and verify the final commit SHA and clean release diff.
3. Confirm the desired tag is unused locally and on the fork remote.
4. Create the tag on that exact commit, then push the intended branch and tag to the `fork` remote (`https://github.com/Leon-Algo/GraphGen.git`). Never push the fork release tag to upstream by accident.
5. Publish release notes that include: distribution version, tag and commit SHA, upstream base commit, key behavior/config changes, profile IDs and asset hashes, compatibility/migration notes, license/source attribution, test and smoke evidence, and any known limitations.

Example commands (replace placeholders only after verification):

```bash
git tag -a graphgen-fork-v1.0.0 <verified-commit> -m "GraphGen fork 1.0.0"
git push fork pms/main
git push fork graphgen-fork-v1.0.0
```

Consumers should pin the release tag or preferably full commit SHA and lock their environment. Tag creation/push is an externally visible release action and must occur only after the release gate passes and the maintainer authorizes publication.
