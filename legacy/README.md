# Legacy Prototypes & Historical Scripts

This directory preserves early exploratory prototypes developed during the initial phase of the project:

* `optimize_lyrics_sync.py` - Initial monolithic prototype containing early phonetic feature extraction, continuous tempo estimation, and genetic optimization routines before the codebase was refactored into the modular `lyrics_aligner/` package.
* `download_top_songs.py` - Early prototype for fetching benchmark tracks. The maintained utility is now located in `scripts/download_top_songs.py`.
* `learned_parameters.json` - Historical root checkpoint. The production weights are now packaged directly inside `lyrics_aligner/learned_parameters.json`.
