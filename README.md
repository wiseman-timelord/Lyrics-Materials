# Lyrics-Materials
Status: Beta - Basic functioning is done, now developing and improving.

### Description:
It will convert lines of lyrics into AI generated images, one for each line. The idea is one could arrange these in a movie editor, and fade in/out between sections of images, to make simple music videos. One could also take the line of lyrics and the generated images, and feed that into AI video generator to make clips, that could then be assembled.

### Media:
- Early version, somewhat working (v0.08)...
![Generation_Page](https://github.com/wiseman-timelord/Lyrics-Materials/blob/main/media/Generation_Page.jpg)

### Features:
- Local Windows tool: paste lyrics → one still per non-marker lyric line → `output/<song_name>/` folder of numbered PNGs for use in external AI video tools.
- Not a video app: slideshow assembly, section fades, Whisper timing, and audio input were removed when converting from Lyrics-Slideshow.
- Clever Model Handling; If Thinking and ImageGen share the same Vulkan device and either uses M-Lock, Thinking fully unloads before Flux loads — they are never resident together.
- Two-phase orchestration: Phase 1 = assessment + prompts (Thinking preferred, else Encoder),  Phase 2 = image generation (Flux on its backend)

### Libraries:
```
**llama.cpp** (`llama-completion`) for song analysis + per-line visual prompts.
**stable-diffusion.cpp** (`sd-cli`) for Flux.2-klein stills.
ffmpeg remains installed for utility/probe use only — not used for materials output.
```

### Models:
- Available on [HuggingFace.Co](https://huggingface.co/)... 
```
| Role | File | Used for |
|------|------|----------|
| Encoder | `Huihui-Qwen3-VL-4B-Instruct-abliterated*.Q4_K_M.gguf` | Analysis/prompts; also passed as Flux.2 `--llm` text conditioner |
| Thinking (optional) | `Huihui-Qwen3-VL-4B-Thinking-abliterated*.Q5_K_M.gguf` | Richer analysis + prompts when set (preferred over Encoder for Phase 1) |
| Diffuser | `flux-2-klein-4b-Q8_0.gguf` | Image generation (`--diffusion-model`) |
| VAE | `diffusion_pytorch_model.safetensors` |
| mmproj | not used for pure text — any `mmproj*.gguf` is auto-moved to `models\mmproj\` |
```

### Instructions:
- Usage...
```
1. User enters **song name** (folder slug) and pastes lyrics; optional single **reference image** for central character.
2. Phase 1 — **Thinking** (if set) else **Encoder** (`llama-completion`):
   - Overall song + section analysis
   - **CHARACTER_MAP** per section: `none | silhouette | partial | full`
   - One visual prompt per lyric line (blanks and pure [INTRO/CHORUS/OUTRO] markers skipped)
3. Explicit unload barrier if Thinking and Flux share a device under M-Lock.
4. Phase 2 — **FLUX.2-klein** (`sd-cli`):
   - `--diffusion-model` + `--vae` + `--llm` = Encoder (Qwen3-VL) path
   - Reference photo attached with `-r` only when that line’s presence ≠ `none`
   - Hard-coded **768×512** stills named `NNN - lyric line.png`
5. Project folder: `output/<song_name_with_underscores>/` with images, `lyrics.txt`, `analysis.txt`, `prompts.txt`, `character_map.txt`.
```

### STRUCTURE:
```
Lyrics-Materials/
├── Lyrics-Materials.bat
├── launcher.py
├── installer.py
├── data/          constants, configs, binaries, ffmpeg, temp
├── scripts/       configure, display, inference, utilities
├── models/
└── output/<song_name>/
        001 - first lyric line.png
        …
        lyrics.txt · analysis.txt · prompts.txt · character_map.txt
```

### Development:
- Working on it, til it works.
