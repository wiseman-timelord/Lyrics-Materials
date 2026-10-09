# Lyrics-Materials
Status: Beta - Developing and improving. This is Experimental. Note the images shown below were generated with v0.20, generation and reference image likeness, is now figured out, and the interface is quite ok.

### Description:
It will convert lines of lyrics into AI generated images, one for each line. The idea is one could arrange these in a movie editor, and fade in/out between sections of images, to make simple music videos. One could also take the line of lyrics and the generated images, and feed that into AI video generator to make clips, that could then be assembled.

### Media:
- As intended, it now makes 3 types of images, also with, reference image and some controls (v0.20)...
![Generation_Page](https://github.com/wiseman-timelord/Lyrics-Materials/blob/main/media/Generation_Page.jpg)

### Features:
- **Local Tool**; paste lyrics → one still per non-marker lyric line → `output/<song_name>/` folder of numbered PNGs for use in external AI video tools.
- **Clever Model Handling;** If Thinking and ImageGen share the same Vulkan device and either uses M-Lock, Thinking fully unloads before Flux loads — they are never resident together.
- **Two-phase orchestration;** Phase 1 = assessment + prompts (Thinking preferred, else Encoder),  Phase 2 = image generation (Flux on its backend)
- **llama.cpp;** (`llama-completion`) for song analysis + per-line visual prompts.
- **stable-diffusion.cpp;** (`sd-cli`) for Flux.2-klein stills.
- **ffmpeg;** for utility/probe use only — not used for materials output.

### Models
Available on [HuggingFace.Co](https://huggingface.co/)...  
- Encoder: [Huihui-Qwen3-VL-4B-Instruct-abliterated-GGUF](https://huggingface.co/mradermacher/Huihui-Qwen3-VL-4B-Instruct-abliterated-GGUF/tree/main) for text encoding images (I used q4).
- Thinking: [Huihui-Qwen3-VL-4B-Thinking-abliterated-GGUF](https://huggingface.co/mradermacher/Huihui-Qwen3-VL-4B-Thinking-abliterated-GGUF) for Rich analysis + prompts when set (I used q4).
- Diffuser: [flux-2-klein-4b-GGUF](https://huggingface.co/models?search=flux2%204b%20gguf) Image generation (one of those, cant remember which one I used) (I used q8).
- VAE: [diffusion_pytorch_model.safetensors](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/vae/diffusion_pytorch_model.safetensors?download=true) the other image generation file (one file).
- Note the mmproj is not used, and if present then is auto-moved to `models\mmproj\`

### Instructions:
- These are my current instructions for using the program...(Note; If you are unsure as to if Flux2 will be ok with your settings and VRam settings, then you may want Task Manager open at this stage, with the python process selected ready to End Task if you get VRam overload on model load, while to also be able to monitor goings on with the GPU memory/shaders if everything goes ok. 
- As I have repeating stated "for 8GB VRam you would want image settings to, 768x512 or 640x360 (half 720p), that is with Diffuser Placement set to "Split". I would assume one could load the complete Flux2 model to GPU on 12GB card, but then Encoder must be loaded at same time, so possibly still need cpu for Encoder if Full; failure to take consideration may crash or mess up graphics drivers, worse case scenatio requiring factory reset install.)
1. The program will start, you will be in a new project, so collapse the left pane, unless you are going to hop sessions.
2. User enters **song name** (folder slug) and pastes lyrics; optional (advised) single **reference image** for central character.
3. Ensure Image frequency and other settings are correct (and not too wild, see notes), then click "Generate All Assets".
4. When all assets are generated, then click on "Lyrics Thumbnails", have a look at the images, and regenerate them individually as required.
5. When images are all how you intended, then copy them to the movie making app you have. For basic/simple editing, possibly you could use "Microsoft Movie Maker" that came with the "Microsoft Essentials 2012" package.
- Lyrics should be in format of (as many chorus as you like)...
```
[Intro]
**intro**

[Chorus 1]
**chorus**

[Chorus 2]
**chorus**

[Chorus 3]
**chorus**

[Outro]
**outro**
```

### Notation:
- Ensure that the reference image is not HUGE, and I advise trimming it down to, torso and head, or bust and head, then setting the bodyshape correctly. Limiting the reference image to, smaller body area and full head, assists with facial likeness. 
- At 8 Steps per image, most things turn out ok, but at 10 steps the eyes will more likely be correct and not weird looking. At 12 steps, its going to take forever, but I assume the eyes will be 100% correct at that point. 
- Remember by editing `.\scripts\configure.py` script it is possible to have custom hair/clothes/etc. Ie for clothing you would for example search for something unique like "Rocker" in configure script, and then change, prompt detail and relating GUI option, for that one or one of the others you wont use. 
- When a session is under-way, ensure to collapse the session slots column on the left, with the little "--><--" button. This will optimize the interface a little.
- If the Generation page panels relating to Details Mode do not show, try switching back and forth, between Panel Mode's (there are a lot of items to display). 
- This program uses Flux2-4B, if people like the program and have the money, then please donate ~£250 to my kofi/patreon, and then I will make the program compatible with Qwen3-8b and Flux2-Klein-8b. No point developing what I cant test. 

### Struture:
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
- Apparently the qwen3-vl-thinking model can generate the prompts AND encode the images? if this is the case, please ensure to shift ALL instruct model duties over to the thinking model, so we can reduce the number of models used  
- Thumbnails on a new page, to speed up and fix issues on the Generation page, Generation page needs renaming to Management page.
- Option to have NONE of the images with background/other characters; option for all scenes will as are appropriate ONLY feature reference character or just the scene. 

### Disclaimer:
- Do not over-load your GPU, it could cause graphics driver crash. As stated, ensure you understand the capabilities of your card. My max settings for 8 GB GPU was 768x512 (no bigger), and image models were loaded in Split mode. If your graphics driver does start to crash due to low ram, then try to close the program (python command prompt is fastest) IMMEDIATELY, and re-think your configurations.
