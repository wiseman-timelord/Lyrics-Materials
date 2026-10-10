# Lyrics-Materials
Status: Beta - Mid-Late development, improving, correcting. Note the images shown below were generated with v0.25.

### Description:
It convert lines of lyrics into AI generated images, one for each line. The idea is one could arrange these in a movie editor, and fade in/out between sections of images with theme images in the non-lyrics parts for fillers, to make simple music videos. One could also take the line of lyrics and the generated images, and feed that into AI video generator to make clips, that could then be assembled into a video.

### Media:
- Here is the Initial Project Page where there is now better image frequency control (v0.26)...
![Generation_Page](https://github.com/wiseman-timelord/Lyrics-Materials/blob/main/media/Management_Page.jpg)

- Here one reviews work and re-generates individual unfitting images (v0.25)...
![Generation_Page](https://github.com/wiseman-timelord/Lyrics-Materials/blob/main/media/Thumbnails_Page.jpg)

- The Configuration page is refined, with two individual models columns (v0.25)...
![Generation_Page](https://github.com/wiseman-timelord/Lyrics-Materials/blob/main/media/Configuration_Page.jpg)

### Features:
- **User Character;** controls for, Reference Image, Gender, Bodyshape, Physical Age 25-95, Hair Style, Outfit Worn.
- **Clever Model Handling;** configure the Encoding processes to, OtherGPU or CPU, to extend Flux2 GPU Memory, while still generating Prompts on GPU too.
- **Local Tool;** not requiring the use of online antigenic services to create your own local model based music slideshows.
- **libraries;** llama.cpp for song analysis + per-line visual prompts, stable-diffusion.cpp for Flux.2-klein. ffmpeg for utility/probe.

### Models
Available on [HuggingFace.Co](https://huggingface.co/)...  
- Encoder/Prompting: [Huihui-Qwen3-VL-4B-Thinking-abliterated-GGUF](https://huggingface.co/mradermacher/Huihui-Qwen3-VL-4B-Thinking-abliterated-GGUF) for Rich analysis + prompts when set (I used q4).
- Diffuser: [flux-2-klein-4b-GGUF](https://huggingface.co/models?search=flux2%204b%20gguf) Image generation (one of those, cant remember which one I used) (I used q8).
- VAE: [diffusion_pytorch_model.safetensors](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/vae/diffusion_pytorch_model.safetensors?download=true) the other image generation file (one file).
- Note the mmproj from the Qwen3-VL-4B is not used, and if present then is auto-moved to `**model_folder**\mmproj\`, saving complication in the scripts, but the mmproj is superseded by Flux2.
- Program designed for Portrait mode monitor, but it should/will work on Landscape too just with some sliders.

### Instructions:
- These are my current instructions for using the program...(Note; If you are unsure as to if Flux2 will be ok with your settings and VRam settings, then open Task Manager, there check python process, and monitor GPU memory/shaders. 
- As I have repeating stated "for 8GB VRam you would want image settings to, 768x512 or 640x360 (half 720p), that is with Diffuser Placement set to "Split". I would assume one could load the complete Flux2 model to GPU on 12GB card, but then Encoder must be loaded at same time, so possibly still need cpu for Encoder if Full; failure to take consideration may crash or mess up graphics drivers, worse case scenario requiring factory reset install.)
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
- Got your own lyrics, but do need your own songs to make AI music slideshow videos with Lyrics-Materials, then why not give [Suno](https://suno.com/invite/@wisemantimelord) (its an affiliate link so I get kudos), but its what I typically use, from experience I found basic use to be simple to understand, you yourself can do the singing, while the more you look at the interface and learn it then the more you can find.
- Ensure that the reference image is not HUGE, and I advise trimming it down to, torso and head, or bust and head, then setting the bodyshape correctly. Limiting the reference image to, smaller body area and full head, assists with facial likeness. 
- At 8 Steps per image, most things turn out ok, but at 10 steps the eyes will more likely be correct and not weird looking. At 12 steps, its going to take forever, but I assume the eyes will be 100% correct at that point. 
- Remember by editing `.\scripts\configure.py` script it is possible to have custom hair/clothes/etc. Ie for clothing you would for example search for something unique like "Rocker" in configure script, and then change, prompt detail and relating GUI option, for that one or one of the others you wont use. 
- When a session is under-way, ensure to collapse the session slots column on the left, with the little "--><--" button. This will optimize the interface a little.
- If the Generation page panels relating to Details Mode do not show, try switching back and forth, between Panel Mode's (there are a lot of items to display). 
- This program uses Flux2-4B, if people would prefer a similar program, but that can make motion music videos, then I would need someone to donate enough to get a 16GB GPU, albeit I could also make Lyrics-Materials compatible with Qwen3-8b and Flux2-Klein-8b. No point developing what I cant test/use. 

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
- The outfits need improving, current list is from Image Glamour plus 2 for artist, though after review. For more range, enumerate common outfit types, and create larger list, also 
- add another control for footwear, ie, black-boots, manila-boots, black-shoes, manila-shoes, black-trainers, manila-trainers, none-specified (no text segment in prompt).
- Its not meant to be moving the mmproj to ".\models\mmproj\*", its instead supposed to be creating a mmproj folder in the location where the actual model being used is in, and moving it there.
- Option to have NONE of the images with background/other characters; option for all scenes will as are appropriate ONLY feature reference character or just the scene. 

### Disclaimer:
- Warnings of quantum-weirdness in advance, but if you are familiar with Flux-2-Klein and qwen3-4b-thinking, this seems to be a good showcasing of what it can do, but you yourself are the one whom pulls the lever on your own configurations and Lyrics, then unexpected results will occur, because I probably havn't seen it before, one can always use the Negative prompt to filter out their phobias/fears in a box on the bottom of panel at location of  `Management > Name and Lyrics`.
- Do not over-load your GPU, it could cause graphics driver crash. As stated, ensure you understand the capabilities of your card. My max settings for 8 GB GPU was 768x512 (no bigger), and image models were loaded in Split mode. If your graphics driver does start to crash due to low ram, then try to close the program (python command prompt is fastest) IMMEDIATELY, and re-think your configurations.
