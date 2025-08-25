# dycoke_token_compression

This is **NOT** an official implementation of DyCoke. For the official implementation, please refer to this [repo](https://github.com/KD-TAO/DyCoke).

As compared to official implementation, This repo integrates DyCoke with more recent VLMs such as Gemma3 and InternVL3


## Setup

To use this repo, you need two key packages to be installed in a venv with `python>=3.10`
```commandline
torch==2.5.1
tranformers==4.53.0
```

## Demo

To have a quick demo of DyCoke with Gemma3, please run

```commandline
python test_gemma3_with_dycoke.py --video resources/example_video.mp4 --prompt "Explain the video." --use_dycoke
```

DyCoke can be tunred off by removing `--use_dycoke` argument. Please note that to avoid OOM errors I have configured
`utils/video_reader.py` to select every 12th frame.

### Gemma3 with DyCoke vs Gemma3 without DyCoke

### Gemma3 with DyCoke vs Gemma3 without DyCoke

<table>
  <tr>
    <td align="center"><b>With DyCoke</b></td>
    <td align="center"><b>Without DyCoke</b></td>
  </tr>
  <tr>
    <td>
      <img src="resources/demo_with_dycoke.webm" alt="Demo with DyCoke enabled">
    </td>
    <td>
      <img src="resources/demo_without_dycoke.webm" alt="Demo without DyCoke enabled">
    </td>
  </tr>
</table>
