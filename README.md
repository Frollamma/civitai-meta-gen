# Metadata generation tool for Civitai

A cli tool that adds metadata to an image to make it compatible with Civitai format. With one short command you can add metadata to one or multiple images.

## Problem Solved

When uploading images to Civitai, the platform requires AI generation metadata to be present in the image files. Without this metadata, uploads are blocked with "can't detect metadata" errors. This tool adds realistic AI generation metadata to your images, making them compatible with Civitai's requirements.

## Installation

To install simply run

```sh
pip install
```

## Usage

The typical usage is adding metadata images:

```sh
civitai-meta-gen add my_image.png
OK: my_image.png -> /home/user/Desktop/my_image_with_metadata.png
Done. ok=1 failed=0
```

For all the commands run

```sh
civitai-meta-gen --help
```

for each command, you can get more information by running it with `--help`, for example, for the `add` command you can run

```sh
civitai-meta-gen add --help
```

## Credits

This project was initially forked from `https://github.com/m0nkeypantz/Civitai-metadata-compatibility-tool/`, that does the same with a GUI, I simply wanted a CLI instead of a GUI.
