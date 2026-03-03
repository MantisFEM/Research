#! /usr/bin/env bash

# This script will update 'README.md' from 'markdown/README.md'.

# Check if 'pandoc' is installed and in 'PATH'.
if ! command -v pandoc >/dev/null 2>&1; then
	echo "'pandoc' not found in 'PATH'. Please install or add to 'PATH'."
	exit 1
fi

# Update README
pandoc markdown/README.md \
	--from markdown+citations \
	--to gfm \
	--citeproc \
	--wrap=none \
	--csl=markdown/apa.csl \
	--bibliography=markdown/references.bib \
	--metadata=reference-section-title:References \
	-o README.md
