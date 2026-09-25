# Cado website

A static, responsive website for GitHub Pages. Source: `index.html`, `style.css`, and `app.js`.

## Visual direction

Cream paper, olive ink, sage and dusty blue. Oversized lowercase typography, editorial rules, botanical print artwork. References supplied by the user: Tooooools, Sort Office, Mac Moss, and two attached images. Kevin Liu's imagegen-frontend-web skill informed palette discipline, varied section rhythm, whitespace, and art direction. The deliverable is a working website rather than a set of screenshot mockups.

## Generated artwork

`assets/clover-field.png` was generated using the built-in image_gen tool and inspected. The user's first attachment served as a visual reference; neither source attachment is published.

Prompt:

> Use case: stylized-concept. Asset type: original website hero illustration, landscape 1536x1024. Use the supplied image as a STYLE REFERENCE only: its tactile, stippled risograph/lithograph grain, softly fuzzy ink edges, soft green and blue botanical palette, and ivory four-leaf clover negative-space motifs. Create a new composition of exactly THREE avocado-shaped oval pebble cross-sections at gently different oblique angles. A large main oval in the center, and two smaller flanking ovals, separated with generous negative space and broad margins. Each oval contains one ivory FOUR-leaf clover, with exactly four heart-shaped leaves and a delicate subtle stem. Give each flattened oval a softly varied internal ink wash: pale lime #bccf8a, muted sage #aabaaf, dusty blue #a9bfc6. Sophisticated flat analog botanical artwork, airy and quiet. Uniform warm ivory paper background #f5f3e8. The forms are like the abstract avocado/clover print reference, not realistic food. Tactile speckled ink and subtle lithographic color overlap within the ovals. No text, no interface, no logos, no border, no photorealism, no glossy or shiny 3D, no cast shadows on the background. Output one single landscape image.

## Data and usage

Benchmark values are sourced from `tplane/docs/results/rft-faultbench.md`, evaluated September 24, 2026. The page includes the naive baseline, false-alarm counts, label protocol, baseline provenance, and selection-bias caveat. The outcome diagram is explicitly illustrative. The local Python example disables resource sampling for portability; see SDK documentation for production readers.

## Local preview

From the repository root:

```sh
python3 -m http.server 4173 --directory website
```
