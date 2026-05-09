# Memetic Pyramid PSO for faster Fractal Image Compression
This project proposes a novel approach to accelerate Fractal Image Compression (FIC) encoding by employing a memetic algorithm combined with Pyramid Particle Swarm Optimization (PSO).
In our `encoders/` folder we provide not only the optimal algorithm, but also the baseline algorithm for experiment.

## Requirements
The code is tested on FreeBSD and Arch Linux (Other Unix-like OS might also run our program successfully). Python>=3.6 is required to run the code.

## Installation
Just simply install python>=3.6 and Pillow.

## Usage
### Run Experiments
To run the experiments with the default configurations on all images in the default directory, simply run:
```shell
$ python3 run_experiments.py
```

If you want to test your proposed PSO method and the baseline on a specific image (e.g., boat.png), you can use:
```shell
$ python3 run_experiments.py --methods baseline pso --image boat.png
```

To test the algorithms with larger block sizes (which yields a higher compression ratio but lower image quality) and a smaller image resolution:
```
$ python3 run_experiments.py --image-size 128 --range-size 8 --domain-size 16
```


### Decode from .fic file
```shell
$ python fic_bitstream.py results/boat_full_search.fic
$ python fic_bitstream.py results/boat_full_search.fic results/boat_decoded.png
```

If you want to decode the `.fic` file from our `fic_bitstream.py` library, there is a sample code:
```python
from fic_bitstream import decode_fic_file

reconstructed = decode_fic_file(
    fic_path='results/boat_pso.fic',
    output_path='results/boat_decoded.png',
    decode_iterations=20
)
```
If the `output_path` is not given, the library will rename it automatically.

## Compression file format
Our compression file is `.fig`, which encoded from our FIC encoder algorithms.
The `.fig` file format is:
  ┌-------------------------------------┐
  |  Header (16 bytes)                  |
  |    magic:        4 bytes  "FIC1"    |
  |    img\_height:   2 bytes  uint16   |
  |    img\_width:    2 bytes  uint16   |
  |    range\_size:   1 byte   uint8    |
  |    domain\_size:  1 byte   uint8    |
  |    domain\_stride:1 byte   uint8    |
  |    bits\_s:       1 byte   uint8    |
  |    bits\_o:       1 byte   uint8    |
  |    reserved:     3 bytes  (padding) |
  |─────────────────────────────────────|
  |  Payload (bitstream)                |
  |    Every fractal code = 29 bits     |
  |    [domain\_idx | iso | s\_q | o\_q]|
  └-------------------------------------┘
The `.fig` will pack tightly, and padding with zeros if the last byte is less than 1 byte.

## License
This project is [MIT](https://choosealicense.com/licenses/mit/) licensed.
