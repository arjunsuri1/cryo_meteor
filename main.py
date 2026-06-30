import argparse
from mapio import read_map, write_map
from denoise import tv_denoise_grid


def main():
    parser = argparse.ArgumentParser(
        description="TV-denoise a cryo-EM map; lambda chosen by half-map FSC."
    )
    parser.add_argument("half1", help="first half-map (.mrc)")
    parser.add_argument("half2", help="second half-map (.mrc)")
    parser.add_argument("-o", "--output", default="denoised.mrc",
                        help="output path (default: denoised.mrc)")
    args = parser.parse_args()

    a, voxel_size, origin = read_map(args.half1)
    b, _, _               = read_map(args.half2)

    denoised = tv_denoise_grid(a, b)

    write_map(args.output, denoised, voxel_size, origin)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()