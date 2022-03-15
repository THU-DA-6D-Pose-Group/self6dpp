import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="gen ycbv cfgs from another base cfgs")
    parser.add_argument("--base_cfg", type=str)
    parser.add_argument("--old", type=str)
    parser.add_argument("--new", type=str)
    args = parser.parse_args()
    base_cfg = args.base_cfg
    old = args.old
    new = args.new

    new_cfg = base_cfg.replace(old, new)
    os.system(f"cp -r {base_cfg} {new_cfg}")
    paths = os.listdir(new_cfg)
    for path in paths:
        new_path = path.replace(old, new)
        os.system(f"mv {new_cfg}/{path} {new_cfg}/{new_path}")
    os.system("ls %s|xargs -I {} sed -i 's/%s/%s/' %s/{}" % (new_cfg, old, new, new_cfg))


if __name__ == "__main__":
    main()
