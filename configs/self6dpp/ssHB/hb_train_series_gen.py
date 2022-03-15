import os


def main():
    base = "./ss_dibr_mlBCE_FreezeBN_woCenter_woDepth_refinePM10_10e_train018"
    VARY_PERCENT_SPLITS = [
        "train045",
        "train090",
        "train180",
        "train270",
        "train360",
        "train450",
        "train540",
        "train630",
        "train720",
        "train810",
        "train900",
    ]
    for split in VARY_PERCENT_SPLITS:
        new_dir = base.replace("train018", split)
        os.system(f"cp -r {base} {new_dir}")
        paths = os.listdir(new_dir)
        for path in paths:
            new_path = path.replace("train018", split)
            os.system(f"mv {new_dir}/{path} {new_dir}/{new_path}")
        os.system("ls %s|xargs -I {} sed -i 's/train018/%s/' %s/{}" % (new_dir, split, new_dir))


if __name__ == "__main__":
    main()
