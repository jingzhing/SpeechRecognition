import argparse

from speech_digit.data import download_and_prepare_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Redownload and rebuild data/filtered")
    args = parser.parse_args()
    download_and_prepare_dataset(force=args.force)


if __name__ == "__main__":
    main()
