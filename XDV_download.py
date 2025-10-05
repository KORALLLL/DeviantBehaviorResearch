from huggingface_hub import snapshot_download
import time
import tqdm
import argparse

def download(rootdir: str, timeout: int, test: bool):
    error_429 = True
    pattern = "data/video/test*" if test else "data/video/*"
    while error_429:
        try:
            snapshot_download(
                repo_id="jherng/xd-violence",
                repo_type="dataset",
                revision="main",
                allow_patterns=[pattern],
                ignore_patterns=[
                    "data/video/test_videos/Before.Sunset.2004__#01-14-30_01-16-51_label_A.mp4",
                ],
                cache_dir="./local_cache",
                # force_download=True,
                # resume_download=False,
                local_dir=rootdir,
            )
            error_429 = False
            print("All files downloaded!")
        except Exception as e:
            print(e)
            print(f"Wait for {timeout} mins")
            for i in tqdm.tqdm(range(timeout * 60), desc="time"):
                time.sleep(1)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download dataset from Hugging Face Hub')
    parser.add_argument('--rootdir', type=str, required=True, help='Root directory for download')
    parser.add_argument('--timeout', type=int, default=6, help='Timeout in minutes (default: 6)')
    parser.add_argument('--test_only', action='store_true', help='Download only test part')
    
    args = parser.parse_args()
    download(rootdir=args.rootdir, timeout=args.timeout, test=args.test_only)