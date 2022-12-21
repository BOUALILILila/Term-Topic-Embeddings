import os
import requests
from tqdm.autonotebook import tqdm
from typing import Callable, Optional

def keep_vocab_item(token: int, count: int, min_count: int, trim_rule: Optional[Callable]=None) -> bool:
    """
    Parameters
    ----------
    token : int
        Input token id.
    count : int
        Number of times that word appeared in a corpus.
    min_count : int
        Discard words with frequency smaller than this.
    trim_rule : function, optional
        Custom function to decide whether to keep or discard this word.
        If a custom `trim_rule` is not specified, the default behaviour is simply `count >= min_count`.
    Returns
    -------
    bool
        True if `token` should stay, False otherwise.
    """
    default_res = count >= min_count

    if trim_rule is None:
        return default_res
    else:
        keep = trim_rule(token, count, min_count)
        return keep

def http_get(url, path):
    """
    Downloads a URL to a given path on disc
    """
    if os.path.dirname(path) != '':
        os.makedirs(os.path.dirname(path), exist_ok=True)

    req = requests.get(url, stream=True)
    if req.status_code != 200:
        print("Exception when trying to download {}. Response {}".format(url, req.status_code), file=sys.stderr)
        req.raise_for_status()
        return

    download_filepath = path+"_part"
    with open(download_filepath, "wb") as file_binary:
        content_length = req.headers.get('Content-Length')
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total, unit_scale=True)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                progress.update(len(chunk))
                file_binary.write(chunk)

    os.rename(download_filepath, path)
    progress.close()
