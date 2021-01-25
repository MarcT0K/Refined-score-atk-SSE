"""Everything needed to extract Enron and Apache email datasets.
"""

import email
import glob
import mailbox
import os
import random

from typing import Tuple

import colorlog
import pandas as pd
import tqdm


logger = colorlog.getLogger("QueRyvolution")


def split_df(dframe, frac=0.5):
    first_split = dframe.sample(frac=frac)
    second_split = dframe.drop(first_split.index)
    return first_split, second_split


def get_body_from_enron_email(mail):
    """To get the content from raw email"""
    msg = email.message_from_string(mail)
    parts = []
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            parts.append(part.get_payload())
    return "".join(parts)


def get_body_from_mboxmsg(msg):
    parts = []
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            parts.append(part.get_payload())
    body = "".join(parts)
    body = body.split("To unsubscribe")[0]
    return body


def extract_2_enron_mailboxes(
    maildir_directory="../maildir/"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = os.path.expanduser(maildir_directory)

    eligible_users = [
        "baughman-d",
        "gay-r",
        "heard-m",
        "hendrickson-s",
        "linder-e",
        "arnold-j",
    ]

    res = []

    for mailbox_path in eligible_users:
        mail_contents = []
        mails = []
        for mailfile_path in glob.glob(f"{path}/{mailbox_path}/*/*"):
            try:
                with open(mailfile_path, "r") as mailfile:
                    raw_mail = mailfile.read()
                    mail_contents.append(get_body_from_enron_email(raw_mail))
                    mails.append(mailfile_path)
            except IsADirectoryError:
                pass
        res.append(pd.DataFrame(data={"filename": mails, "mail_body": mail_contents}))

    return pd.concat(res[:-1]), res[-1]


def extract_sent_mail_contents(maildir_directory="../maildir/") -> pd.DataFrame:
    path = os.path.expanduser(maildir_directory)
    mails = glob.glob(f"{path}/*/_sent_mail/*")

    mail_contents = []
    for mailfile_path in tqdm.tqdm(iterable=mails, desc="Reading the emails"):
        with open(mailfile_path, "r") as mailfile:
            raw_mail = mailfile.read()
            mail_contents.append(get_body_from_enron_email(raw_mail))

    return pd.DataFrame(data={"filename": mails, "mail_body": mail_contents})


def extract_apache_ml(maildir_directory="../apache_ml/") -> pd.DataFrame:
    path = os.path.expanduser(maildir_directory)
    mails = glob.glob(f"{path}/*")
    mail_contents = []
    mail_ids = []
    for mbox_path in tqdm.tqdm(iterable=mails, desc="Reading the emails"):
        for mail in mailbox.mbox(mbox_path):
            mail_content = get_body_from_mboxmsg(mail)
            mail_contents.append(mail_content)
            mail_ids.append(mail["Message-ID"])
    return pd.DataFrame(data={"filename": mail_ids, "mail_body": mail_contents})
