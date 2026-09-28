Changelog
=========

One file per release, named for the version: v0.7.0.txt. From v0.7.0 on, /news
shows each one in a single Discord embed, so:

  * The first line is the title: the version and the date.
  * The rest is Discord markdown -- **bold** group labels, "- " bullets, no hard
    line wraps -- and stays under 4,000 characters (tests/test_news.py).
  * A one-line summary, then Added / Changed / Fixed / Performance. Omit a group
    with nothing in it rather than writing "none".

Entries name the command or the setting a user would touch, not the function
that changed, in a line or two each. The code documents how and why; this says
what a user will notice. The files before v0.7.0 are longer, plain-text and not
shown by /news.

Announcement copy derived from an entry lives beside it as
v<version>-announcement.txt, so the post that went out can be diffed against
what actually shipped. It may run to two Discord messages: keep each part under
2,000 characters and split at a heading.
