# iphone-photo-dedup
Fixing the hell that has become managing iPhone photo and video downloads. Particularly solving the issue of re-downloads with HEIC and JPG when the os support transitioned.

The goal is to keep the original file in whatever format we think that the original was in when captured. (All derivative works are lesser in quality.) Simultaneously, we organize and deduplicate all the files that we can.

I meant to keep this a tight little script but it kept growing as I wanted it to handle more and more corner cases. Now that it is non-trivial and somewhat robust I wanted to share it with others. (I searched for a while for a similar script that could really help me to remove duplicates around this special use case. None satisfied my needs.)

# Guidelines:
1. Do no harm. AKA-> NEVER modify input files
2. All input directories need to be on the same drive. That drive must support hard links.
*Note: We are creating a curated view of what is already there using the filesystem as a DB and the OS file explorer as a visualizer for both files and contained data.*
3. Only new file data created are a pickled cache of temporary data to speed up subsequent runs
4. Things are designed to keep all processors maximally busy. (Sometimes that can mean worse performance for spinning disks and large number of CPUs.)
5. I expect that you can delete everything in the duplicates directory (other than the unknown time directory) with little risk of losing anything

I need to turn these TODOs into tickets.
# TODOS:
1. Allow people to keep sidecar JPGs with DNGs (By default I decided to keep them for now. Still want a flag for this.)
2. Put all of the files that don't have the correct metadata in a separate folder. Not normal and not duplicates.
3. MD5s and content identifiers don't work like you might expect. Figure out why? (Download of same data and same content have different information.)
4. Perhaps in place of #3, Allow a sanity check of content of images with a std deviation of difference (TODONE!)
5. AAE files? (XML files too?) Are these useful for anything? They seem to be proprietary, nothing supports them, and I can't tell if this will ever change. (TODONE!) I am now bringing along AAE files.
6. Add a first pass scan and use the imghdr library to inspect the file type and change it to the correct one before the de-duplication? (TODONE!) Using magic lib for this.
BTW I ran into the issue in #6 when getting master files from an Apple Photos library where it had a bunch of mov files listed as .jpg for no reason...
Don't change the actual input filename just modify the f_name_ext to be the correct one for de-dup purposes.
7. Create an ascii explanation of all the files that get created and where they go.
8. Separate classes and functions into files.
9. Determine disk media and run with 1 process or N-1 depending if it is a spinning disk or SSD respectively ?
10. Make an options where, in place of #4 using a cached thumbnail of every image that is 32x32 use the whole image data and change the threshold to a percentage based value.
11. Make it easier to review files that were ignored?
12. Expand supported file types to include other raw files and Tiffs that might have been downloaded to an iOS device
13. Create top level iphone name directories above date directories
14. Dedup HEIC files.
15. Add modified time as the last sorting parameter. (Oldest wins.)

How to handle the case where another jpg was created from a heic and
that has the same size but is not newer than the heic?

# Random Discoveries
- Microsoft Photo Viewer will change the size of the image when it downloads files. It appears to be always slightly larger making the original one always get selected. (If you see unexpected duplicates get through check the content creator.) All this to say that if you used Windows to download files at any point those will be different sizes and have different EXIF info than if you did it with a mac. 
-

# Output guidelines
```
CWD -- \                    # current working directory that the program is run in
      ---- dup_dict_mov.p  # list of dictionaries with movie information
      ---- dup_dict_pic.p  # list of dictionaries with picture information
      ---- outdir          # this is where the output data goes! (all hard links) (we handle name collisions with this dir name)
```
