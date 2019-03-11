from pathlib import Path
import glob
import sys
import collections
from collections import defaultdict
from collections import OrderedDict
import os
import shutil
import math
import re
import pickle
import multiprocessing
from multiprocessing import Process, Queue, Pool
import itertools
from os.path import splitext
import mimetypes
from mimetypes import guess_all_extensions

# Non standard libs
import exiftool
import magic
from PIL import Image
import numpy as np


# GLOBAL STUFF

# We have to init mimetypes for some reason
mimetypes.init()


# TODO single source of truth for all these sets and maps
# It isn't that hard I just haven't wanted to do it yet.

extensions = set([
    '.aae',
    '.jpg',
    '.jpeg',
    '.png',
    '.heic',
    '.heif',
    '.mov',
    '.m4v',
    '.mp4',
    '.dng'
    ])

def in_extensions(f):
    input_ext = splitext(f)[1].lower()
    return any(ext in input_ext for ext in extensions)

can_dedup_ext = set([
    '.jpg',
    '.jpeg',
    '.png',
    # '.heic', # PIL doesn't support these and programs that edit them tend to output jpeg still
    # '.heif',
    ])

dis_f_type_map = OrderedDict([
    ('.aae', 'meta'),
    ('.png', 'screenshot'),
    ('.jpg', 'image'),
    ('.jpeg', 'image'),
    ('.m4v', 'video'),
    ('.mp4', 'video'),
    ('.mov', 'video'),
    ('.heic', 'image'),
    ('.heif', 'image'),
    ('.dng', 'image')
])

# How much I care about each file type
# lower means it will be sorted closer to front of list
dis_f_type_importance_map = {
    '.aae': 9,
    '.png': 8,
    '.jpeg': 7,
    '.jpg': 6,
    '.m4v': 5,
    '.mp4': 4,
    '.mov': 3,
    '.heic': 2,
    '.heif': 1,
    '.dng': 0,
}

# How much I care about each file type
# lower means it will be sorted closer to front of list
dis_compressor_importance_map = {
    'comp_unknown': 3,
    'H.264': 1,
    'HEVC': 0,
}

mov_tags = [
            'File:FileSize',
            'File:FileName',
            'File:FileModifyDate',
            'QuickTime:Model',
            'QuickTime:CreationDate',
            'QuickTime:ModifyDate',
            'QuickTime:ImageWidth',
            'QuickTime:ImageHeight',
            'QuickTime:CompressorName',
            'QuickTime:Duration',
            'QuickTime:ContentIdentifier'
            ]

pic_tags = [
            'File:FileSize',
            'File:FileName',
            'File:FileModifyDate',
            'EXIF:Model',
            'EXIF:CreateDate',
            'EXIF:SubSecTimeOriginal',
            'EXIF:ModifyDate',
            'EXIF:ImageWidth',
            'EXIF:ImageHeight',
            'File:FileType',
            ]

# A regular expression to find file names of a type that we expect
# and can isolate the important subset of.
f_name_re = re.compile(r'(img_E?\d{4})|(dsc_\d{4})|([a-zA-Z]{4}\d{4})', re.IGNORECASE)


# END GLOBAL STUFF

# I am having the caller pass in exactly what they want the extension to be
# that way this function isn't expected to parse which subset of the
# extension(s) are desired to be used
def get_proper_ext(expected_ext, full_path):
    return_ext = expected_ext
    # we use magic library to really use the binary information from the file
    # to guess at the mime type
    mime_type = magic.from_file(full_path, mime=True)
    if mime_type :
        # If we can find a mime type then we find out valid
        # extensions and fix them. For most things this will be a no op
        #
        # DNGs are a special case. They technically are tiff files but they should
        # have a mime of 'image/x-adobe-dng' but magic doesn't produce that.
        if mime_type == 'image/tiff' and expected_ext == '.dng' :
            pass
        # AAEs are also a special case. They technically are XML files but we
        # want to maintain the aae extension
        elif mime_type == 'text/xml' and expected_ext == '.aae' :
            pass
        else :
            valid_ext_list = guess_all_extensions(mime_type)
            if not expected_ext in valid_ext_list and len(valid_ext_list) > 0 :
                # Try to pick the extension we expect
                familiar_exts = extensions.intersection(set(valid_ext_list))
                if familiar_exts :
                    return_ext = familiar_exts.pop()
                else :
                    # Otherwise just pick a random extesion
                    return_ext = valid_ext_list[0].lower()
    return return_ext

def partition(pred, iterable):
    trues = []
    falses = []
    for item in iterable:
        if pred(item):
            trues.append(item)
        else:
            falses.append(item)
    return trues, falses

# TODO - to dedup raw images this explains what is necessary: https://stackoverflow.com/questions/2422050/raw-image-processing-in-python
# I am taking a stance to not do that as it is extremely rare that a raw would be edited in some way.
#
# Caller: Filter HEIC/HEIF files and add them back to the list after getting deduped list.
# They can't be deduped. Yet...
def dedup_images(image_list):
    debug_diff_list = [0]
    out_list = []
    diff_list = []
    out_list.append(image_list[0])
    size = (32, 32)
    im_thresh = 100
    first_image_f_name = image_list[0]['full_path']
    first_im = Image.open(first_image_f_name)
    first_im.thumbnail(size, Image.BILINEAR)
    # Make the thumbnail actually be a square image no matter the aspect ratio
    first_im_sq = Image.new('RGBA', size, (255, 255, 255, 0))
    first_im_sq.paste(
        first_im, (int((size[0] - first_im.size[0]) // 2), int((size[1] - first_im.size[1]) //2))
    )
    first_im_np = np.array(first_im_sq)
    diff_list.append(first_im_np)

    for image_meta in image_list[1:] :
        temp_im_f_name = image_meta['full_path']
        temp_im = Image.open(temp_im_f_name)
        temp_im.thumbnail(size, Image.BILINEAR)
        # Make the thumbnail actually be a square image no matter the aspect ratio
        temp_im_sq = Image.new('RGBA', size, (255, 255, 255, 0))
        temp_im_sq.paste(
            temp_im, (int((size[0] - temp_im.size[0]) // 2), int((size[1] - temp_im.size[1]) // 2))
        )
        temp_im_np = np.array(temp_im_sq)

        # If this image is different from all other images then keep it
        any_same = False
        for diff_im in diff_list :
            dif = np.sum(np.square(np.subtract(diff_im[:], temp_im_np[:])))
            if dif < im_thresh :
                any_same = True
                break

        # If this is a unique image add it to our out list and our check list
        if not any_same :
            out_list.append(image_meta)
            diff_list.append(temp_im_np)


    print('dedup_image_list:')
    print(image_list[0]['full_path'])
    print(debug_diff_list)
    print('end dedup----------')
    return out_list

from itertools import zip_longest
def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def chunker(iterable, n):
    return grouper(n, iterable)

def multiProcessReduce(inputIter):
    dup_dict_pic = defaultdict(list)
    dup_dict_mov = defaultdict(list)

    with exiftool.ExifTool() as et:
        for f in inputIter :
            # Our chunker inserts Nones to pad out the list.
            # We just ignore these.
            if f == None:
                continue
            proper_ext = splitext(f)[1].lower()
            # Now find out what the extenion for this type of file SHOULD be.
            # (Added due to Photos.app having JPG files that were MOV files.)
            #
            # TODO - make this an optional command line arg?
            # could also reduce dependency requirements
            proper_ext = get_proper_ext(proper_ext, f)

            # process movies and photos differently
            is_movie = '.mov' in proper_ext or \
                       '.mp4' in proper_ext

            regex_matched = False
            gen_tags = []
            if is_movie :
                gen_tags = mov_tags
            else :
                gen_tags = pic_tags
            metadata = et.get_tags(gen_tags, f)
            metadata['full_path'] = f
            #print(f)
            metadata['f_name_no_ext'] = metadata['File:FileName'].split('.')[0].lower()
            metadata['f_name_ext'] = proper_ext
            metadata['f_name_eight_char'] = metadata['f_name_no_ext'][:8]
            f_name_match = f_name_re.search(metadata['File:FileName'])
            real_name = metadata['f_name_no_ext']
            if f_name_match:
                real_name = f_name_match.group(0)
                regex_matched = True
            else :
                # what are the cases where we don't match the filename
                print('Filename doesn\'t match regex:')
                print(metadata['File:FileName'])
            metadata['f_name_real_name'] = real_name

            height = 0
            width = 0
            creation_time = "_Unknown"
            if is_movie :
                image_width_key = 'QuickTime:ImageWidth'
                image_height_key = 'QuickTime:ImageHeight'
                if not image_width_key in metadata:
                    metadata[image_width_key] = 0
                if not image_height_key in metadata:
                    metadata[image_height_key] = 0
                compressor_key = 'QuickTime:CompressorName'
                # TODO - does this indicate a corrupted file?
                if not compressor_key in metadata:
                    metadata[compressor_key] = 'comp_unknown'

                if 'QuickTime:CreationDate' in metadata :
                    creation_time = metadata['QuickTime:CreationDate']
                    if not regex_matched :
                        metadata['f_name_real_name'] = '_' + creation_time.replace(r':', '-').replace(' ', '_')
                dup_dict_mov[creation_time].append(metadata)

            else :
                # Hit an odd issue where pictures didn't have size info
                image_width_key = 'EXIF:ImageWidth'
                image_height_key = 'EXIF:ImageHeight'
                if not image_width_key in metadata:
                    metadata[image_width_key] = 0
                if not image_height_key in metadata:
                    metadata[image_height_key] = 0
                if 'EXIF:CreateDate' in metadata :
                    sub_sec = '000'
                    sub_sec_key = 'EXIF:SubSecTimeOriginal'
                    if sub_sec_key in metadata :
                        sub_sec = str(metadata[sub_sec_key])
                    creation_time = str(metadata['EXIF:CreateDate']) + '_' + sub_sec
                    if not regex_matched :
                        metadata['f_name_real_name'] = '_' + creation_time.replace(r':', '-').replace(' ', '_')
                dup_dict_pic[creation_time].append(metadata)
    return (dup_dict_pic, dup_dict_mov)

class FileLinkManager:
    def __init__(self, is_mock=True):
        self.dir_cache = set()
        self.f_cache = set()
        self.is_mock = is_mock
        self.cur_date_subdir = 'unset'
        self.cur_model_dir = 'Unset'
        self.cam_model_dict = {}
        self.top_outdir = ''
        self.top_dup_outdir = ''

    def set_top_dirs(self, outdir, dup_outdir) :
        self.top_outdir = outdir
        self.top_dup_outdir = dup_outdir

    def set_camera_model(self, cam_model) :
        if cam_model == '':
            self.cur_model_dir = 'Unknown'
            temp_model_dir = os.path.join(self.top_outdir, self.cur_model_dir)
            if not self.is_mock and not os.path.exists(temp_model_dir):
                os.mkdir(temp_model_dir)
            temp_dup_model_dir = os.path.join(self.top_dup_outdir, self.cur_model_dir)
            if not self.is_mock and not os.path.exists(temp_dup_model_dir):
                os.mkdir(temp_dup_model_dir)
        elif cam_model in self.cam_model_dict :
            self.cur_model_dir = self.cam_model_dict[cam_model]
        else:
            self.cur_model_dir = cam_model.replace(' ', '_')
            #self.cur_model_dir = slugify(self.cur_model_dir)
            self.cam_model_dict[cam_model] = self.cur_model_dir
            temp_model_dir = os.path.join(self.top_outdir, self.cur_model_dir)
            if not self.is_mock and not os.path.exists(temp_model_dir):
                os.mkdir(temp_model_dir)
            temp_dup_model_dir = os.path.join(self.top_dup_outdir, self.cur_model_dir)
            if not self.is_mock and not os.path.exists(temp_dup_model_dir):
                os.mkdir(temp_dup_model_dir)

    def set_date_subdir(self, date_sub) :
        self.cur_date_subdir = date_sub

    # There could be filename collisions in the same month directory for
    # various reasons. Make sure to handle it gracefully.
    def find_free_f_name(self, dir, dis_name) :
        new_file_name = os.path.join(dir, dis_name)
        dir_i = 2
        while new_file_name in self.f_cache :
            new_dir = os.path.join(dir, '_'+ str(dir_i))
            if not self.is_mock and not os.path.exists(new_dir):
                os.mkdir(new_dir)
            new_file_name = os.path.join(new_dir, dis_name)
            dir_i = dir_i +1
        return new_file_name

    def dir_check_make(self, input_dir):
        if not input_dir in self.dir_cache and not os.path.exists(input_dir):
            if not self.is_mock :
                os.mkdir(input_dir)
            self.dir_cache.add(input_dir)

    def get_file_name_dir(self, metadata, is_unknown=False, is_dup=False):
        cam_name = ''
        if 'EXIF:Model' in metadata :
            cam_name = metadata['EXIF:Model']
        elif 'QuickTime:Model' in metadata :
            cam_name = metadata['QuickTime:Model']
        self.set_camera_model(cam_name)
        dir = ''
        if is_dup :
            dir = os.path.join(self.top_dup_outdir, self.cur_model_dir)
        else :
            dir = os.path.join(self.top_outdir, self.cur_model_dir)

        if is_unknown == False :
            dir = os.path.join(dir, self.cur_date_subdir)
        else :
            # Gauntlet of guesses for making unknown time known
            unknown_time =  ''
            if 'EXIF:ModifyDate' in metadata :
                time = metadata['EXIF:ModifyDate']
                unknown_time = time[0:4] + '-' + time[5:7]
                print('EXIF:ModifyDate')
                print(time)
                print(unknown_time)
            elif 'QuickTime:ModifyDate' in metadata :
                time = metadata['QuickTime:ModifyDate']
                unknown_time = time[0:4] + '-' + time[5:7]
                print('QuickTime:ModifyDate')
                print(time)
                print(unknown_time)
            elif 'File:FileModifyDate' in metadata :
                time = metadata['File:FileModifyDate']
                unknown_time = time[0:4] + '-' + time[5:7]
                print('File:FileModifyDate')
                print(time)
                print(unknown_time)
            else :
                print('No modify date!')
                unknown_time = self.cur_date_subdir
            dir = os.path.join(dir, unknown_time)
        self.dir_check_make(dir)
        return dir

    # expectation: dir needs to be a string representation of a directory
    def link_files(self, metadata_list, is_unknown=False):
        for metadata in metadata_list:
            f_ext = metadata['f_name_ext']
            dis_name = metadata['f_name_real_name'] + f_ext
            dis_name = dis_name.upper()

            dir = self.get_file_name_dir(metadata, is_unknown=is_unknown, is_dup=False)
            new_file_name = self.find_free_f_name(dir, dis_name)

            if not self.is_mock :
                os.link(metadata['full_path'], new_file_name)
            else :
                print(new_file_name)
            self.f_cache.add(new_file_name)

    def link_dup_files(self, dup_i, metadata_list, is_unknown=False):

        for i, metadata in enumerate(metadata_list):
            f_ext = metadata['f_name_ext']
            dis_name = metadata['f_name_real_name'] + f_ext
            dis_name = dis_name.upper()

            dir = self.get_file_name_dir(metadata, is_unknown=is_unknown, is_dup=True)

            dup_f_name = 'DUP_' + str(dup_i) + "_" + str(i) + '--' + dis_name
            dup_f_name = dup_f_name.upper()
            new_file_name = self.find_free_f_name(dir, dup_f_name)

            if not self.is_mock :
                os.link(metadata['full_path'], new_file_name)
            else :
                print(new_file_name)
            self.f_cache.add(new_file_name)

############################ MAIN
def my_main_func() :

    this_run_mock = False

    # accept n path input arguments
    # they should be directories

    flm = FileLinkManager(is_mock=this_run_mock)



    # do an initial inspection of the directories to make sure they are valid
    for cmd_path in sys.argv[1:] :
        input_path = Path(cmd_path)
        if not input_path.is_dir():
            print(str(input_path) + ' is not a valid directory! Please try again.')



    dup_dict_pic_f = "dup_dict_pic.p"
    dup_dict_mov_f = "dup_dict_mov.p"


    dup_dict_pic = defaultdict(list)
    dup_dict_mov = defaultdict(list)

    if os.path.exists(dup_dict_pic_f) and os.path.exists(dup_dict_mov_f):
        dup_dict_pic = pickle.load( open( dup_dict_pic_f, "rb" ) )
        dup_dict_mov = pickle.load( open( dup_dict_mov_f, "rb" ) )
    else :
        num_proc = multiprocessing.cpu_count()-1
        # for large numbers of files
        #num_proc = 2

        # Mine exif data from all files
        # We use this with statement here to allow the exif mining to happen
        # before processing and then tear whatever memory it has used down.
        print(sys.argv[1:])
        for cmd_path in sys.argv[1:] :
                print('processing top level path: ')
                print(cmd_path)

                paths_reviewed, paths_ignored = partition(in_extensions,
                          glob.iglob(os.path.join(cmd_path, '**') ,
                                                            recursive=True) )
                if False :
                    print('Files reviewed:  ------------')
                    for f in paths_reviewed :
                        print(f)
                    print('Files reviewed END ------- ')

                if True :
                    print('Files NOT reviewed:  ------------')
                    for f in paths_ignored :
                        print(f)
                    print('Files NOT reviewed END ------- ')

                # Optional turn generator into list if we want to
                # pre-populate all the files before the next step.
                #
                # I think this option is better for spinning disks and
                # computers with lots of memory.
                paths_reviewed = list(paths_reviewed)
                total_files = len(paths_reviewed)
                chunker_size = 100
                print('num files: ')
                print(total_files)
                print('loop iterations:')
                print(math.ceil(total_files / chunker_size))

                # for i, test_paths in enumerate(chunker(paths_reviewed, chunker_size)) :
                #     print(i)
                #     print(len(test_paths))
                #     print(test_paths[0])

                 # start worker processes
                pool = Pool(processes=num_proc)
                # Use the multi Process function with the lines from the file
                # as they come in and hand them off to sub processes to be consumed.
                tuple_list = []
                for i, pool_tuple in \
                    enumerate(pool.imap_unordered(multiProcessReduce,
                             chunker(paths_reviewed, chunker_size))) :
                    tuple_list.append(pool_tuple)
                    print('iter ' + str(i))
                # discard pool forcefully
                pool.close()
                pool.join()

                print('len tuple_list:')
                print(len(tuple_list))

                for temp_dup_dict_pic, temp_dup_dict_mov in tuple_list :
                    for key in temp_dup_dict_pic:
                        dup_dict_pic[key].extend(temp_dup_dict_pic[key])
                    for key in temp_dup_dict_mov:
                        dup_dict_mov[key].extend(temp_dup_dict_mov[key])


        pickle.dump( dup_dict_pic, open( dup_dict_pic_f, "wb" ) )
        pickle.dump( dup_dict_mov, open( dup_dict_mov_f, "wb" ) )

    # detect the current working directory and print it
    start_output_dir = os.path.join(os.getcwd(), 'outdir')
    output_dir = start_output_dir
    # Unique output directory
    path_i = 1
    while os.path.exists(output_dir):
        output_dir = start_output_dir + '_' + str(path_i)
        path_i = path_i +1
    output_dir_dup = os.path.join(output_dir, 'duplicates')

    if not this_run_mock :
        # we now know that this check is trivial but I am leaving it in
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if not os.path.exists(output_dir_dup):
            os.mkdir(output_dir_dup)

    # Now we have dictionaries of all the files ordered by capture time.
    # Only run this for debugging purposes and
    # only if your dictionary sizes are very small.
    #
    if False:
        print('dup_dict_mov')
        print(dup_dict_mov)
        print(len(dup_dict_mov))

        print('dup_dict_pic')
        print(dup_dict_pic)
        print(len(dup_dict_pic))

    # iterate through all times
    # create a folder (if it doesn't exit) for YYYY-MM
    # for each file in the list of files :
    # Group together files with the same filename minus extension
    # for each of those files
    # group those by identical dimensions. aka  WxH (This seperates obvious edits)
    # for each item in each of those groups:
    # sort the group by the newer file type type
    # keep only one of the newer type (top file of the sorted list)
    # Write that file to the output directory or the output intent log

    # remove unknown time files and look at those seperately
    dup_pic_unknown = dup_dict_pic['_Unknown']
    #del(dup_dict_pic['Unknown'])
    print("Unknown time pics: ")
    for pic in dup_pic_unknown:
        f_ext = pic['f_name_ext']
        dis_name = pic['f_name_real_name'] + f_ext
        dis_name = dis_name.upper()
        print(dis_name)

    dup_mov_unknown = dup_dict_mov['_Unknown']
    #del(dup_dict_mov['Unknown'])
    print("Unknown time movs: ")
    for mov in dup_mov_unknown:
        f_ext = mov['f_name_ext']
        dis_name = mov['f_name_real_name'] + f_ext
        dis_name = dis_name.upper()
        print(dis_name)


    dup_list = [
      {
        'message': "Processed movs: ",
        'dup_dict': dup_dict_mov,
        'gen_lambda': lambda i: (dis_f_type_importance_map[i['f_name_ext']],
                                 dis_compressor_importance_map[i['QuickTime:CompressorName']],
                                 len(i['f_name_no_ext'])
                                 ),
        'f_prefix': 'mov_',
        'gen_filt_key': 'QuickTime:CompressorName',
        'height_key': 'QuickTime:ImageHeight',
        'width_key': 'QuickTime:ImageWidth',
        'dup_unknown': dup_mov_unknown,
      },
      {
        'message': "Processed pics: ",
        'dup_dict': dup_dict_pic,
        'gen_lambda': lambda i: (dis_f_type_importance_map[i['f_name_ext']],
                                len(i['f_name_no_ext'])
                                ),
        'f_prefix': 'pic_',
        'gen_filt_key': 'EXIF:ImageWidth',
        'height_key': 'EXIF:ImageHeight',
        'width_key': 'EXIF:ImageWidth',
        'dup_unknown': dup_pic_unknown,
      }
    ]

    flm.set_top_dirs(output_dir, output_dir_dup)


    for d_i in dup_list:
        print(d_i['message'])

        gen_prefix = d_i['f_prefix']
        gen_filt_key = d_i['gen_filt_key']
        f_prefix = d_i['f_prefix']
        gen_lambda = d_i['gen_lambda']
        dup_dict = d_i['dup_dict']
        temp_unknown = d_i['dup_unknown']
        height_key = d_i['height_key']
        width_key = d_i['width_key']
        dup_i = 0


        # temp_unknown_dup_dir = os.path.join(output_dir_dup, 'UnknownTime')
        # # scoping for variables inside
        # if True:
        #     disambiguate_dict = defaultdict(list)
        #     for f in f_list:
        #         f_ext = f['f_name_ext']
        #         dis_name = f['f_name_real_name'] + '_' + dis_f_type_map[f_ext]
        #         disambiguate_dict[dis_name].append(f)
        #
        #     for dis_name, d_f_list in disambiguate_dict.items() :
        #         flm.link_files(temp_unknown_dup_dir, temp_unknown)

        for time, f_list in dup_dict.items() :
            date_subdir = time[0:4] + '-' + time[5:7]
            flm.set_date_subdir(date_subdir)

            if len(f_list) > 1 :
                disambiguate_dict = defaultdict(list)
                for f in f_list:
                    f_ext = f['f_name_ext']
                    dis_name = f['f_name_real_name'] + '_' + \
                                str(f[width_key]) + 'x' + str(f[height_key]) + \
                                '_' + dis_f_type_map[f_ext]
                    disambiguate_dict[dis_name].append(f)

                for dis_name, d_f_list in disambiguate_dict.items() :
                    if len(d_f_list) > 1 :

                        print('-----' + date_subdir)
                        print('dis_name')
                        print(dis_name)


                        # if f_prefix is 'mov_' :
                        #     for f in d_f_list :
                        #         print(f['QuickTime:CompressorName'])

                        sorted_list = sorted(d_f_list, key = gen_lambda )


                        print(len(sorted_list))
                        for f in sorted_list :
                            print(f['f_name_no_ext'])

                        # NOTE:
                        # You CAN'T use a md5 or a EXIF content identifier or even
                        # EXIF Modification date because all of those
                        # get changed every time you download data via image capture.
                        #

                        # By default we just assume the first file of the
                        # sorted list is correct.
                        list_of_f_to_link = sorted_list[0:1]

                        # OLD sorting method that didn't work because things are perfect.
                        # used = set()
                        # filt_list = [x for x in sorted_list
                        #                 if x[gen_filt_key] not in used
                        #                 and (used.add(x[gen_filt_key]) or True)]
                        #
                        filt_list = sorted_list

                        # We have the ability to corsely inspect the binary contents
                        # and analyze them for differences. (For normal image types.)
                        if any(f['f_name_ext'] in can_dedup_ext
                                for f in sorted_list) :
                            # seperate the images that can be deduplicated from the ones that can't
                            (can_dedup, cant_dedup) = partition(lambda f: f['f_name_ext'] in can_dedup_ext, sorted_list)
                            # deduplicate those files
                            # if there are more than one
                            if can_dedup :
                                dedup_list = dedup_images(can_dedup)
                                # concatenate the two lists
                                cant_and_dedup = cant_dedup + dedup_list
                                cant_and_dedup = sorted(cant_and_dedup, key = gen_lambda )
                                # If the time is unknown then the duplicates
                                # are likely NOT duplicates. Just named the same.
                                #
                                # If all file types can be inspected then we will
                                # assume we have removed all duplicates.

                                if time == '_Unknown' or len(cant_dedup) == 0 :
                                    list_of_f_to_link = cant_and_dedup
                                    filt_list = []
                                # If the time is the same then the duplicates might
                                # be edits but we are going to put them in the
                                # duplicate folder anyway.
                                else :
                                    list_of_f_to_link = cant_and_dedup[0:1]
                                    filt_list = cant_and_dedup


                        print('Pre filter: ' + str(len(sorted_list)))
                        print('Post filter: '+ str(len(filt_list)))



                        print([x[gen_filt_key] for x in filt_list])
                        print([x['full_path'] for x in filt_list])
                        if len(filt_list) > 1 :

                            # choose the file with the shortest filename
                            flm.link_files(list_of_f_to_link,
                                                time == '_Unknown')
                            flm.link_dup_files(gen_prefix + str(dup_i),
                                                filt_list,
                                                time == '_Unknown')
                            dup_i = dup_i +1
                        else :
                            flm.link_files(list_of_f_to_link, time == '_Unknown')
                    else :
                        flm.link_files(d_f_list, time == '_Unknown')
            else :
                flm.link_files(f_list, time == '_Unknown')

    print(' -------------- Program ended without error')

if __name__ == '__main__':
    my_main_func()
