import queue
import subprocess
import threading
import numpy as np
import ffmpeg
from pathlib import Path
from core import pathex
from core.interact import interact as io

def extract_video(input_file, output_dir, output_ext=None, fps=None):
    input_file_path = Path(input_file)
    output_path = Path(output_dir)

    if not output_path.exists():
        output_path.mkdir(exist_ok=True)


    if input_file_path.suffix == '.*':
        input_file_path = pathex.get_first_file_by_stem (input_file_path.parent, input_file_path.stem)
    else:
        if not input_file_path.exists():
            input_file_path = None

    if input_file_path is None:
        io.log_err("input_file not found.")
        return

    if fps is None:
        fps = io.input_int ("Enter FPS", 0, help_message="How many frames of every second of the video will be extracted. 0 - full fps")

    if output_ext is None:
        output_ext = io.input_str ("Output image format", "png", ["png","jpg"], help_message="png is lossless, but extraction is x10 slower for HDD, requires x10 more disk space than jpg.")

    for filename in pathex.get_image_paths (output_path, ['.'+output_ext]):
        Path(filename).unlink()

    job = ffmpeg.input(str(input_file_path))

    kwargs = {'pix_fmt': 'rgb24'}
    if fps != 0:
        kwargs.update ({'r':str(fps)})

    if output_ext == 'jpg':
        kwargs.update ({'q:v':'2'}) #highest quality for jpg

    job = job.output( str (output_path / ('%5d.'+output_ext)), **kwargs )

    try:
        job = job.run()
    except:
        io.log_err ("ffmpeg fail, job commandline:" + str(job.compile()) )

def cut_video ( input_file, from_time=None, to_time=None, audio_track_id=None, bitrate=None):
    input_file_path = Path(input_file)
    if input_file_path is None:
        io.log_err("input_file not found.")
        return

    output_file_path = input_file_path.parent / (input_file_path.stem + "_cut" + input_file_path.suffix)

    if from_time is None:
        from_time = io.input_str ("From time", "00:00:00.000")

    if to_time is None:
        to_time = io.input_str ("To time", "00:00:00.000")

    if audio_track_id is None:
        audio_track_id = io.input_int ("Specify audio track id.", 0)

    if bitrate is None:
        bitrate = max (1, io.input_int ("Bitrate of output file in MB/s", 25) )

    kwargs = {"c:v": "libx264",
              "b:v": "%dM" %(bitrate),
              "pix_fmt": "yuv420p",
             }

    job = ffmpeg.input(str(input_file_path), ss=from_time, to=to_time)

    job_v = job['v:0']
    job_a = job['a:' + str(audio_track_id) + '?' ]

    job = ffmpeg.output(job_v, job_a, str(output_file_path), **kwargs).overwrite_output()

    try:
        job = job.run()
    except:
        io.log_err ("ffmpeg fail, job commandline:" + str(job.compile()) )

def denoise_image_sequence( input_dir, ext=None, factor=None ):
    input_path = Path(input_dir)

    if not input_path.exists():
        io.log_err("input_dir not found.")
        return

    image_paths = [ Path(filepath) for filepath in pathex.get_image_paths(input_path) ]

    # Check extension of all images
    image_paths_suffix = None
    for filepath in image_paths:
        if image_paths_suffix is None:
            image_paths_suffix = filepath.suffix
        else:
            if filepath.suffix != image_paths_suffix:
                io.log_err(f"All images in {input_path.name} should be with the same extension.")
                return

    if factor is None:
        factor = np.clip ( io.input_int ("Denoise factor?", 7, add_info="1-20"), 1, 20 )

    # Rename to temporary filenames
    for i,filepath in io.progress_bar_generator( enumerate(image_paths), "Renaming", leave=False):
        src = filepath
        dst = filepath.parent / ( f'{i+1:06}_{filepath.name}' )
        try:
            src.rename (dst)
        except:
            io.log_error ('fail to rename %s' % (src.name) )
            return

    # Rename to sequental filenames
    for i,filepath in io.progress_bar_generator( enumerate(image_paths), "Renaming", leave=False):

        src = filepath.parent / ( f'{i+1:06}_{filepath.name}' )
        dst = filepath.parent / ( f'{i+1:06}{filepath.suffix}' )
        try:
            src.rename (dst)
        except:
            io.log_error ('fail to rename %s' % (src.name) )
            return

    # Process image sequence in ffmpeg
    kwargs = {}
    if image_paths_suffix == '.jpg':
        kwargs.update ({'q:v':'2'})

    job = ( ffmpeg
            .input(str ( input_path / ('%6d'+image_paths_suffix) ) )
            .filter("hqdn3d", factor, factor, 5,5)
            .output(str ( input_path / ('%6d'+image_paths_suffix) ), **kwargs )
           )

    try:
        job = job.run()
    except:
        io.log_err ("ffmpeg fail, job commandline:" + str(job.compile()) )

    # Rename to temporary filenames
    for i,filepath in io.progress_bar_generator( enumerate(image_paths), "Renaming", leave=False):
        src = filepath.parent / ( f'{i+1:06}{filepath.suffix}' )
        dst = filepath.parent / ( f'{i+1:06}_{filepath.name}' )
        try:
            src.rename (dst)
        except:
            io.log_error ('fail to rename %s' % (src.name) )
            return

    # Rename to initial filenames
    for i,filepath in io.progress_bar_generator( enumerate(image_paths), "Renaming", leave=False):
        src = filepath.parent / ( f'{i+1:06}_{filepath.name}' )
        dst = filepath

        try:
            src.rename (dst)
        except:
            io.log_error ('fail to rename %s' % (src.name) )
            return

_NVENC_AVAILABLE = None

def _nvenc_available():
    """One-time probe: can this machine's GPU driver actually initialize h264_nvenc?"""
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is None:
        try:
            result = subprocess.run(
                ['ffmpeg', '-hide_banner', '-v', 'error',
                 '-f', 'lavfi', '-i', 'color=black:size=256x256:rate=25:duration=0.2',
                 '-c:v', 'h264_nvenc', '-f', 'null', '-'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
            _NVENC_AVAILABLE = result.returncode == 0
        except Exception:
            _NVENC_AVAILABLE = False
    return _NVENC_AVAILABLE

def _encode_frames_via_pipe(job, input_image_paths):
    """Feed image files into an ffmpeg pipe job, reading ahead on a worker
    thread so disk I/O overlaps with encoding. Returns the ffmpeg exit code."""
    prefetch = queue.Queue(maxsize=16)

    def _read_frames():
        try:
            for image_path in input_image_paths:
                with open (image_path, "rb") as f:
                    prefetch.put (f.read())
        except Exception as error:
            prefetch.put (error)
            return
        prefetch.put (None)

    job_run = job.run_async(pipe_stdin=True)
    reader_thread = threading.Thread(target=_read_frames, daemon=True)
    reader_thread.start()
    try:
        while True:
            item = prefetch.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            job_run.stdin.write (item)
    except OSError:
        # ffmpeg died mid-stream (e.g. the encoder failed to initialize);
        # drain the reader, then report the exit code to the caller.
        while True:
            item = prefetch.get()
            if item is None or isinstance(item, Exception):
                break
    try:
        job_run.stdin.close()
    except OSError:
        pass
    job_run.wait()
    return job_run.returncode

def video_from_sequence( input_dir, output_file, reference_file=None, ext=None, fps=None, bitrate=None, include_audio=False, lossless=None, preset=None ):
    input_path = Path(input_dir)
    output_file_path = Path(output_file)
    reference_file_path = Path(reference_file) if reference_file is not None else None

    if not input_path.exists():
        io.log_err("input_dir not found.")
        return

    if not output_file_path.parent.exists():
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

    out_ext = output_file_path.suffix

    if ext is None:
        ext = io.input_str ("Input image format (extension)", "png")

    if lossless is None:
        lossless = io.input_bool ("Use lossless codec", False)

    video_id = None
    audio_id = None
    ref_in_a = None
    if reference_file_path is not None:
        if reference_file_path.suffix == '.*':
            reference_file_path = pathex.get_first_file_by_stem (reference_file_path.parent, reference_file_path.stem)
        else:
            if not reference_file_path.exists():
                reference_file_path = None

        if reference_file_path is None:
            io.log_err("reference_file not found.")
            return

        #probing reference file
        probe = ffmpeg.probe (str(reference_file_path))

        #getting first video and audio streams id with fps
        for stream in probe['streams']:
            if video_id is None and stream['codec_type'] == 'video':
                video_id = stream['index']
                fps = stream['r_frame_rate']

            if audio_id is None and stream['codec_type'] == 'audio':
                audio_id = stream['index']

        if audio_id is not None:
            #has audio track
            ref_in_a = ffmpeg.input (str(reference_file_path))[str(audio_id)]

    if fps is None:
        #if fps not specified and not overwritten by reference-file
        fps = max (1, io.input_int ("Enter FPS", 25) )

    if not lossless and bitrate is None:
        bitrate = max (1, io.input_int ("Bitrate of output file in MB/s", 16) )

    input_image_paths = pathex.get_image_paths(input_path)
    if len(input_image_paths) == 0:
        io.log_err(f"No frames found in {input_path}.")
        return

    i_in = ffmpeg.input('pipe:', format='image2pipe', r=fps, thread_queue_size=64)

    output_args = [i_in]

    if include_audio and ref_in_a is not None:
        output_args += [ref_in_a]

    output_args += [str (output_file_path)]

    audio_kwargs = {}
    if include_audio and ref_in_a is not None:
        audio_kwargs = {"c:a": "aac",
                        "b:a": "192k",
                        "ar" : "48000",
                        "strict": "experimental"
                        }

    def _video_kwargs(codec):
        if lossless:
            # crf 0 output is bit-identical at any x264 preset; veryfast encodes
            # several times faster than the default (medium) at a file-size cost.
            return {"c:v": "libx264",
                    "crf": "0",
                    "preset": preset or "veryfast",
                    "pix_fmt": "yuv420p",
                    }
        if codec == 'h264_nvenc':
            nvenc_presets = ('p1','p2','p3','p4','p5','p6','p7','slow','medium','fast','hq','hp')
            return {"c:v": "h264_nvenc",
                    "b:v": "%dM" %(bitrate),
                    "preset": preset if preset in nvenc_presets else "p5",
                    "pix_fmt": "yuv420p",
                    }
        return {"c:v": "libx264",
                "b:v": "%dM" %(bitrate),
                "preset": preset or "medium",
                "pix_fmt": "yuv420p",
                }

    # Bitrate encodes go through NVENC when the GPU driver supports it (5-10x
    # faster than libx264); lossless stays on libx264 for exact crf-0 output.
    codecs_to_try = ['libx264']
    if not lossless and _nvenc_available():
        codecs_to_try.insert(0, 'h264_nvenc')

    for codec in codecs_to_try:
        output_kwargs = { **_video_kwargs(codec), **audio_kwargs }
        job = ( ffmpeg.output(*output_args, **output_kwargs).overwrite_output() )
        if codec == 'h264_nvenc':
            io.log_info ("Encoding with NVENC hardware encoder.")
        try:
            returncode = _encode_frames_via_pipe(job, input_image_paths)
        except Exception:
            io.log_err ("ffmpeg fail, job commandline:" + str(job.compile()) )
            return
        if returncode == 0:
            return
        if codec == 'h264_nvenc':
            io.log_info ("NVENC encode failed, falling back to libx264.")
        else:
            io.log_err (f"ffmpeg exited with code {returncode}, job commandline:" + str(job.compile()) )
