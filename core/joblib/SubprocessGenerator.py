import multiprocessing
import queue as Queue
import threading
import time
import traceback


_MESSAGE_TAG = '__dfl_subprocess_generator_v1__'


class SubprocessGenerator(object):
    
    @staticmethod
    def launch_thread(generator):
        try:
            generator._start()
        except BaseException as error:
            generator._start_error = error
        finally:
            generator._start_finished = True
        
    @staticmethod
    def start_in_parallel(generator_list, timeout=120.0):
        """
        Start list of generators in parallel
        """
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        launch_threads = []
        for generator in generator_list:
            thread = threading.Thread(target=SubprocessGenerator.launch_thread, args=(generator,) )
            thread.daemon = True
            thread.start()
            launch_threads.append(thread)

        while not all ([generator._is_started() for generator in generator_list]):
            failed = next(
                (
                    generator
                    for generator in generator_list
                    if generator._start_error is not None
                ),
                None,
            )
            if failed is None:
                failed = next(
                    (
                        generator
                        for generator in generator_list
                        if generator._start_finished and not generator._is_started()
                    ),
                    None,
                )
                if failed is not None:
                    failed._start_error = RuntimeError(
                        "The data-loader worker was closed before it started."
                    )
            if failed is not None:
                for generator in generator_list:
                    generator.close()
                for thread in launch_threads:
                    thread.join(timeout=1.0)
                # A sibling may have completed p.start() while the first close
                # was running. Close once more after starter threads settle.
                for generator in generator_list:
                    generator.close()
                raise RuntimeError(
                    "A data-loader worker failed to start."
                ) from failed._start_error
            if deadline is not None and time.monotonic() >= deadline:
                for generator in generator_list:
                    generator.close()
                for thread in launch_threads:
                    thread.join(timeout=1.0)
                for generator in generator_list:
                    generator.close()
                raise RuntimeError(
                    f"Timed out after {float(timeout):g} seconds while starting "
                    "data-loader workers."
                )
            time.sleep(0.005)
    
    def __init__(self, generator_func, user_param=None, prefetch=2,
                 start_now=True, result_timeout=120.0):
        super().__init__()
        self.prefetch = prefetch
        self.result_timeout = (
            None if result_timeout is None else max(0.1, float(result_timeout))
        )
        self.generator_func = generator_func
        self.user_param = user_param
        self.sc_queue = multiprocessing.Queue()
        self.cs_queue = multiprocessing.Queue()
        self.p = None
        self.closed = False
        self._start_error = None
        self._start_finished = False
        if start_now:
            self._start()

    def _start(self):
        if self.closed:
            return
        if self.p == None:
            user_param = self.user_param
            self.user_param = None
            p = multiprocessing.Process(target=self.process_func, args=(user_param,) )
            p.daemon = True
            p.start()
            self.p = p
            # close() can race a parallel p.start() while self.p is still None.
            # If that happened, take ownership here and immediately reap the
            # newly started process instead of leaking it.
            if self.closed:
                self.p = None
                self._stop_process(p)
            
    def _is_started(self):
        return self.p is not None
        
    def process_func(self, user_param):
        try:
            self.generator_func = self.generator_func(user_param)
            while True:
                while self.prefetch > -1:
                    try:
                        gen_data = next(self.generator_func)
                    except StopIteration:
                        self._send_message('stop')
                        return
                    self._send_message('data', gen_data)
                    self.prefetch -= 1
                self.sc_queue.get()
                self.prefetch += 1
        except BaseException as error:
            self._send_message(
                'error',
                {
                    'type': f'{type(error).__module__}.{type(error).__name__}',
                    'message': str(error),
                    'traceback': traceback.format_exc(),
                },
            )

    def _send_message(self, kind, payload=None):
        try:
            self.cs_queue.put((_MESSAGE_TAG, kind, payload))
        except BaseException:
            # The parent may already have closed its queue while shutting down.
            pass

    def __iter__(self):
        return self

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['p']
        return self_dict

    def __next__(self):
        if self.closed:
            raise StopIteration()
        self._start()
        deadline = (
            None if self.result_timeout is None
            else time.monotonic() + self.result_timeout
        )

        while True:
            process = self.p
            if process is None:
                self.close()
                raise RuntimeError('The data-loader worker did not start.')

            wait_seconds = 1.0
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timeout = self.result_timeout
                    self.close()
                    raise RuntimeError(
                        f'Timed out after {timeout:g} seconds waiting for a '
                        'data-loader worker batch.'
                    )
                wait_seconds = min(wait_seconds, remaining)

            try:
                message = self.cs_queue.get(timeout=wait_seconds)
            except Queue.Empty:
                if not process.is_alive():
                    process.join(timeout=0.1)
                    exit_code = process.exitcode
                    self.close()
                    raise RuntimeError(
                        'The data-loader worker exited without returning a '
                        f'batch (exit code {exit_code}).'
                    )
                continue

            if (
                not isinstance(message, tuple)
                or len(message) != 3
                or message[0] != _MESSAGE_TAG
            ):
                self.close()
                raise RuntimeError(
                    'The data-loader worker returned an invalid message.'
                )

            _, kind, payload = message
            if kind == 'stop':
                self.close()
                raise StopIteration()
            if kind == 'error':
                error_type = payload.get('type', 'unknown error')
                error_message = payload.get('message', '')
                child_traceback = payload.get('traceback', '')
                self.close()
                raise RuntimeError(
                    f'Data-loader worker failed with {error_type}: '
                    f'{error_message}\n{child_traceback}'
                )
            if kind != 'data':
                self.close()
                raise RuntimeError(
                    f'The data-loader worker returned unknown message kind: {kind!r}.'
                )

            self.sc_queue.put(1)
            return payload

    def close(self):
        self.closed = True
        p = self.p
        self.p = None
        if p is not None:
            self._stop_process(p)
        for process_queue in (self.sc_queue, self.cs_queue):
            try:
                process_queue.close()
                process_queue.cancel_join_thread()
            except Exception:
                pass

    @staticmethod
    def _stop_process(process):
        try:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1.0)
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
