import time


def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() -
                                       startTime_for_tictoc) + " seconds.")
        return time.time() - startTime_for_tictoc
    else:
        print("Toc: start time not set")
        return None
